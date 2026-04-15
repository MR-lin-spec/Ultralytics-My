import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class StripConv(nn.Module):
    """条纹卷积：针对铁路细长结构设计"""
    def __init__(self, c1, c2, k=7):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2 // 2, (1, k), padding=(0, k // 2), bias=False)
        self.cv2 = nn.Conv2d(c1, c2 // 2, (k, 1), padding=(k // 2, 0), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        return self.act(self.bn(torch.cat([x1, x2], dim=1)))


class CBAM(nn.Module):
    """轻量级注意力模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = self.sigmoid(avg_out + max_out)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * sa


class ASPP(nn.Module):
    """空洞空间金字塔池化"""
    def __init__(self, c1, c2, rates=[1, 3, 5]):
        super().__init__()
        self.stages = nn.ModuleList()
        self.stages.append(nn.Sequential(
            nn.Conv2d(c1, c2 // 4, 1, bias=False),
            nn.BatchNorm2d(c2 // 4),
            nn.SiLU()
        ))
        for rate in rates:
            self.stages.append(nn.Sequential(
                nn.Conv2d(c1, c2 // 4, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(c2 // 4),
                nn.SiLU()
            ))
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c2 // 4, 1, bias=False),
            nn.BatchNorm2d(c2 // 4),
            nn.SiLU()
        )
        self.project = Conv(c2, c2, 1, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        feats = []
        for stage in self.stages:
            feats.append(stage(x))
        global_feat = self.global_avg(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        feats.append(global_feat)
        out = torch.cat(feats, dim=1)
        return self.project(out)


class RailSPPELAN(nn.Module):
    """
    改进版SPPELAN：修复通道数计算错误
    关键修复：确保所有分支输出通道数与cv5的输入通道数匹配
    """
    def __init__(self, c1, c2, c3, k=5, use_strip=True, use_aspp=False, use_attention=True):
        super().__init__()
        self.c = c3
        self.use_strip = use_strip
        self.use_aspp = use_aspp
        self.use_attention = use_attention
        
        # 初始降维：c1 -> c3
        self.cv1 = Conv(c1, c3, 1, 1)
        
        # SPP分支：每个分支输出都是 c3（因为MaxPool不改变通道数）
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        
        # 计算分支数量（每个分支输出c3通道）
        branch_count = 4  # cv1 + cv2 + cv3 + cv4
        
        # 条纹卷积分支：输入c3，输出c3（保持通道一致）
        if use_strip:
            self.strip_conv = StripConv(c3, c3)
            branch_count += 1
        
        # ASPP分支：输入c3，输出c3
        if use_aspp:
            self.aspp = ASPP(c3, c3)
            branch_count += 1
        
        # 注意力模块（通道数不变）
        if use_attention:
            self.attention = CBAM(c2)
        
        # 最终融合：branch_count * c3 -> c2
        self.cv5 = Conv(branch_count * c3, c2, 1, 1)
        
        # 残差连接
        self.shortcut = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        # 初始卷积：x -> [B, c3, H, W]
        x1 = self.cv1(x)
        y = [x1]
        
        # SPP分支：每个都是 [B, c3, H, W]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        
        # 条纹卷积分支：[B, c3, H, W]
        if self.use_strip:
            y.append(self.strip_conv(y[0]))
        
        # ASPP分支：[B, c3, H, W]
        if self.use_aspp:
            y.append(self.aspp(y[0]))
        
        # 特征融合
        out = torch.cat(y, dim=1)  # [B, branch_count*c3, H, W]
        out = self.cv5(out)        # [B, c2, H, W]
        
        # 注意力加权
        if self.use_attention:
            out = self.attention(out)
        
        return out + residual


# ==================== 验证测试 ====================

if __name__ == "__main__":
    def test_model(model, input_shape, name):
        x = torch.randn(input_shape)
        try:
            out = model(x)
            print(f"✓ {name}")
            print(f"  输入: {x.shape} -> 输出: {out.shape}")
            assert x.shape[2:] == out.shape[2:], "空间尺寸不一致！"
            return True
        except Exception as e:
            print(f"✗ {name}: {e}")
            return False

    print("=" * 60)
    print("测试不同配置下的输入输出尺寸一致性")
    print("=" * 60)
    
    configs = [
        (256, 256, 128, True, False, True),
        (512, 512, 256, True, True, True),
        (128, 128, 64, False, False, True),
        (256, 128, 64, True, False, False),
    ]
    
    for c1, c2, c3, strip, aspp, attn in configs:
        print(f"\n配置: c1={c1}, c2={c2}, c3={c3}, strip={strip}, aspp={aspp}, attn={attn}")
        model = ImprovedSPPELAN(c1, c2, c3, use_strip=strip, use_aspp=aspp, use_attention=attn)
        test_model(model, (4, c1, 20, 20), "测试1: 20x20")
        test_model(model, (4, c1, 80, 80), "测试2: 80x80")
    
    # 对比原始版本
    print("\n" + "=" * 60)
    print("与原始SPPELAN对比")
    print("=" * 60)
    
    class OriginalSPPELAN(nn.Module):
        def __init__(self, c1, c2, c3, k=5):
            super().__init__()
            self.c = c3
            self.cv1 = Conv(c1, c3, 1, 1)
            self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            self.cv5 = Conv(4 * c3, c2, 1, 1)
        
        def forward(self, x):
            y = [self.cv1(x)]
            y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
            return self.cv5(torch.cat(y, 1))
    
    orig = OriginalSPPELAN(256, 256, 128)
    improved = ImprovedSPPELAN(256, 256, 128, use_strip=True, use_aspp=False, use_attention=True)
    
    x = torch.randn(2, 256, 40, 40)
    print(f"\n原始SPPELAN: {x.shape} -> {orig(x).shape}")
    print(f"改进版SPPELAN: {x.shape} -> {improved(x).shape}")
    
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n参数量对比:")
    print(f"  原始: {count_params(orig):,}")
    print(f"  改进: {count_params(improved):,}")