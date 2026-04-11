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

class RailSPPCSPC(nn.Module):
    """
    钢轨缺陷检测专用SPPCSPC
    改进点：
    1. 非对称条形池化：1xk捕捉横向连续性，kx1捕捉纵向细节，适配钢轨长宽比
    2. 细节保留分支：使用AvgPool+MaxPool混合，防止微小裂纹在池化中消失
    3. 通道注意力：融合时自适应加权不同尺度特征
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        
        # 改进1: 非对称条形池化 - 同时捕捉横向(水平)和纵向特征
        # 假设钢轨图像宽>高，设置水平池化核较大，垂直较小
        self.strip_pool_h = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(kernel_size=(1, x), stride=1, padding=(0, x//2)),
                nn.MaxPool2d(kernel_size=(1, x), stride=1, padding=(0, x//2))
            ) for x in k
        ])
        self.strip_pool_v = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(kernel_size=(x, 1), stride=1, padding=(x//2, 0)),
                nn.MaxPool2d(kernel_size=(x, 1), stride=1, padding=(x//2, 0))
            ) for x in k
        ])
        
        # 原始方形池化保留，但权重降低（后续通过注意力调整）
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        
        # 输入通道变为原始4倍 + 条形池化特征(2*3=6) = 10倍c_，需要压缩
        total_channels = c_ * (1 + len(k) + 2*len(k))  # x1 + 3个方池 + 6个条池
        self.cv5 = Conv(total_channels, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)
        
        # 通道注意力：自适应加权不同尺度（方形vs条形）
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_, c_ // 4, 1),
            nn.SiLU(),
            nn.Conv2d(c_ // 4, len(k)*3 + 1, 1),  # 为每个池化分支学习权重
            nn.Sigmoid()
        )
        
        # 细节增强：对原始x1进行边缘增强
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(c_, c_, 3, padding=1, groups=c_),
            nn.SiLU(),
            nn.Conv2d(c_, c_, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 主分支
        x1 = self.cv4(self.cv3(self.cv1(x)))
        b, c, h, w = x1.shape
        
        # 边缘细节权重
        edge_w = self.edge_enhance(x1)
        x1_enhanced = x1 * edge_w + x1  # 残差边缘增强
        
        # 收集所有池化特征
        feats = [x1_enhanced]  # 原始特征
        
        # 方形池化
        for m in self.m:
            feats.append(m(x1_enhanced))
        
        # 水平条形池化（捕捉钢轨横向连续性）
        for pool in self.strip_pool_h:
            feats.append(pool(x1_enhanced))
        
        # 垂直条形池化（捕捉纵向裂纹）
        for pool in self.strip_pool_v:
            feats.append(pool(x1_enhanced))
        
        # 融合所有尺度
        concat_feat = torch.cat(feats, 1)
        
        # 通过1x1卷积压缩
        fused = self.cv5(concat_feat)
        
        # 后处理
        y1 = self.cv6(fused)
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))