import torch
import torch.nn as nn
from torch.nn import functional as F

from ultralytics.nn.modules.conv import LightConv

class SimAM(torch.nn.Module):
    def __init__(self, channels = None,out_channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y) 

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
 
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class HGBlock_SimAM(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2
        self.cv = SimAM(c2)
        
    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv(self.ec(self.sc(torch.cat(y, 1))))
        return y + x if self.add else y
class RailSimAM_Lite(nn.Module):
    """
    适用于钢轨缺陷检测的轻量SimAM改进版
    改进点：
    1. 增加水平条带池化，捕捉钢轨纵向上下文
    2. 局部-全局统计信息融合，增强小目标感知
    3. 通道自适应加权，强化缺陷特征通道
    """
    def __init__(self, channels, out_channels=None, e_lambda=1e-4, strip_height=1, strip_width=7):
        super().__init__()
        self.channels = channels
        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid()
        
        # 水平条带池化：专门捕捉钢轨纵向（水平方向）的特征分布
        # strip_height=1表示在垂直方向压缩，保留水平方向信息
        self.strip_pool = nn.AdaptiveAvgPool2d((strip_height, None))
        
        # 轻量级水平卷积，用于建模水平方向依赖关系
        self.h_conv = nn.Sequential(
            nn.Conv2d(channels, channels//4, kernel_size=(1, 7), padding=(0, 3), groups=channels//4),
            nn.BatchNorm2d(channels//4),
            nn.SiLU(),
            nn.Conv2d(channels//4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 通道注意力精炼
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//8, 1),
            nn.SiLU(),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        
        # 可学习融合参数
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):

        b, c, h, w = x.size()
        
        # 原始SimAM分支（全局统计）
        n = w * h - 1
        mu = x.mean(dim=[2,3], keepdim=True)
        x_minus_mu_square = (x - mu).pow(2)
        simam_weight = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        simam_out = x * self.activation(simam_weight)
        
        # 水平条带注意力分支（局部纵向统计）
        # 对垂直方向进行平均池化，保留水平方向特征
        strip_feat = self.strip_pool(x)  # [B, C, 1, W]
        # 插值回原尺寸，每个像素获得其所在水平线的全局信息
        strip_feat = F.interpolate(strip_feat, size=(h, w), mode='bilinear', align_corners=False)
        h_weight = self.h_conv(strip_feat)
        h_out = x * h_weight
        
        # 自适应融合：原始SimAM + 水平条带
        # gamma初始为0，beta初始为1，训练初期接近原始SimAM，后期学习融合
        fused = self.gamma * simam_out + self.beta * h_out
        
        # 通道精炼：抑制背景通道，增强缺陷相关通道
        channel_weight = self.channel_gate(fused)
        out = fused * channel_weight
        
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(channels={self.channels}, lambda={self.e_lambda})"
class HGBlock_Rail(nn.Module):
    """
    针对钢轨检测优化的HGBlock
    改进：
    1. 使用RailSimAM替代原始SimAM
    2. 增加跨层特征融合，防止小目标信息丢失
    3. 自适应残差权重
    """
    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)
        self.add = shortcut and c1 == c2
        
        # 使用改进的RailSimAM-Lite（平衡速度与精度）
        self.cv = RailSimAM_Lite(c2, strip_height=1, strip_width=7)
        
        # 跨层特征压缩，防止小目标信息在深层丢失
        self.feat_compress = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2//4, 1),
            nn.SiLU()
        )
        
        # 自适应残差权重（类似于SKNet）
        self.residual_gate = nn.Sequential(
            nn.Linear(c2//4, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        y = [x]
        # 收集每一层的输出用于后续融合
        for m in self.m:
            y.append(m(y[-1]))
        
        # 特征拼接与压缩
        concat_feat = torch.cat(y, 1)
        squeezed = self.sc(concat_feat)
        excitated = self.ec(squeezed)
        
        # 应用RailSimAM
        attn_out = self.cv(excitated)
        
        # 自适应残差：根据特征内容决定是否使用残差
        if self.add:
            # 生成全局特征描述符
            global_feat = self.feat_compress(attn_out).view(attn_out.size(0), -1)
            weights = self.residual_gate(global_feat)  # [B, 2]
            # weights[:, 0] 对应主路径权重，weights[:, 1] 对应残差权重
            w0 = weights[:, 0].view(-1, 1, 1, 1)
            w1 = weights[:, 1].view(-1, 1, 1, 1)
            return w0 * attn_out + w1 * x
        else:
            return attn_out