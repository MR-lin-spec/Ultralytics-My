from ultralytics import RTDETR
import torch
import torch.nn as nn

# 加载模型
rtdetr = RTDETR("/root/Ultralytics-My/best_mysuccessfully.pt")
model = rtdetr.model

def fix_activation_module(module, path=""):
    """
    递归修复所有 activation 模块，添加 exporter 需要的所有属性
    """
    for name, child in module.named_children():
        full_path = f"{path}.{name}" if path else name
        
        if type(child).__name__ == 'activation':
            # ========== 修复 gn 属性 ==========
            if not hasattr(child, 'gn'):
                # 如果没有 gn，添加 Identity 作为占位（forward 中实际用的是 bn）
                child.gn = nn.Identity()
                print(f"[Fix] Added 'gn' (Identity) at: {full_path}")
            
            # ========== 修复卷积属性（用于 exporter 的 NMSModel wrap） ==========
            # stride: 默认为 1（activation 内部 conv 的 stride 实际为 1）
            if not hasattr(child, 'stride'):
                child.stride = (1, 1)
                print(f"[Fix] Added 'stride' at: {full_path}")
            
            # padding: 使用 act_num 作为 padding（与 forward 中一致）
            if not hasattr(child, 'padding'):
                padding_val = child.act_num if hasattr(child, 'act_num') else 3
                child.padding = (padding_val, padding_val)
                print(f"[Fix] Added 'padding' at: {full_path}")
            
            # dilation: 默认为 1
            if not hasattr(child, 'dilation'):
                child.dilation = (1, 1)
                print(f"[Fix] Added 'dilation' at: {full_path}")
            
            # groups: 等于 dim（与 forward 中 groups=self.dim 一致）
            if not hasattr(child, 'groups'):
                child.groups = child.dim if hasattr(child, 'dim') else child.weight.size(0)
                print(f"[Fix] Added 'groups'={child.groups} at: {full_path}")
            
            # in_channels / out_channels: 都等于 dim
            if not hasattr(child, 'in_channels'):
                child.in_channels = child.dim if hasattr(child, 'dim') else child.weight.size(0)
                print(f"[Fix] Added 'in_channels' at: {full_path}")
            
            if not hasattr(child, 'out_channels'):
                child.out_channels = child.dim if hasattr(child, 'dim') else child.weight.size(0)
                print(f"[Fix] Added 'out_channels' at: {full_path}")
            
            # kernel_size: 从 weight shape 获取
            if not hasattr(child, 'kernel_size'):
                if hasattr(child, 'weight') and child.weight is not None:
                    k = child.weight.size(-1)
                    child.kernel_size = (k, k)
                    print(f"[Fix] Added 'kernel_size'={child.kernel_size} at: {full_path}")
            
            # bias: 检查是否存在（forward 中可能用到）
            if not hasattr(child, 'bias'):
                child.bias = None
                print(f"[Fix] Initialized 'bias'=None at: {full_path}")
        
        # 递归处理子模块
        fix_activation_module(child, full_path)

# 执行修复
print("=" * 50)
print("开始修复 activation 模块...")
print("=" * 50)
fix_activation_module(model)
print("=" * 50)
print("修复完成，开始导出...")
print("=" * 50)

# 导出 ONNX
try:
    rtdetr.export(format="tflite", simplify=True,opset=17,int8=True)
    print("导出成功！")
except Exception as e:
    print(f"导出失败: {e}")
    # 如果还是失败，尝试禁用 NMS
    print("\n尝试禁用 NMS 导出...")
    try:
        rtdetr.export(format="tflite", nms=False, simplify=True,opset=17,int8=True)
        print("禁用 NMS 后导出成功！")
    except Exception as e2:
        print(f"禁用 NMS 后仍失败: {e2}")