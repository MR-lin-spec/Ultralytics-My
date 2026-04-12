# -*- coding: utf-8 -*-
# @Time : 2024-11-2024/11/4 16:01
# @Author : 
# @File : test.py

from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEPETrainer
from ultralytics import RTDETR
import torch as th
import re

def load_weights_with_freeze(pretrained_path, model, save_mapped_path=None):
    """
    加载预训练权重，并自动冻结匹配的层，不匹配的层保持可训练
    处理键名前缀差异: model.0.xxx vs model.model.0.xxx
    """
    # 加载预训练权重
    pretrained = th.load(pretrained_path, map_location='cpu')['model'].state_dict()
    
    # 获取当前模型的state_dict和模型对象
    if hasattr(model, 'model'):
        current_model = model.model
    else:
        current_model = model
    
    current_state = current_model.state_dict()
    
    # 诊断：打印样本键名
    print("=== 键名诊断 ===")
    print(f"预训练样本: {list(pretrained.keys())[0]}")
    print(f"当前模型样本: {list(current_state.keys())[0]}")
    
    # 定义层号映射表（处理SimAM插入导致的错位）
    layer_mapping = {
        # Backbone - 直接对应（0-8层结构相同）
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
        # HGBlock -> HGBlock_EMA 尝试直接映射
        5: 5, 6: 6, 7: 7, 8: 8,
        # 原始第9层(HGBlock stage4) -> 第10层（因为第9层插入了SimAM）
        9: 10,
        # Head部分整体+1（因为SimAM插入）
        10: 11, 11: 12, 12: 13, 13: 14, 14: 15,
        15: 16, 16: 17, 17: 18, 18: 19, 19: 20,
        20: 21, 21: 22, 22: 23, 23: 24, 24: 25,
        25: 26, 26: 27, 27: 28
    }
    
    # 记录匹配和不匹配的层号
    matched_layers = set()
    unmatched_layers = set()
    
    # 执行映射
    mapped_weights = {}
    matched_count = 0
    shape_mismatch = []
    key_not_found = []
    
    for old_key, old_weight in pretrained.items():
        # 解析层号
        match = re.match(r'model\.(\d+)\.(.*)', old_key)
        if not match:
            key_not_found.append(f"{old_key} (无法解析层号)")
            continue
        
        old_layer = int(match.group(1))
        rest = match.group(2)
        
        if old_layer not in layer_mapping:
            key_not_found.append(f"{old_key} (层{old_layer}无映射)")
            continue
        
        new_layer = layer_mapping[old_layer]
        new_key = f"model.{new_layer}.{rest}"
        
        # 尝试匹配
        if new_key in current_state:
            if current_state[new_key].shape == old_weight.shape:
                mapped_weights[new_key] = old_weight
                matched_count += 1
                matched_layers.add(new_layer)
            else:
                shape_mismatch.append(
                    f"{old_key} -> {new_key} (shape: {old_weight.shape} vs {current_state[new_key].shape})"
                )
                unmatched_layers.add(new_layer)
        else:
            key_not_found.append(f"{old_key} -> {new_key} (键不存在)")
            unmatched_layers.add(new_layer)
    
    # 加载匹配的权重
    current_state.update(mapped_weights)
    current_model.load_state_dict(current_state, strict=False)
    
    # ========== 冻结匹配的层，不匹配的层保持训练 ==========
    print(f"\n{'='*70}")
    print(f"冻结策略: 匹配层冻结，不匹配层训练")
    print(f"{'='*70}")
    
    # 获取所有参数
    all_params = list(current_model.named_parameters())
    frozen_count = 0
    trainable_count = 0
    
    for name, param in all_params:
        # 解析层号
        match = re.match(r'model\.(\d+)\.', name)
        if match:
            layer_num = int(match.group(1))
            
            if layer_num in matched_layers:
                # 冻结匹配的层
                param.requires_grad = False
                frozen_count += 1
            else:
                # 不匹配的层保持可训练
                param.requires_grad = True
                trainable_count += 1
        else:
            # 非模型层（如 strides, anchors等），默认冻结
            param.requires_grad = False
            frozen_count += 1
    
    # 打印冻结信息
    print(f"匹配并冻结的层: {sorted(matched_layers)}")
    print(f"不匹配/训练的层: {sorted(unmatched_layers)}")
    print(f"\n冻结参数数量: {frozen_count}")
    print(f"可训练参数数量: {trainable_count}")
    print(f"冻结比例: {100*frozen_count/(frozen_count+trainable_count):.1f}%")
    
    # 验证冻结状态
    print(f"\n{'='*70}")
    print(f"冻结状态验证 (前5层示例)")
    print(f"{'='*70}")
    for name, param in all_params[:10]:
        match = re.match(r'model\.(\d+)\.', name)
        if match:
            layer = int(match.group(1))
            status = "冻结" if not param.requires_grad else "训练"
            match_status = "匹配" if layer in matched_layers else "不匹配"
            print(f"  Layer {layer}: {status} ({match_status}) - {name[:50]}...")
    
    # 打印统计
    total = len(pretrained)
    print(f"\n{'='*70}")
    print(f"权重加载统计")
    print(f"{'='*70}")
    print(f"总权重数: {total}")
    print(f"成功匹配: {matched_count} ({100*matched_count/total:.1f}%)")
    print(f"形状不匹配: {len(shape_mismatch)}")
    print(f"键不存在: {len(key_not_found)}")
    
    # 保存映射后的权重
    if save_mapped_path and matched_count > 0:
        th.save({
            'model': current_model.state_dict(),
            'date': 'mapped',
            'version': 'custom'
        }, save_mapped_path)
        print(f"\n已保存映射权重: {save_mapped_path}")
    
    return model


def get_trainable_params_info(model):
    """获取可训练参数的详细信息"""
    if hasattr(model, 'model'):
        current_model = model.model
    else:
        current_model = model
    
    total_params = sum(p.numel() for p in current_model.parameters())
    trainable_params = sum(p.numel() for p in current_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n{'='*70}")
    print(f"参数统计")
    print(f"{'='*70}")
    print(f"总参数: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M, {100*trainable_params/total_params:.1f}%)")
    print(f"冻结参数: {frozen_params:,} ({frozen_params/1e6:.2f}M, {100*frozen_params/total_params:.1f}%)")
    
    return trainable_params


# ==================== 主程序 ====================

# 1. 创建模型
config_path = "/root/Ultralytics-My/ultralytics/cfg/models/rt-detr/rtdetr-l-simam-dynamic-simam-smallloss.yaml"
rtdet = RTDETR(config_path)

# 2. 加载权重并自动冻结匹配层
rtdet = load_weights_with_freeze(
    pretrained_path='rtdetr-l.pt',
    model=rtdet,
    save_mapped_path='rtdetr-l-simam-mapped.pt'
)

# 3. 打印参数信息
get_trainable_params_info(rtdet)

# 4. 打印模型信息
print(f"\n{'='*70}")
rtdet.info()

# 5. 训练（匹配层已冻结，只训练不匹配层）
results = rtdet.train(
    data="/root/Ultralytics-My/ultralytics/cfg/datasets/railway-big-data-nocombine.yaml",
    epochs=100,
    patience=50,
    save_json=True,
    save_log=True,
    log_file="runs/train/exp/train_nocombine.log",
    
    # 优化器设置（针对部分训练）
    lr0=0.0001,        # 可以使用较大学习率，因为大部分参数已冻结
    lrf=0.0001,
    batch=32,
    momentum=0.937,
    weight_decay=0.0005,
    
    # 不需要再设置freeze，因为我们已经手动冻结了
    # freeze参数在这里不需要，因为我们已经在代码中处理了
)

# 6. 如果需要解冻所有层进行微调（可选第二阶段训练）
"""
# 阶段2：解冻所有层，使用小学习率微调
print("\n解冻所有层进行微调...")
for param in rtdet.model.parameters():
    param.requires_grad = True

results = rtdet.train(
    data="/root/Ultralytics-My/ultralytics/cfg/datasets/railway-small-data.yaml",
    epochs=20,       # 额外20轮微调
    lr0=0.0001,      # 很小学习率
    lrf=0.1,
    resume=True,     # 从上一轮继续
)
"""