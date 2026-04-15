# -*- coding: utf-8 -*-
# @Time : 2024-11-04 16:01
# @File : train_and_benchmark_final.py

import warnings
import argparse
import os
import time
import numpy as np
import torch
import torch as th
import re
from ultralytics import YOLO, RTDETR
from ultralytics.utils.torch_utils import select_device
#from ultralytics.nn.tasks import attempt_load_weights
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== 训练部分函数 (来自代码2) ====================

def load_weights_with_freeze(pretrained_path, model, save_mapped_path=None):
    """
    加载预训练权重，并自动冻结匹配的层，不匹配的层保持可训练
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
    
    # 定义层号映射表（根据你的模型结构调整，这里保留之前的逻辑）
    layer_mapping = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
        9: 10, # 假设SimAM插入导致错位
        10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20,
        20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 28
    }
    
    matched_layers = set()
    mapped_weights = {}
    matched_count = 0
    
    for old_key, old_weight in pretrained.items():
        match = re.match(r'model\.(\d+)\.(.*)', old_key)
        if not match: continue
        
        old_layer = int(match.group(1))
        rest = match.group(2)
        
        if old_layer not in layer_mapping: continue
        
        new_layer = layer_mapping[old_layer]
        new_key = f"model.{new_layer}.{rest}"
        
        if new_key in current_state:
            if current_state[new_key].shape == old_weight.shape:
                mapped_weights[new_key] = old_weight
                matched_count += 1
                matched_layers.add(new_layer)
    
    current_state.update(mapped_weights)
    current_model.load_state_dict(current_state, strict=False)
    
    # ========== 冻结策略 ==========
    all_params = list(current_model.named_parameters())
    frozen_count = 0
    trainable_count = 0
    
    for name, param in all_params:
        match = re.match(r'model\.(\d+)\.', name)
        if match:
            layer_num = int(match.group(1))
            if layer_num in matched_layers:
                param.requires_grad = False
                frozen_count += 1
            else:
                param.requires_grad = True
                trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
    
    print(f"冻结参数数量: {frozen_count}")
    print(f"可训练参数数量: {trainable_count}")
    
    if save_mapped_path and matched_count > 0:
        th.save({'model': current_model.state_dict()}, save_mapped_path)
        print(f"映射后的权重已保存至: {save_mapped_path}")
    
    return model

def get_trainable_params_info(model):
    if hasattr(model, 'model'):
        current_model = model.model
    else:
        current_model = model
    
    total_params = sum(p.numel() for p in current_model.parameters())
    trainable_params = sum(p.numel() for p in current_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    return trainable_params

# ==================== 推理/测试部分函数 (来自代码1) ====================

def get_weight_size(path):
    try:
        stats = os.stat(path)
        return f'{stats.st_size / (1024 ** 2):.1f}'
    except OSError as e:
        logging.error(f"Error getting weight size: {e}")
        return "N/A"

def warmup_model(model, device, example_inputs, iterations=200):
    logging.info("Beginning warmup...")
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(iterations), desc='Warmup'):
            model(example_inputs)

def test_model_latency(model, device, example_inputs, iterations=1000):
    logging.info("Testing latency...")
    model.eval()
    time_arr = []
    with torch.no_grad():
        for _ in tqdm(range(iterations), desc='Latency Test'):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start_time = time.time()
            model(example_inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            end_time = time.time()
            time_arr.append(end_time - start_time)

    return np.mean(time_arr), np.std(time_arr)

# ==================== 主程序入口 ====================

def main(opt):
    # --- 第一阶段：训练 ---
    logging.info("Starting Training Phase...")
    
    # 1. 创建模型 (使用你提供的路径)
    config_path = "/root/Ultralytics-My/ultralytics/cfg/models/rt-detr/trdetr-l-smallighting-SPPELAN-SimAM.yaml"
    rtdet = RTDETR(config_path)

    # 2. 加载权重并自动冻结匹配层
    # 默认使用 rtdetr-l.pt，可以通过命令行参数修改
    if os.path.exists(opt.pretrained_weights):
        rtdet = load_weights_with_freeze(
            pretrained_path=opt.pretrained_weights,
            model=rtdet,
            save_mapped_path='rtdetr-l-simam-mapped.pt' # 保存映射后的权重
        )
    else:
        logging.warning(f"Pretrained weights not found at {opt.pretrained_weights}. Training from scratch.")

    # 3. 打印参数信息
    get_trainable_params_info(rtdet)

    # 4. 打印模型信息
    print(f"\n{'='*70}")
    rtdet.info()
    print(f"{'='*70}\n")

    # 5. 训练（匹配层已冻结，只训练不匹配层）
    logging.info("Starting Training...")
    results = rtdet.train(
        data="/root/Ultralytics-My/ultralytics/cfg/datasets/railway-big-data-nocombine.yaml",
        epochs=opt.epochs,
        patience=50,
        save_json=True,
        save_log=True,
        log_file="runs/train/exp/train_nocombine.log",
        
        # 优化器设置
        lr0=0.0001,
        lrf=0.0001,
        batch=32,
        momentum=0.937,
        weight_decay=0.0005,
        
        # 图片尺寸
        imgsz=opt.imgs[0], 
    )
    
    # 获取训练好的最佳权重路径
    trained_weights_path = os.path.join("runs", "train", "exp", "weights", "best.pt")
    if not os.path.exists(trained_weights_path):
        trained_weights_path = os.path.join("runs", "train", "exp", "weights", "last.pt")
    
    logging.info(f"Training completed. Using weights for inference: {trained_weights_path}")

    # --- 第二阶段：性能测试 ---
    logging.info("Starting Inference Benchmarking Phase...")
    
    device = select_device(opt.device)
    
    # 加载训练好的权重进行测试
    #model = attempt_load_weights(trained_weights_path, device=device, fuse=True)
    model = model.to(device).fuse()
    
    example_inputs = torch.randn((opt.batch_size, 3, *opt.imgs)).to(device)

    if opt.half:
        model = model.half()
        example_inputs = example_inputs.half()

    warmup_model(model, device, example_inputs, opt.warmup)
    mean_latency, std_latency = test_model_latency(model, device, example_inputs, opt.testtime)

    logging.info(f"Model weights: {trained_weights_path} Size: {get_weight_size(trained_weights_path)}M "
                 f"(Batch size: {opt.batch_size}) Latency: {mean_latency:.5f}s ± {std_latency:.5f}s "
                 f"FPS: {opt.batch_size / mean_latency:.1f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RT-DETR and Benchmark Performance.")
    # 训练参数
    parser.add_argument('--pretrained_weights', type=str, default='rtdetr-l.pt', help='pretrained weights path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    
    # 推理/测试参数
    parser.add_argument('--batch_size', type=int, default=1, help='total batch size for inference test')
    parser.add_argument('--imgs', nargs='+', type=int, default=[640, 640], help='image sizes [height, width]')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--warmup', default=200, type=int, help='warmup iterations')
    parser.add_argument('--testtime', default=1000, type=int, help='test iterations for latency')
    parser.add_argument('--half', action='store_true', help='use FP16 mode for inference')
    
    opt = parser.parse_args()
    main(opt)