from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEPETrainer
from ultralytics import RTDETR
import torch as th
import re

rtdetr=RTDETR("best_rtdetr_rail.pt")
results = rtdetr.train(
    data="/root/Ultralytics-My/ultralytics/cfg/datasets/railway-big-data-nocombine.yaml",
    epochs=100,
    patience=50,
    #save_json=True,
    save_log=True,
    log_file="runs/train/exp/train_rtdetr_nocombine.log",
    batch=32,
    # 优化器设置（针对部分训练）
    lr0=0.0001,        # 可以使用较大学习率，因为大部分参数已冻结
    lrf=0.0001,
    momentum=0.937,
    weight_decay=0.0005,
    
    # 不需要再设置freeze，因为我们已经手动冻结了
    # freeze参数在这里不需要，因为我们已经在代码中处理了
)
