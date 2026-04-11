# -*- coding: utf-8 -*-
# @Time : 2024-11-2024/11/4 16:01
# @Author : 林子涵
# @File : test.py

from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEPETrainer
from ultralytics import RTDETR

# Load a pretrained YOLO11n model
#model = YOLO("/root/autodl-tmp/PretrainedModels/yolo11n.pt")  # Load a pretrained YOLO11n model

rtdet = RTDETR("/root/Ultralytics-My/ultralytics/cfg/models/rt-detr/rtdetr-l-simam-dynamic-simam-sppcspc.yaml")  # Load a pretrained RTDETR model

"""


"""

# Fine-tune on your detection dataset
results = rtdet.train(
    data="/root/Ultralytics-My/ultralytics/cfg/datasets/railway-small-data.yaml",  # Detection dataset
    epochs=80,
    patience=50,
    #trainer=YOLOEPETrainer,  # <- Important: use detection trainer
)

# Run inference on 'bus.jpg' with arguments


# Train the model on the COCO8 example dataset for 100 epochs
#results = rtdet.train(data="coco128.yaml", epochs=100, imgsz=640)

# Run inference with the YOLO26n model on the 'bus.jpg' image
#results = model("bus.jpg")
