import torch
# 强制启用CUDA
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEPETrainer
from ultralytics import RTDETR

# 或者在创建模型前确保CUDA可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rtdet = RTDETR("/root/Ultralytics-My/ultralytics/cfg/models/rt-detr/rtdetr-mamba-b.yaml")
rtdet = rtdet.to(device)