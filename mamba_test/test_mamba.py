import torch, os
print("Torch:", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("GPU?  :", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA 不可用，请检查驱动/容器"
 
# Mamba 前向一把（GPU）
from mamba_ssm import Mamba
m = Mamba(d_model=256, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(1, 1024, 256, device="cuda")
y = m(x)
print("Forward OK, y.shape =", y.shape)
