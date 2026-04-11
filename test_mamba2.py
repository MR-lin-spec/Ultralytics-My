import importlib
import torch

def check_module(name: str):
    """检查指定模块是否安装并输出版本信息。"""
    try:
        module = importlib.import_module(name)
        version = getattr(module, '__version__', '未知版本')
        print(f"{name} 已安装，版本：{version}")
    except ImportError:
        print(f"{name} 未安装")

def main():
    print("====== PyTorch 环境信息 ======")
    print(f"torch.__version__          = {torch.__version__}")
    print(f"torch.version.cuda         = {torch.version.cuda}")
    print(f"torch.cuda.is_available()  = {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count()  = {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前 CUDA 设备: {torch.cuda.get_device_name(0)}")

    print("\n====== 其他模块检查 ======")
    for module_name in ["selective_scan_cuda", "mamba_ssm", "causal_conv1d", "torchvision"]:
        check_module(module_name)

if __name__ == "__main__":
    main()
