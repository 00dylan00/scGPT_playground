import torch, sys

if torch.cuda.is_available():
    print("Available GPUs:", torch.cuda.device_count())
    cuda_version = torch.version.cuda
    print(f"PyTorch is using CUDA version: {cuda_version}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


else:
    print("CUDA is not available.")
    print(torch.cuda.is_available())
    cuda_version = torch.version.cuda
    print(f"PyTorch is using CUDA version: {cuda_version}")
    sys.exit(0)

