import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.rand((3000, 3000), device=device)
b = torch.rand((3000, 3000), device=device)
c = a @ b

print(f"运行设备：{device}")
print(f"结果形状：{c.shape}")
print(f"结果均值：{c.mean().item():.4f}")