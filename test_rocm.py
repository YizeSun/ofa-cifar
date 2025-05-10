import torch

print(torch.version)
print(torch.version.cuda)
print(torch.version.hip)

# Expected outputs
# 2.6
# None
# 6.2 # or another ROCm version

print(torch.__version__)
x = torch.rand(2, 2).to("cuda")
print(x @ x)