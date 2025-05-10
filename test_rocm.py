import torch
print(torch.__version__)
print(torch.version.hip)
x = torch.rand(2, 2).to("cuda")
print(x @ x)