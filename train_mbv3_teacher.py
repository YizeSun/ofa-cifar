import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 👇 OFA MobileNetV3Large
from ofa.imagenet_classification.networks import MobileNetV3Large

# === Device Compatibility ===
if hasattr(torch.version, "hip") and torch.version.hip is not None:
    device = torch.device("cuda")  # ROCm/AMD
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA
else:
    device = torch.device("cpu")

print(f"🔍 Using device: {device}")
print("PyTorch version:", torch.__version__)
print("ROCm version:", torch.version.hip)
print("CUDA available:", torch.cuda.is_available())

print(f"🔍 Using device: {device}")

# === Configuration ===
num_epochs = 100
batch_size = 128
learning_rate = 0.05
width_mult = 1.0
ks = 7
expand_ratio = 6
depth_param = 4
n_classes = 10

cifar_data_path = "/lustre/hpe/ws12/ws12.a/ws/xmuyzsun-WK0/ofa-cifar/datasets/"

# === CIFAR-10 Data ===
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
size, pad = 32, 4
transform_train = transforms.Compose([
    transforms.Resize(size,padding=pad),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
# transform_test = transforms.Compose([
#     transforms.Resize(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
test_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])

trainset = torchvision.datasets.CIFAR10(root=cifar_data_path, train=True, download=False, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root=cifar_data_path, train=False, download=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# === Teacher Model: OFA-aligned MobileNetV3Large ===
model = MobileNetV3Large(
    n_classes=n_classes,
    bn_param=(0.1, 1e-3),
    dropout_rate=0,
    width_mult=width_mult,
    ks=ks,
    expand_ratio=expand_ratio,
    depth_param=depth_param
)
model.to(device)

# === Loss & Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# === Training ===
print("🔧 Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    print(f"✅ Epoch {epoch+1}: avg loss = {total_loss / len(trainloader):.4f}")

# === Save Teacher Model ===
torch.save(model.state_dict(), f"{cifar_data_path}/ofa_teacher_mbv3_cifar10.pth")
print("📦 Saved teacher model as: ofa_teacher_mbv3_cifar10.pth")

# === Evaluation ===
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"🎯 Final Test Accuracy: {accuracy:.2f}%")
