import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ofa.imagenet_classification.networks import MobileNetV3Large

# === Device setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔍 Using device: {device}")

# === Path to model ===
model_path = "/lustre/hpe/ws12/ws12.a/ws/xmuyzsun-WK0/ofa-cifar/datasets/ofa_teacher_mbv3_cifar10.pth"

# === Rebuild model exactly as in train_mbv3_teacher.py ===
model = MobileNetV3Large(
    n_classes=10,
    bn_param=(0.1, 1e-3),
    dropout_rate=0,
    width_mult=1.0,
    ks=7,
    expand_ratio=6,
    depth_param=4
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === CIFAR-10 test data ===
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.CIFAR10(
    root="/lustre/hpe/ws12/ws12.a/ws/xmuyzsun-WK0/ofa-cifar/datasets/",
    train=False,
    download=False,
    transform=transform_test
)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# === Evaluate accuracy ===
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"✅ Accuracy on CIFAR-10 test set: {accuracy:.2f}%")
