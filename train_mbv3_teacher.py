import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from ofa.imagenet_classification.networks import MobileNetV3Large
import deepspeed

# === Distributed Init ===
def setup():
    deepspeed.init_distributed(dist_backend='nccl')  # assumes NCCL + ROCm for MI300A
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    print(f'[DDP SETUP] rank={rank}, local_rank={local_rank}, world_size={world_size}')
    return local_rank

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    local_rank = setup()
    device = torch.device("cuda", local_rank)

    # === Config ===
    num_epochs = 100
    batch_size = 128
    learning_rate = 0.05
    width_mult = 1.0
    ks = 7 # [3,5,7]
    expand_ratio = 6 # [3,4,6]
    depth_param = 4 # [2,3,4]
    n_classes = 10

    cifar_data_path = "/lustre/hpe/ws12/ws12.a/ws/xmuyzsun-WK0/ofa-cifar/datasets"
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    size = 32 # 224

    transform_train = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # === Dataset and Sampler
    trainset = torchvision.datasets.CIFAR10(root=cifar_data_path, train=True, download=False, transform=transform_train)
    train_sampler = DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root=cifar_data_path, train=False, download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    # === Model
    model = MobileNetV3Large(
        n_classes=n_classes,
        bn_param=(0.1, 1e-5),
        dropout_rate=0,
        width_mult=width_mult,
        ks=ks,
        expand_ratio=expand_ratio,
        depth_param=depth_param
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    rank = dist.get_rank()
    if rank == 0:
        print(f"[TRAINING] Starting training on {dist.get_world_size()} processes")

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0

        epoch_iter = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) if rank == 0 else trainloader

        for inputs, labels in epoch_iter:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if rank == 0:
            print(f"✅ Epoch {epoch+1}: avg loss = {total_loss / len(trainloader):.4f}")

    # === Save Model (only by rank 0) ===
    if rank == 0:
        torch.save({"state_dict": model.module.state_dict()}, f"{cifar_data_path}/ofa_teacher_12051745_{size}_cifar10.pth")
        print("📦 Saved teacher model")

        # === Evaluation
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
        print(f"🎯 Final Test Accuracy: {100 * correct / total:.2f}%")

    cleanup()

if __name__ == "__main__":
    main()
