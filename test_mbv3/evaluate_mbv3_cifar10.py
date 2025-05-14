import argparse
import json
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
from ofa.model_zoo import ofa_net
import deepspeed

import argparse
import json
import random
import numpy as np

def summarize_all_evaluations(file_paths):
    all_accuracies = []
    all_losses = []

    for path in file_paths:
        with open(path, 'r') as f:
            graph = json.load(f)
            evals = graph.get('evaluation', [])
            if isinstance(evals, list) and all(isinstance(e, dict) for e in evals):
                for e in evals:
                    all_accuracies.append(e['accuracy'])
                    all_losses.append(e['loss'])

    if not all_accuracies:
        print("⚠️ No valid evaluations found.")
        return

    acc_mean = np.mean(all_accuracies)
    acc_range = (np.max(all_accuracies) - np.min(all_accuracies)) / 2
    loss_mean = np.mean(all_losses)
    loss_range = (np.max(all_losses) - np.min(all_losses)) / 2

    print("\n📊 Overall Evaluation Summary:")
    print(f"  Accuracy = {acc_mean:.4f} ± {acc_range:.4f}")
    print(f"  Loss     = {loss_mean:.4f} ± {loss_range:.4f}")

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

# --- Main pipeline ---
def main():
    local_rank = setup()
    parser = argparse.ArgumentParser(description="Evaluate or summarize a graph architecture.")
    parser.add_argument('--input', type=str, nargs='+', required=True, help='Input JSON file(s)')
    parser.add_argument('--summary', action='store_true', help='Summarize evaluation results')
    args = parser.parse_args()

    if args.summary:
        summarize_all_evaluations(args.input)
        return
    input_path = args.input

    device = torch.device("cuda", local_rank)

    for input_file in input_path:
        with open(input_file, 'r') as f:
            my_graph = json.load(f)

        if 'evaluation' not in my_graph or not isinstance(my_graph['evaluation'], list):
            my_graph['evaluation'] = []

    # === Config ===
    num_epochs = 100
    batch_size = 128
    learning_rate = 0.05
    width_mult = 1.0
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
    super_net = ofa_net("ofa_mbv3_d234_e346_k357_w1.0", pretrained=False)
    # 'ks': [7, 5, 3, 3, 5, 5, 3, 5, 7, 7, 3, 5, 5, 7, 5, 5, 7, 3, 5, 7], 'e': [6, 4, 3, 6, 3, 3, 6, 4, 3, 4, 3, 6, 3, 6, 6, 6, 4, 6, 3, 6], 'd': [3, 2, 4, 3, 3]
    model = super_net.set_active_subnet(ks=my_graph["ks_e_d"]['ks']
                        , e=my_graph["ks_e_d"]['e']
                        , d=my_graph["ks_e_d"]['d'])
    model = model.to(device)


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
            my_result = {}
            print(f"✅ Epoch {epoch+1}: avg loss = {total_loss / len(trainloader):.4f}")
            my_result["loss"]=total_loss / len(trainloader)

    # === Save Model (only by rank 0) ===
    if rank == 0:
        # torch.save({"state_dict": model.module.state_dict()}, f"{cifar_data_path}/ofa_teacher_12051745_{size}_cifar10.pth")
        # print("📦 Saved teacher model")

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
        my_result["accuracy"] = 100 * correct / total

        my_graph['evaluation'].append(my_result)

        with open(input_file, 'w') as f:
            json.dump(my_graph, f, indent=2)

        print(f"✅ Evaluated: {input_file}")
    cleanup()

if __name__ == "__main__":
    main()
