import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from scheduler.warumup_cosine_lr import WarmupCosineLR
from torch.utils.data import DataLoader

def _make_divisible(v, divisor, min_value=None):
    """
    Ensures that all layers have a channel number that is divisible by divisor.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def init_weights_kaiming(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def sync_device(device):
    """Synchronize device operations for CUDA and MPS to avoid hangs."""
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

# Setup logger to console and file
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = datetime.now().strftime('train_default_%Y%m%d.log')
log_path = os.path.join(log_dir, log_filename)

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path)
    ]
)

# Hyperparameters
NUM_WORKERS = 6
BATCH_SIZE = 256
EPOCHS = 200
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.01


def train(student_model, device, loader, criterion, optimizer, scheduler, epoch, teacher_model=None, teacher_loader=None):
    student_model.train()
    # Initialize teacher loader iterator if a separate loader is provided
    if teacher_model is not None and teacher_loader is not None:
        teacher_iter = iter(teacher_loader)

    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc=f"Train Epoch {epoch}", unit="batch")):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if teacher_model is not None:
            # Fetch teacher inputs from separate loader if available
            if teacher_loader is not None:
                try:
                    t_inputs, t_targets = next(teacher_iter)
                except StopIteration:
                    teacher_iter = iter(teacher_loader)
                    t_inputs, t_targets = next(teacher_iter)
                t_inputs = t_inputs.to(device)
            else:
                t_inputs = inputs
            with torch.no_grad():
                teacher_outputs = teacher_model(t_inputs)

        student_outputs = student_model(inputs)
        if teacher_model is not None:
            loss = criterion(student_outputs, teacher_outputs, targets)
        else:
            loss = criterion(student_outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()
        sync_device(device)

        total_loss += loss.item()
        _, predicted = student_outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 100 == 0:
            logging.info(f'Epoch {epoch} | Step {batch_idx+1}/{len(loader)} | Loss: {total_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%')
    
    # End of epoch logging
    logging.info(f"Epoch {epoch} training completed")

def evaluate(model, device, loader, criterion):
    model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating", unit="batch", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    logging.info(f'Test Loss: {total_loss/len(loader):.4f} | Test Acc: {acc:.2f}%')
    # Ensure all evaluation ops are completed
    sync_device(device)
    
    return acc

def get_optimizer(model):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM,
        nesterov=True,
    )
    return optimizer

def get_scheduler(optimizer, training_length):
    total_steps = EPOCHS * training_length
    scheduler = WarmupCosineLR(
        optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
    )
    return scheduler

def get_dataset_loader(resize=None):
    # Data transforms for CIFAR-10
    if resize is not None:
        transform_train = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

    # Dynamically set num_workers and pin_memory based on device
    device = get_device()
    num_workers = 0 if device == 'mps' else NUM_WORKERS
    persistent_workers = False if num_workers == 0 else True
    pin_memory = True if device == 'cuda' else False

    # Datasets and loaders
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    return trainloader, testloader

def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'