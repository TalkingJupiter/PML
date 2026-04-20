import logging
import os
import argparse

import torch
import torch.nn as nn
import timm

import models.ghostnetv3_small as ghostnetv3_small
from models.resnet import resnet50
from loss.feature_distillation_loss import FeatureDistillationLoss
from feature_hooks import FeatureExtractor
from utils import (
    evaluate,
    get_device,
    get_dataset_loader,
    get_optimizer,
    get_scheduler,
    init_weights_kaiming,
    EPOCHS,
    sync_device,
    _make_divisible
)
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Feature-based KD: ResNet50 -> GhostNetV3_small on CIFAR-10")
    parser.add_argument("--student_width", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default="experiments")
    parser.add_argument("--run_name", type=str, default="Feature_KD_GN-S_1.0x_from_R50")
    parser.add_argument("--teacher_run", type=str, default="resnet/resnet50_seed0")
    parser.add_argument("--teacher_ckpt", type=str, default="best_model.pth")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for feature distillation loss")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()

def _load_state_dict_flexible(model: torch.nn.Module, ckpt_obj):
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        state = ckpt_obj["model_state_dict"]
    else:
        state = ckpt_obj
    model.load_state_dict(state)

def train_feature_kd(student, teacher, s_extractor, t_extractor, device, loader, criterion, optimizer, scheduler, epoch):
    student.train()
    teacher.eval()
    
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc=f"Train Epoch {epoch}", unit="batch")):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = teacher(inputs)
            teacher_features = t_extractor.get_features()

        student_logits = student(inputs)
        student_features = s_extractor.get_features()

        loss = criterion(student_logits, teacher_logits, student_features, teacher_features, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()
        sync_device(device)

        total_loss += loss.item()
        _, predicted = student_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            logging.info(f'Epoch {epoch} | Step {batch_idx+1}/{len(loader)} | Loss: {total_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(loader), 100. * correct / total

def main():
    args = parse_args()
    run_dir = os.path.join(args.outdir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True
    )

    logging.info("Starting Feature-based KD training.")
    torch.manual_seed(0)
    device = get_device()
    logging.info(f"Using device: {device}")

    trainloader, testloader = get_dataset_loader()
    if args.batch_size != 256:
        from torch.utils.data import DataLoader
        trainloader = DataLoader(trainloader.dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
        testloader = DataLoader(testloader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)

    # ===== Student model =====
    width = args.student_width
    student = timm.create_model("ghostnetv3_small", width=width, num_classes=10)
    init_weights_kaiming(student)
    student.to(device)

    # ===== Teacher model =====
    teacher = resnet50(pretrained=False, device=device).to(device)
    teacher_path = os.path.join(args.outdir, args.teacher_run, args.teacher_ckpt)
    if os.path.isfile(teacher_path):
        teacher_ckpt = torch.load(teacher_path, map_location=device)
        _load_state_dict_flexible(teacher, teacher_ckpt)
        logging.info(f"Loaded teacher from: {teacher_path}")
    else:
        logging.warning(f"Teacher checkpoint NOT found at {teacher_path}. Training with uninitialized teacher (for testing)!")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ===== Feature Extractors =====
    s_layer_names = ['blocks.1', 'blocks.3', 'blocks.4', 'blocks.5']
    t_layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    
    s_extractor = FeatureExtractor(student, s_layer_names)
    t_extractor = FeatureExtractor(teacher, t_layer_names)

    # ===== Loss, optimizer, scheduler =====
    # Define channels based on width
    s_channels = [
        _make_divisible(20 * width, 4),
        _make_divisible(64 * width, 4),
        _make_divisible(80 * width, 4),
        _make_divisible(160 * width, 4)
    ]
    t_channels = [256, 512, 1024, 2048]

    criterion = FeatureDistillationLoss(
        s_channels, t_channels, 
        temperature=5.0, alpha=0.7, beta=args.beta
    ).to(device)

    # Important: optimizer must include projection parameters
    optimizer = get_optimizer(student)
    optimizer.add_param_group({'params': criterion.projections.parameters()})
    
    scheduler = get_scheduler(optimizer, training_length=len(trainloader))

    # ===== Training loop =====
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_feature_kd(
            student, teacher, s_extractor, t_extractor, device, 
            trainloader, criterion, optimizer, scheduler, epoch
        )
        
        test_acc = evaluate(student, device, testloader, nn.CrossEntropyLoss())
        
        logging.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                     f"Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), os.path.join(run_dir, "best_model.pth"))
            logging.info(f"Saved new best model with accuracy: {best_acc:.2f}%")

    logging.info("Training complete.")

if __name__ == "__main__":
    main()
