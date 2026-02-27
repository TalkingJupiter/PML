import logging
import os

import torch
import torch.nn as nn
import timm
import argparse

import models.ghostnetv3 as ghostnetv3
from utils import (
    train,
    evaluate,
    get_device,
    get_dataset_loader,
    get_scheduler,
    get_optimizer,
    init_weights_kaiming,
    EPOCHS
)

def parse_args():
    parser = argparse.ArgumentParser(description= "GN-D 1.0x")
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default="GN-D 1.0x")
    return parser.parse_args()

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable

def log_param_table(model, name="model"):
    total, trainable, non_trainable = count_params(model)
    print("=" * 60)
    print(f"Parameter summary for: {name}")
    print(f"Total params      : {total:,}")
    print(f"Trainable params  : {trainable:,}")
    print(f"Non-trainable     : {non_trainable:,}")
    print("=" * 60)

def main():
    args = parse_args()
    torch.manual_seed(0)
    run_dir = os.path.join(args.outdir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    log_file = os.path.join(run_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers= [
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )

    print("Starting default training.")
    device = get_device()
    logging.info(f'Using device: {device}')

    trainloader, testloader = get_dataset_loader()

    # Model: GhostNetV3
    model = timm.create_model('ghostnetv3', width=1.0, num_classes=10)
    log_param_table(model, name=f"ghostnetv3 (width=1.0)")
    init_weights_kaiming(model)
    model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, training_length=len(trainloader))

    # Resume from checkpoint if exists
    checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
    best_model_path = os.path.join(run_dir, "best_model.pth")

    start_epoch = 1
    best_acc = 0.0
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        logging.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']}, best_acc {best_acc:.2f}%)")

    for epoch in range(start_epoch, EPOCHS + 1):
        train(model, device, trainloader, criterion, optimizer, scheduler, epoch)
        acc = evaluate(model, device, testloader, criterion)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'New best accuracy: {best_acc:.2f}%, model saved.')

        # Save checkpoint to resume training
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }
        torch.save(checkpoint, checkpoint_path)

    logging.info(f'Training complete. Best Test Accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
