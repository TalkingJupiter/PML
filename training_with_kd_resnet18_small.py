import logging
import os
import argparse

import torch
import torch.nn as nn
import timm

import models.ghostnetv3_small as ghostnetv3_small
from models.resnet import resnet18
from loss.distillation_loss import DistillationLoss
from utils import (
    train,
    evaluate,
    get_device,
    get_dataset_loader,
    get_optimizer,
    get_scheduler,
    init_weights_kaiming,
    EPOCHS
)

def parse_args():
    parser = argparse.ArgumentParser(description="KD: ResNet18 (teacher) -> GhostNetV3_small (student) on CIFAR-10")
    parser.add_argument("--student_width", type=float, default=1.0)

    # Match your experiments layout like: experiments/<run_name>/best_model.pth
    parser.add_argument("--outdir", type=str, default="experiments")
    parser.add_argument("--run_name", type=str, default="KD_GN-S_1.0x_from_R18")

    # Teacher model location under experiments
    parser.add_argument("--teacher_run", type=str, default="R18_default")
    parser.add_argument("--teacher_ckpt", type=str, default="best_model.pth")
    return parser.parse_args()

def _load_state_dict_flexible(model: torch.nn.Module, ckpt_obj):
    """
    Supports:
      1) torch.save(model.state_dict(), path)
      2) torch.save({'model_state_dict': model.state_dict(), ...}, path)
    """
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        state = ckpt_obj["model_state_dict"]
    else:
        state = ckpt_obj
    model.load_state_dict(state)

def main():
    args = parse_args()

    # ===== Run dir + logging (same style as train_default_small.py) =====
    run_dir = os.path.join(args.outdir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True
    )

    logging.info("Starting KD training.")
    torch.manual_seed(0)
    device = get_device()
    logging.info(f"Using device: {device}")

    trainloader, testloader = get_dataset_loader()

    # ===== Student model: GhostNetV3 =====
    width = args.student_width
    logging.info(f"Using GhostNetV3_small student with width {width}")
    student = timm.create_model("ghostnetv3_small", width=width, num_classes=10)
    init_weights_kaiming(student)
    student.to(device)

    # ===== Teacher model: ResNet-18 (load from experiments/<teacher_run>/best_model.pth) =====
    teacher = resnet18(pretrained=False, device=device).to(device)

    teacher_path = os.path.join(args.outdir, args.teacher_run, args.teacher_ckpt)
    if not os.path.isfile(teacher_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found:\n"
            f"  expected: {teacher_path}\n"
            f"Fix: set --teacher_run <jobname> (folder under {args.outdir}) "
            f"or --teacher_ckpt <filename>."
        )

    teacher_ckpt = torch.load(teacher_path, map_location=device)
    _load_state_dict_flexible(teacher, teacher_ckpt)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    logging.info(f"Loaded teacher from: {teacher_path}")

    # ===== Loss, optimizer, scheduler =====
    criterion = DistillationLoss(temperature=5.0, alpha=0.7)
    optimizer = get_optimizer(student)
    scheduler = get_scheduler(optimizer, training_length=len(trainloader))

    # ===== Checkpointing under experiments/<run_name>/ =====
    checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
    best_model_path = os.path.join(run_dir, "best_model.pth")

    start_epoch = 1
    best_acc = 0.0
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        logging.info(
            f"Loaded checkpoint '{checkpoint_path}' "
            f"(epoch {checkpoint['epoch']}, best_acc {best_acc:.2f}%)"
        )

    for epoch in range(start_epoch, EPOCHS + 1):
        train(
            student,
            device,
            trainloader,
            criterion,
            optimizer,
            scheduler,
            epoch,
            teacher_model=teacher
        )

        acc = evaluate(student, device, testloader, nn.CrossEntropyLoss())
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), best_model_path)
            logging.info(f"New best accuracy: {best_acc:.2f}%, model saved to {best_model_path}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "teacher_path": teacher_path,
            "student_width": width,
        }
        torch.save(checkpoint, checkpoint_path)

    logging.info(f"Training complete. Best Test Accuracy: {best_acc:.2f}%")
    logging.info(f"Best student saved at: {best_model_path}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()