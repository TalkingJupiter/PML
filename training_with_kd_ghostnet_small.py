import logging
import os
import argparse

import torch
import torch.nn as nn
import timm

import models.ghostnetv3_small as ghostnetv3_small
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
    p = argparse.ArgumentParser(description="KD: GhostNetV3_small (teacher width=2.8) -> GhostNetV3_small (student width=1.0) on CIFAR-10")

    # widths
    p.add_argument("--student_width", type=float, default=1.0)
    p.add_argument("--teacher_width", type=float, default=2.8)

    # experiments layout
    p.add_argument("--outdir", type=str, default="experiments")
    p.add_argument("--run_name", type=str, default="KD_GN-S_1.0x_from_GN-S_2.8x")

    # teacher checkpoint location: experiments/<teacher_run>/best_model.pth
    p.add_argument("--teacher_run", type=str, default="GN-S_2.8x_default")
    p.add_argument("--teacher_ckpt", type=str, default="best_model.pth")

    # KD hyperparams
    p.add_argument("--temperature", type=float, default=5.0)
    p.add_argument("--alpha", type=float, default=0.7)

    return p.parse_args()

def _load_state_dict_flexible(model: torch.nn.Module, ckpt_obj):
    """
    Supports both formats:
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

    # ===== Run dir + logging (matches your experiments/<run_name>/train.log pattern) =====
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

    # ===== Student model =====
    logging.info(f"Student: ghostnetv3_small width={args.student_width}")
    student = timm.create_model("ghostnetv3_small", width=args.student_width, num_classes=10)
    init_weights_kaiming(student)
    student.to(device)

    # ===== Teacher model (loaded from experiments/<teacher_run>/<teacher_ckpt>) =====
    logging.info(f"Teacher: ghostnetv3_small width={args.teacher_width}")
    teacher = timm.create_model("ghostnetv3_small", width=args.teacher_width, num_classes=10).to(device)

    teacher_path = os.path.join(args.outdir, args.teacher_run, args.teacher_ckpt)
    if not os.path.isfile(teacher_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found:\n"
            f"  expected: {teacher_path}\n\n"
            f"Fix:\n"
            f"  - set --teacher_run to the folder name under {args.outdir}\n"
            f"  - or set --teacher_ckpt if your filename isn't {args.teacher_ckpt}\n"
        )

    teacher_ckpt = torch.load(teacher_path, map_location=device)
    _load_state_dict_flexible(teacher, teacher_ckpt)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    logging.info(f"Loaded teacher weights from: {teacher_path}")

    # ===== Loss, optimizer, scheduler =====
    criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha)
    optimizer = get_optimizer(student)
    scheduler = get_scheduler(optimizer, training_length=len(trainloader))

    # ===== Checkpointing under experiments/<run_name>/ =====
    checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
    best_model_path = os.path.join(run_dir, "best_model.pth")

    start_epoch = 1
    best_acc = 0.0
    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]
        logging.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {ckpt['epoch']}, best_acc {best_acc:.2f}%)")

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

        ckpt = {
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "student_width": args.student_width,
            "teacher_width": args.teacher_width,
            "teacher_path": teacher_path,
            "temperature": args.temperature,
            "alpha": args.alpha,
        }
        torch.save(ckpt, checkpoint_path)

    logging.info(f"Training complete. Best Test Accuracy: {best_acc:.2f}%")
    logging.info(f"Best student saved at: {best_model_path}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()