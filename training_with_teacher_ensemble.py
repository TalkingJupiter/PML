import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import argparse

import models.ghostnetv3 as ghostnetv3
from models.resnet import resnet50
from models.densenet import densenet161
from models.vgg import vgg13_bn
from models.inception import inception_v3
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
    parser = argparse.ArgumentParser(description="Ensamble GhostNetV3")
    parser.add_argument("--student_width", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default="experiments")
    parser.add_argument("--run_name", type=str, default="Train_teacher_ensemble_gnd_1.0x")

    return parser.parse_args()
# Add ensemble teacher wrapper
class EnsembleTeacher(nn.Module):
    def __init__(self, teachers):
        super().__init__()
        self.teachers = nn.ModuleList(teachers)
    
    def forward(self, x):
        outputs = [t(x) for t in self.teachers]
        return sum(outputs) / len(outputs)

# Replace main with ensemble distillation
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

    print("Starting Teaching Ensemble training.")
    torch.manual_seed(0)
    device = get_device()
    logging.info(f'Using device: {device}')
    trainloader, testloader = get_dataset_loader()

    # Initialize student
    student = timm.create_model('ghostnetv3', width=args.student_width, num_classes=10)
    init_weights_kaiming(student)
    student.to(device)

    # Initialize teachers
    teacher_inits = [densenet161, vgg13_bn, resnet50, inception_v3]
    teacher_ckpts = [
        "experiments/densenet/densenet161_seed0/best_model.pth",
        "experiments/vgg13/vgg13_bn_seed0/best_model.pth",
        "experiments/resnet/resnet50_seed0/best_model.pth",
        "experiments/inceptionv3/inception_v3_seed0/best_model.pth"
    ]

    teachers = []
    for init, ckpt in zip(teacher_inits, teacher_ckpts):
        tm = init(pretrained=False, num_classes=10)
        tm.load_state_dict(torch.load(ckpt, map_location=device))
        tm.to(device)
        tm.eval()
        teachers.append(tm)
    ensemble_teacher = EnsembleTeacher(teachers)
    ensemble_teacher.to(device)
    ensemble_teacher.eval()

    # Setup training utilities
    criterion = DistillationLoss(temperature=1.0, alpha=0.5)
    optimizer = get_optimizer(student)
    scheduler = get_scheduler(optimizer, training_length=len(trainloader))

    # Resume from checkpoint if exists
    checkpoint_path = 'ensemble_ghostnetv3_gnd_cifar10_checkpoint.pth'
    start_epoch = 1
    best_acc = 0.0
    ckpt_path = 'ensemble_ghostnetv3_gnd_cifar10.pth'
    
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        logging.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']}, best_acc {best_acc:.2f}%)")

    for epoch in range(start_epoch, EPOCHS + 1):
        train(student, device, trainloader, criterion, optimizer, scheduler, epoch, teacher_model=ensemble_teacher)
        acc = evaluate(student, device, testloader, nn.CrossEntropyLoss())

        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), ckpt_path)
            logging.info(f'New best accuracy: {best_acc:.2f}%, model saved to {ckpt_path}')

        # Save checkpoint to resume training
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
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
