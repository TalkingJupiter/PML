import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import argparse
import models.ghostnetv3 as ghostnetv3
from models.resnet import resnet50, resnet34, resnet18
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
    parser = argparse.ArgumentParser(description="Training w/ Teacher for GhostNetV3")
    parser.add_argument("--student_width", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default="experiments")
    parser.add_argument("--run_name", type=str, default="Train_w_teacher_gnd1.0x")
    parser.add_argument("--teacher_run", type=str, default="experiments/resnet")
    parser.add_argument("--teacher_ckpt", type=str, default="best_model.pth")

    return parser.parse_args()

def main():
    args = parse_args()
    width = args.student_width
    run_dir = os.path.join(args.outdir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True
    )

    print("Starting Teaching Assistant training.")
    torch.manual_seed(0)

    device = get_device()
    logging.info(f'Using device: {device}')
    trainloader, testloader = get_dataset_loader()

    # Define distillation stages: (student_name, init_student_fn, teacher_name, init_teacher_fn)
    stages = [
        ('resnet34', lambda: resnet34(pretrained=False, device=device), 'resnet50', lambda: resnet50(pretrained=False, device=device)),
        ('resnet18', lambda: resnet18(pretrained=False, device=device), 'resnet34', lambda: resnet34(pretrained=False, device=device)),
        ('ghostnetv3', lambda: timm.create_model('ghostnetv3', width=width, num_classes=10), 'resnet18', lambda: resnet18(pretrained=False, device=device)),
    ]
    
    # Master checkpoint for overall progress
    master_checkpoint_path = f'teacher_assistant_master_checkpoint_gnd_{width}.pth'
    start_stage = 0
    completed_stages = {}
    
    # Load master checkpoint if exists
    if os.path.isfile(master_checkpoint_path):
        master_checkpoint = torch.load(master_checkpoint_path, map_location=device)
        start_stage = master_checkpoint['current_stage']
        completed_stages = master_checkpoint.get('completed_stages', {})
        logging.info(f"Loaded master checkpoint. Resuming from stage {start_stage}")
    
    prev_checkpoint = None
    for stage_idx in range(start_stage, len(stages)):
        student_name, init_student, teacher_name, init_teacher = stages[stage_idx]
        
        # Initialize student and teacher
        student = init_student()
        init_weights_kaiming(student)
        student.to(device)

        # Load teacher from previous stage if available
        teacher = init_teacher()
        if stage_idx == 0:
            teacher_path = os.path.join(args.teacher_run, f"{teacher_name}_seed0", args.teacher_ckpt) #experiments/resnet50, resnet50_seed0, best_method.ckpt
            if not os.path.exists(teacher_path):
                raise FileNotFoundError(f"Teacher checkpiunt did not found {teacher_path}")
            teacher.load_state_dict(torch.load(teacher_path, map_location=device))
            logging.info(f"Loaded teacher {teacher_name} from {teacher_path}")
        
        if stage_idx > 0:
            prev_student_name = stages[stage_idx - 1][0]
            prev_checkpoint = f"assistant_{prev_student_name}.pth"
            if os.path.isfile(prev_checkpoint):
                teacher.load_state_dict(torch.load(prev_checkpoint, map_location=device))
                logging.info(f"Loaded teacher {teacher_name} from {prev_checkpoint}")
        teacher.to(device)
        teacher.eval()

        logging.info(f"Stage {stage_idx}: {teacher_name} -> {student_name}")
        criterion = DistillationLoss(temperature=1.0, alpha=0.5)
        optimizer = get_optimizer(student)
        scheduler = get_scheduler(optimizer, training_length=len(trainloader))

        # Stage-specific checkpoint
        stage_checkpoint_path = f"assistant_{student_name}_checkpoint.pth"
        ckpt_path = f"assistant_{student_name}.pth"
        
        start_epoch = 1
        best_acc = 0.0
        
        # Load stage checkpoint if exists
        if os.path.isfile(stage_checkpoint_path):
            stage_checkpoint = torch.load(stage_checkpoint_path, map_location=device)
            student.load_state_dict(stage_checkpoint['model_state_dict'])
            optimizer.load_state_dict(stage_checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(stage_checkpoint['scheduler_state_dict'])
            start_epoch = stage_checkpoint['epoch'] + 1
            best_acc = stage_checkpoint['best_acc']
            logging.info(f"Loaded stage checkpoint for {student_name} (epoch {stage_checkpoint['epoch']}, best_acc {best_acc:.2f}%)")

        for epoch in range(start_epoch, EPOCHS + 1):
            train(student, device, trainloader, criterion, optimizer, scheduler, epoch, teacher_model=teacher)
            acc = evaluate(student, device, testloader, nn.CrossEntropyLoss())
            
            if acc > best_acc:
                best_acc = acc
                torch.save(student.state_dict(), ckpt_path)
                logging.info(f'New best accuracy for {student_name}: {best_acc:.2f}%, model saved to {ckpt_path}')

            # Save stage checkpoint
            stage_checkpoint = {
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'stage_idx': stage_idx,
                'student_name': student_name,
            }
            torch.save(stage_checkpoint, stage_checkpoint_path)
            
        # Mark stage as completed
        completed_stages[student_name] = {
            'best_acc': best_acc,
            'checkpoint_path': ckpt_path
        }
        
        # Update master checkpoint
        master_checkpoint = {
            'current_stage': stage_idx + 1,
            'completed_stages': completed_stages,
        }
        torch.save(master_checkpoint, master_checkpoint_path)
        logging.info(f"Stage {stage_idx} ({student_name}) completed with best accuracy: {best_acc:.2f}%")
        
        # Clean up stage checkpoint after completion
        if os.path.isfile(stage_checkpoint_path):
            os.remove(stage_checkpoint_path)

    logging.info(f'All distillation stages complete.')
    
    # Clean up master checkpoint after all stages complete
    if os.path.isfile(master_checkpoint_path):
        os.remove(master_checkpoint_path)
        
    # Print final summary
    logging.info("=== Teacher Assistant Training Summary ===")
    for stage_name, stage_info in completed_stages.items():
        logging.info(f"{stage_name}: Best Accuracy = {stage_info['best_acc']:.2f}%")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
