import logging
import os
import torch
import torch.nn as nn
import argparse
from models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from utils import (
    train,
    evaluate,
    get_device,
    get_dataset_loader,
    get_optimizer,
    get_scheduler,
    EPOCHS
)

def parse_args():
    parser = argparse.ArgumentParser(description="VGG Train")
    parser.add_argument("--model", type=str, choices=["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"])
    parser.add_argument("--outdir", type=str, default="VGG_X")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    return parser.parse_args()

def build_model(name:str, device, pretrained:bool):
    if name == "vgg11_bn":
        return vgg11_bn(pretrained=pretrained, device=device, num_classes=10)
    if name == "vgg13_bn":
        return vgg13_bn(pretrained=pretrained, device=device, num_classes=10)
    if name == "vgg16_bn":
        return vgg16_bn(pretrained=pretrained, device=device, num_classes=10)
    if name == "vgg19_bn":
        return vgg19_bn(pretrained=pretrained, device=device, num_classes=10)
    raise ValueError(f"Unknown Model: {name}")

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
    torch.manual_seed(args.seed)

    if args.run_name is None:
        run_name = f"{args.model}_seed{args.seed}"
    else:
        run_name = f"{args.run_name}_seed{args.seed}"
    
    run_name = run_name.replace(" ", "_")
    run_dir = os.path.join(args.outdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, f"{args.model}_train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True
    )

    logging.info(f"Starting {args.model} training")
    logging.info(f"Args: {args.model}, pretrained: {args.pretrained}, seed: {args.seed}")
    device = get_device()

    logging.info(f"Using device: {device}")

    trainloader, testloader = get_dataset_loader()

    model = build_model(args.model, device=device, pretrained=args.pretrained).to(device)
    log_param_table(model, name=args.model)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, training_length=len(trainloader))

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
    
    for  epoch in range(start_epoch, EPOCHS + 1):
        train(model, device, trainloader, criterion, optimizer, scheduler, epoch)
        acc = evaluate(model, device, testloader, criterion)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'New best accuracy: {best_acc:.2f}%, model saved.')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }
        torch.save(checkpoint, checkpoint_path)

    logging.info(f'Training complete. Best Test Accurcy: {best_acc:.2f}%')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()





