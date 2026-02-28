from typing import Type, Any
import os
import math
import logging
import argparse

import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.cli import instantiate_class
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger

from torchmetrics import MetricCollection, Accuracy


# =========================
# Config (kept as-is)
# =========================
config = {
    'seed': 2021,
    'trainer': {
        'max_epochs': 100,
        'accelerator': "auto",
        'accumulate_grad_batches': 1,
        'fast_dev_run': False,
        'num_sanity_val_steps': 0,
        # You can add:
        # 'log_every_n_steps': 50,
    },
    'data': {
        'dataset_name': 'cifar10',
        'batch_size': 32,
        'num_workers': 4,
        'size': [224, 224],
        'data_root': 'data',
        'valid_ratio': 0.1
    },
    'model': {
        'backbone_init': {
            'model': 'efficientnet_v2_s_in21k',
            'nclass': 0,  # do not change this
            'pretrained': True,
        },
        'optimizer_init': {
            'class_path': 'torch.optim.SGD',
            'init_args': {
                'lr': 0.01,
                'momentum': 0.95,
                'weight_decay': 0.0005
            }
        },
        'lr_scheduler_init': {
            'class_path': 'torch.optim.lr_scheduler.CosineAnnealingLR',
            'init_args': {
                'T_max': 0  # no need to change this
            }
        }
    }
}


# =========================
# NEW: argparse for outdir/run_name/log cadence
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="EfficientNetV2 Lightning CIFAR")
    p.add_argument("--outdir", type=str, default="experiments")
    p.add_argument("--run_name", type=str, default="efficientnet_cifar10")
    p.add_argument("--seed", type=int, default=config["seed"])
    p.add_argument("--log_every_n_steps", type=int, default=100)
    return p.parse_args()


# =========================
# NEW: callback that writes lines like your manual loop
# =========================
class StepMetricLogger(Callback):
    """
    Logs like:
      2026-... INFO - Epoch 192 | Step 101/196 | Loss: 0.1571 | Acc: 95.79%
      2026-... INFO - Epoch 192 training completed
      2026-... INFO - Test Loss: 0.2983 | Test Acc: 90.33%
    """
    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def _steps_per_epoch(self, trainer) -> int:
        if trainer.num_training_batches is not None:
            return int(trainer.num_training_batches)
        return -1

    @staticmethod
    def _to_float(v):
        if v is None:
            return None
        if torch.is_tensor(v):
            return float(v.detach().cpu())
        return float(v)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.log_every_n_steps is None or self.log_every_n_steps <= 0:
            return

        step_in_epoch = batch_idx + 1
        steps_per_epoch = self._steps_per_epoch(trainer)

        if step_in_epoch == 1 or (step_in_epoch % self.log_every_n_steps == 0):
            m = trainer.callback_metrics
            loss = self._to_float(m.get("train/loss"))
            acc = self._to_float(m.get("train/top@1"))

            if loss is not None and acc is not None and steps_per_epoch > 0:
                logging.info(
                    f"Epoch {trainer.current_epoch + 1} | Step {step_in_epoch}/{steps_per_epoch} | "
                    f"Loss: {loss:.4f} | Acc: {acc * 100.0:.2f}%"
                )
            elif loss is not None:
                logging.info(
                    f"Epoch {trainer.current_epoch + 1} | Step {step_in_epoch} | Loss: {loss:.4f}"
                )

    def on_train_epoch_end(self, trainer, pl_module):
        logging.info(f"Epoch {trainer.current_epoch + 1} training completed")

    def on_test_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        loss = self._to_float(m.get("test/loss"))
        acc = self._to_float(m.get("test/top@1"))
        if loss is not None and acc is not None:
            logging.info(f"Test Loss: {loss:.4f} | Test Acc: {acc * 100.0:.2f}%")


# =========================
# Data
# =========================
class BaseDataModule(LightningDataModule):
    def __init__(self,
                 dataset_name: str,
                 dataset: Type[Any],
                 train_transform: Type[Any],
                 test_transform: Type[Any],
                 batch_size: int = 64,
                 num_workers: int = 4,
                 data_root: str = 'data',
                 valid_ratio: float = 0.1):
        super(BaseDataModule, self).__init__()
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.valid_ratio = valid_ratio
        self.num_classes = None
        self.num_step = None
        self.prepare_data()

    def prepare_data(self) -> None:
        train = self.dataset(root=self.data_root, train=True, download=True)
        test = self.dataset(root=self.data_root, train=False, download=True)
        self.num_classes = len(train.classes)
        self.num_step = len(train) // self.batch_size

        print('-' * 50)
        print('* {} dataset class num: {}'.format(self.dataset_name, len(train.classes)))
        print('* {} train dataset len: {}'.format(self.dataset_name, len(train)))
        print('* {} test dataset len: {}'.format(self.dataset_name, len(test)))
        print('-' * 50)

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            ds = self.dataset(root=self.data_root, train=True, transform=self.train_transform)
            self.train_ds, self.valid_ds = self.split_train_valid(ds)
        elif stage in (None, 'test', 'predict'):
            self.test_ds = self.dataset(root=self.data_root, train=False, transform=self.test_transform)

    def split_train_valid(self, ds):
        ds_len = len(ds)
        valid_ds_len = int(ds_len * self.valid_ratio)
        train_ds_len = ds_len - valid_ds_len
        return random_split(ds, [train_ds_len, valid_ds_len])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class CIFAR(BaseDataModule):
    def __init__(self, dataset_name: str, size: tuple, **kwargs):
        if dataset_name == 'cifar10':
            dataset, mean, std = CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        elif dataset_name == 'cifar100':
            dataset, mean, std = CIFAR100, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        train_transform, test_transform = self.get_trasnforms(mean, std, size)
        super(CIFAR, self).__init__(dataset_name, dataset, train_transform, test_transform, **kwargs)

    def get_trasnforms(self, mean, std, size):
        train = transforms.Compose([
            transforms.Resize(size),
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return train, test


# =========================
# Model
# =========================
class BaseVisionSystem(LightningModule):
    def __init__(self,
                 backbone_init: dict,
                 num_classes: int,
                 num_step: int,
                 max_epochs: int,
                 optimizer_init: dict,
                 lr_scheduler_init: dict):
        super(BaseVisionSystem, self).__init__()

        self.automatic_optimization = True
        self.num_step = num_step
        self.max_epochs = max_epochs

        # EfficientNetV2 backbone via torch.hub (needs internet once unless cached)
        self.backbone = torch.hub.load('hankyul2/EfficientNetV2-pytorch', **backbone_init)
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)

        self.optimizer_init_config = optimizer_init
        self.lr_scheduler_init_config = lr_scheduler_init
        self.criterion = nn.CrossEntropyLoss()

        # IMPORTANT: metrics should match num_classes (use num_classes, not hard-coded 10)
        metrics = MetricCollection({
            'top@1': Accuracy(top_k=1, task="multiclass", num_classes=num_classes),
            'top@5': Accuracy(top_k=5, task="multiclass", num_classes=num_classes),
        })
        self.train_metric = metrics.clone(prefix='train/')
        self.valid_metric = metrics.clone(prefix='valid/')
        self.test_metric = metrics.clone(prefix='test/')

    def forward(self, x):
        return self.fc(self.backbone(x))

    def compute_loss_eval(self, x, y):
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat

    # OLD (kept)
    # def shared_step(self, batch, metric, mode, add_dataloader_idx):
    #     x, y = batch
    #     loss, y_hat = self.compute_loss(x, y) if mode == 'train' else self.compute_loss_eval(x, y)
    #     metric = metric(y_hat, y)
    #     self.log_dict({f'{mode}/loss': loss}, add_dataloader_idx=add_dataloader_idx)
    #     self.log_dict(metric, add_dataloader_idx=add_dataloader_idx, prog_bar=True)
    #     return loss

    # NEW: correct metric update/compute + logs keys used by the callback
    def shared_step(self, batch, metric_collection: MetricCollection, mode: str):
        x, y = batch
        loss, y_hat = self.compute_loss_eval(x, y)

        metric_collection.update(y_hat, y)
        metric_vals = metric_collection.compute()

        # Log loss and metric dict so callback_metrics has:
        #   train/loss, train/top@1, ... valid/top@1, test/top@1, etc.
        self.log(f"{mode}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(metric_vals, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, self.train_metric, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, self.valid_metric, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, self.test_metric, "test")

    def on_train_epoch_end(self):
        self.train_metric.reset()

    def on_validation_epoch_end(self):
        self.valid_metric.reset()

    def on_test_epoch_end(self):
        self.test_metric.reset()

    def configure_optimizers(self):
        optimizer = instantiate_class([
            {'params': self.backbone.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 0.1},
            {'params': self.fc.parameters()},
        ], self.optimizer_init_config)

        lr_scheduler = {
            'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
            'interval': 'step'
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def update_and_get_lr_scheduler_config(self):
        if 'T_max' in self.lr_scheduler_init_config['init_args']:
            self.lr_scheduler_init_config['init_args']['T_max'] = self.num_step * self.max_epochs
        return self.lr_scheduler_init_config


def update_config(config, data):
    config['model']['num_classes'] = data.num_classes
    config['model']['num_step'] = data.num_step
    config['model']['max_epochs'] = config['trainer']['max_epochs']


def setup_file_logging(run_dir: str):
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True
    )
    logging.info(f"Logging to: {log_file}")


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    try:
        import pytorch_lightning as pl
        pl.seed_everything(args.seed, workers=True)
    except Exception:
        pass

    # Make a unique run directory so multiple SLURM jobs don’t collide
    # (timestamp + job id if present)
    job_id = os.environ.get("SLURM_JOB_ID", "nojid")
    ts = torch.tensor(int(torch.randint(0, 10**9, (1,)).item())).item()  # cheap unique-ish
    run_dir = os.path.join(args.outdir, f"{args.run_name}_{job_id}_{ts}")

    setup_file_logging(run_dir)
    logging.info(f"Args: outdir={args.outdir}, run_name={args.run_name}, seed={args.seed}, log_every_n_steps={args.log_every_n_steps}")

    # Data
    data = CIFAR(**config['data'])
    update_config(config, data)

    # Model
    model = BaseVisionSystem(**config['model'])

    # TensorBoard logger (kept, but point it at same run_dir)
    logger = TensorBoardLogger(
        save_dir=run_dir,
        name="tb"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid/top@1",
        mode="max",
        save_top_k=1,
        save_last=True,
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="{epoch}-{valid_top@1:.4f}",
        auto_insert_metric_name=False
    )

    step_logger = StepMetricLogger(log_every_n_steps=args.log_every_n_steps)

    # OLD (kept)
    # trainer = Trainer(
    #     logger=logger,
    #     callbacks=[checkpoint_callback],
    #     **config['trainer']
    # )

    # NEW trainer: includes step_logger and uses your config
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, step_logger],
        **config['trainer']
    )

    logging.info("Starting trainer.fit(...)")
    trainer.fit(model, data)

    logging.info("Starting trainer.test(ckpt_path='best')")
    trainer.test(ckpt_path='best')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()