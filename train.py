"""
    [AUE8088] PA1: Image Classification
        - To run: (aue8088) $ python train.py
        - For better flexibility, consider using LightningCLI in PyTorch Lightning
"""
# PyTorch & Pytorch Lightning
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning import Trainer
import torch

# Custom packages
from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

torch.set_float32_matmul_precision('medium')


@hydra.main(config_path="src/config", config_name="config")
def main(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    optimizer = config_dict['optimizer']
    scheduler = config_dict['scheduler']

    model = SimpleClassifier(
        model_name = cfg.model_name,
        num_classes = cfg.num_classes,
        optimizer_params = optimizer,
        scheduler_params = scheduler,
        cfg = cfg
    )

    datamodule = TinyImageNetDatasetModule(
        batch_size = cfg.batch_size,
        cfg = cfg
    )

    wandb_logger = WandbLogger(
        project = cfg.wandb.project,
        save_dir = cfg.wandb.save_dir,
        entity = cfg.wandb.entity,
        name = cfg.wandb.name,
    )

    trainer = Trainer(
        accelerator = cfg.accelerator,
        devices = cfg.devices,
        precision = cfg.precision_str,
        max_epochs = cfg.num_epochs,
        check_val_every_n_epoch = cfg.val_every_n_epoch,
        logger = wandb_logger,
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(save_top_k=1, monitor='accuracy/val', mode='max'),
        ],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.validate(ckpt_path='best', datamodule=datamodule)
    wandb.finish()

if __name__ == "__main__":
    main()