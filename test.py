"""
    [AUE8088] PA1: Image Classification
        - To run: (aue8088) $ python test.py --ckpt_file wandb/aue8088-pa1/ygeiua2t/checkpoints/epoch\=19-step\=62500.ckpt
"""
# Python packages
import argparse

# PyTorch & Pytorch Lightning
from lightning import Trainer
from torch.utils.flop_counter import FlopCounterMode
import torch

# Custom packages
from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import hydra
from omegaconf import DictConfig

torch.set_float32_matmul_precision('medium')


@hydra.main(config_path="src/config", config_name="config")
def main(cfg: DictConfig):

    model = SimpleClassifier(
        model_name = cfg.model_name,
        num_classes = cfg.num_classes,
        cfg = cfg
    )

    datamodule = TinyImageNetDatasetModule(
        batch_size = 1,
        cfg = cfg
    )

    trainer = Trainer(
        accelerator = cfg.accelerator,
        devices = cfg.devices,
        precision = cfg.precision_str,
        benchmark = True,
        inference_mode = True,
        logger = False,
    )
    
    if cfg.ckpt_file:
        trainer.validate(model, ckpt_path = cfg.ckpt_file, datamodule = datamodule)
    else:
        print("No checkpoint file provided.")

    # FLOP counter
    x, y = next(iter(datamodule.test_dataloader()))
    flop_counter = FlopCounterMode(model, depth=1)

    with flop_counter:
        model(x)

if __name__ == "__main__":
    main()