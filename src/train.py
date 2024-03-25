import warnings


warnings.filterwarnings("ignore")

import os
import torch
from pathlib import Path
import numpy as np
import segmentation_models_pytorch as smp
import lightning as pl
from networks.UNet import LitUNet
from data.dataset import OEMDataLoader, OEMDataset
from torch.utils.data import DataLoader

from lightning.pytorch.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.tuner.tuning import Tuner
from pathlib import Path

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    model = LitUNet(
        arch="unetplusplus", encoder_name="efficientnet-b4", attention=False, lr=3e-4
    )
    oem = OEMDataLoader(batch_size=24, num_classes=9)

    neptune_logger = NeptuneLogger(
        project="gillan/lulc",
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNmRkMjQ3YS1iMzlkLTRjMGUtYmY2OS1iYzBiM2E3NmI3NWYifQ==",
        log_model_checkpoints=False,
    )

    checkpoint_dir = Path("/home/ubuntu/hrl/high-res-lulc/models")

    # trainer = pl.Trainer(fast_dev_run=True, accelerator='gpu', deterministic=True)
    # trainer = pl.Trainer(overfit_batches=1, logger=mlf_logger, accelerator='gpu', deterministic=True)
    trainer = pl.Trainer(
        logger=neptune_logger,
        accelerator="gpu",
        default_root_dir=checkpoint_dir,
        deterministic=True,
        # precision=16,
        profiler="pytorch",
        accumulate_grad_batches=4,
        # callbacks=checkpoint_callback,yo
        enable_checkpointing=True,
    )
    
    trainer.fit(model=model, datamodule=oem, ckpt_path='/home/ubuntu/hrl/high-res-lulc/.neptune/Untitled/LUL-52/checkpoints/epoch=90-step=48555.ckpt')