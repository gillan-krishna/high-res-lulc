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
    oem = OEMDataLoader(batch_size=16)

    # mlf_logger = MLFlowLogger(experiment_name="lulc_b3")
    neptune_logger = NeptuneLogger(
        project="gillan/lulc",
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNmRkMjQ3YS1iMzlkLTRjMGUtYmY2OS1iYzBiM2E3NmI3NWYifQ==",
        log_model_checkpoints=False,
    )

    checkpoint_dir = Path("/home/ubuntu/hrl/high-res-lulc/models")
    # DEFAULTS used by the Trainer
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=checkpoint_dir,
    #     # filename= f'{epoch}-{train/loss}-{other_metric}
    #     save_top_k=2,
    #     save_weights_only=True,
    #     verbose=True,
    #     monitor='train/loss',
    #     mode='min',
    #     # prefix='lulc'
    # )

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
        # callbacks=checkpoint_callback,
        enable_checkpointing=True
    )
    # tuner = Tuner(trainer=trainer)
    # tuner.scale_batch_size(model=model, datamodule=oem)
    # tuner.lr_find(model=model,datamodule=oem, min_lr=3e-3, num_training=100)
    # torch.cuda.empty_cache()
    trainer.fit(model=model, datamodule=oem)

    # DATA_DIR = 'data/processing'

    # TRAIN_LIST = os.path.join(DATA_DIR,'train.txt')
    # VAL_LIST = os.path.join(DATA_DIR, 'val.txt')
    # TEST_LIST = os.path.join(DATA_DIR, 'test.txt')

    # fns = [f for f in Path(DATA_DIR).rglob('*png') if '/images/' in str(f)]
    # train_list = [str(f) for f in fns if '_'.join(f.name.split('_')[:-2])+'.tif' in np.loadtxt(TRAIN_LIST, dtype =str)]
    # val_list = [str(f) for f in fns if '_'.join(f.name.split('_')[:-2])+'.tif' in np.loadtxt(VAL_LIST, dtype =str)]

    # batch_size = 8

    # OEM_train = OEMDataset(img_list= train_list , testing=False, augm=None)
    # OEM_val = OEMDataset(img_list= val_list , testing=False, augm=None)

    # train_loader = DataLoader(OEM_train, batch_size=batch_size)
    # val_loader = DataLoader(OEM_val, batch_size=batch_size)
    # trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
