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

# from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.loggers import MLFlowLogger

# from dotenv import load_dotenv

# load_dotenv()
# api_key = os.getenv("NEPTUNE_API")
# api_key = os.environ('NEPTUNE_API')

if __name__ == "__main__":
    model = LitUNet()
    oem = OEMDataLoader()

    # neptune_logger = NeptuneLogger(
    # project="gillan/lulc",
    # api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNmRkMjQ3YS1iMzlkLTRjMGUtYmY2OS1iYzBiM2E3NmI3NWYifQ==",
    # # api_key=api_key,
    # log_model_checkpoints=False
    # )

    mlf_logger = MLFlowLogger(experiment_name="lulc")

    # trainer = pl.Trainer(fast_dev_run=True)
    # trainer = pl.Trainer(overfit_batches=1, logger=neptune_logger)
    trainer = pl.Trainer(overfit_batches=1, logger=mlf_logger)
    # trainer = pl.Trainer(logger=neptune_logger)


from lightning.pytorch.loggers import NeptuneLogger
# from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.tuner.tuning import Tuner
from pathlib import Path

if __name__ == "__main__":
    model = LitUNet(arch='unetplusplus', encoder_name='efficientnet-b3', attention=False, lr=0.04)
    oem = OEMDataLoader(batch_size=32)
    
    # mlf_logger = MLFlowLogger(experiment_name="lulc_b3")
    neptune_logger = NeptuneLogger(
    project="gillan/lulc",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNmRkMjQ3YS1iMzlkLTRjMGUtYmY2OS1iYzBiM2E3NmI3NWYifQ==",
    log_model_checkpoints=False
    )
    
    checkpoint_dir = Path('/home/ubuntu/hrl/high-res-lulc/models')

    # trainer = pl.Trainer(fast_dev_run=True, accelerator='gpu')
    # trainer = pl.Trainer(overfit_batches=1, logger=mlf_logger, accelerator='gpu')
    trainer = pl.Trainer(logger=neptune_logger, accelerator='gpu', default_root_dir=checkpoint_dir)
    # trainer = pl.Trainer(logger=mlf_logger, accelerator='gpu', default_root_dir='models/')
    tuner = Tuner(trainer=trainer)
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

