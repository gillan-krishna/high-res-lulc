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

if __name__ == '__main__':
    model = LitUNet()
    oem = OEMDataLoader()

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

    neptune_logger = NeptuneLogger(
    project="gillan/lulc",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNmRkMjQ3YS1iMzlkLTRjMGUtYmY2OS1iYzBiM2E3NmI3NWYifQ==",
    log_model_checkpoints=False
    )
    # trainer = pl.Trainer(fast_dev_run=True)
    trainer = pl.Trainer(overfit_batches=1, logger=neptune_logger)
    # trainer = pl.Trainer(logger=neptune_logger)
    trainer.fit(model=model, datamodule=oem)
    # trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)