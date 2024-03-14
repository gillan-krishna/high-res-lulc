import lightning as pl
import segmentation_models_pytorch as smp
# from src.losses import JaccardLoss
from losses import JaccardLoss, FocalLoss
from metrics import fscore
import torch
import neptune

class LitUNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.Unet(encoder_name='efficientnet-b0', encoder_weights='imagenet', in_channels=3, classes=9, activation='softmax')
        self.loss_fn = JaccardLoss()

    def training_step(self, batch, batch_idx):
        # sourcery skip: class-extract-method, inline-immediately-returned-variable
        x, y = batch
        x_hat = self.model(x)
        loss = self.loss_fn(x_hat, y)
        fs = fscore(x_hat, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/fscore", fs, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = self.loss_fn(x_hat, y)
        fs = fscore(x_hat, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/fscore", fs, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # sourcery skip: inline-immediately-returned-variable
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        return optimizer
    