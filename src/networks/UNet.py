import lightning as pl
import segmentation_models_pytorch as smp

from losses import JaccardLoss, FocalLoss
from metrics import fscore

import torch

torch.set_float32_matmul_precision("medium")


class LitUNet(pl.LightningModule):
    def __init__(
        self,
        arch: str = "unetplusplus",
        encoder_name: str = "efficientnet-b3",
        attention: str = False,
        lr=3e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        if attention:
            attention_type = "scse"
        else:
            attention_type = None
        if arch == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=9,
                activation="softmax",
                decoder_attention_type=attention_type,
            )
        elif arch == "unetplusplus":
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=9,
                activation="softmax",
                decoder_attention_type=attention_type,
            )
        else:
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=9,
                activation="softmax",
                decoder_attention_type=attention_type,
            )

        self.loss_fn = JaccardLoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = self.loss_fn(x_hat, y)
        fs = fscore(x_hat, y)
        self.log("train/loss", loss.detach(), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/fscore", fs.detach(), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = self.loss_fn(x_hat, y)
        fs = fscore(x_hat, y)
        self.log("val/loss", loss.detach(), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/fscore", fs.detach(), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # sourcery skip: inline-immediately-returned-variable
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
