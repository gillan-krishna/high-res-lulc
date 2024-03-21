import os
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from . import transforms
from PIL import Image
from skimage.io import imread
import math
from lightning import LightningDataModule
from pathlib import Path
import segmentation_models_pytorch as smp


def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    return np.eye(num_classes, dtype="uint8")[y]


def preprocess_fn(data, to_tensor, preprocess_input, num_classes):
    data["image"] = preprocess_input(np.array(data["image"], dtype="uint8"))
    # data["image"] = preprocess_input(np.array(data["image"], dtype="uint8"))
    # data['mask'] = to_categorical(np.array(data['mask'],dtype='uint8'), num_classes=num_classes)
    data = to_tensor(
        {
            "image": np.array(data["image"], dtype="uint8"),
            "mask": np.array(data["mask"], dtype="uint8"),
        }
    )
    return data


class OEMDataset(Dataset):
    def __init__(self, img_list: list, n_classes: int = 9, testing=False, augm=None):
        self.fn_imgs = [str(f) for f in img_list]
        self.fn_msks = [f.replace("images", "labels") for f in self.fn_imgs]
        self.augm = augm
        self.testing = testing
        self.classes = np.arange(n_classes).tolist()
        self.to_tensor = transforms.ToTensor(classes=self.classes)
        self.preprocess_input = smp.encoders.get_preprocessing_fn("efficientnet-b4")
        self.preprocess_input = smp.encoders.get_preprocessing_fn("efficientnet-b4")
        self.N_CLASSES = n_classes

    def __getitem__(self, idx):
        img = Image.fromarray(imread(self.fn_imgs[idx]))

        if not self.testing:
            msk = Image.fromarray(imread(self.fn_msks[idx]))
        else:
            msk = Image.fromarray(np.zeros(img.size[:2], dtype="uint8"))

        if self.augm is not None:
            data = self.augm({"image": img, "mask": msk})
        else:
            h, w = msk.size
            power_h = math.ceil(np.log2(h) / np.log2(2))
            power_w = math.ceil(np.log2(w) / np.log2(2))
            if 2**power_h != h or 2**power_w != w:
                img = img.resize((2**power_w, 2**power_h), resample=Image.BICUBIC)
                msk = msk.resize((2**power_w, 2**power_h), resample=Image.NEAREST)
            data = {"image": img, "mask": msk}

        data = preprocess_fn(
            data, self.to_tensor, self.preprocess_input, self.N_CLASSES
        )

        del msk, img, power_h, power_w
        return data["image"], data["mask"]

    def __len__(self):
        return len(self.fn_imgs)


class OEMDataLoader(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.save_hyperparameters()
        self.DATA_DIR = "data/processing"

        self.TRAIN_LIST = os.path.join(self.DATA_DIR, "train.txt")
        self.VAL_LIST = os.path.join(self.DATA_DIR, "val.txt")
        self.TEST_LIST = os.path.join(self.DATA_DIR, "test.txt")

        fns = [f for f in Path(self.DATA_DIR).rglob("*png") if "/images/" in str(f)]
        self.train_list = [
            str(f)
            for f in fns
            if "_".join(f.name.split("_")[:-2]) + ".tif"
            in np.loadtxt(self.TRAIN_LIST, dtype=str)
        ]
        self.val_list = [
            str(f)
            for f in fns
            if "_".join(f.name.split("_")[:-2]) + ".tif"
            in np.loadtxt(self.VAL_LIST, dtype=str)
        ]
        self.test_list = [
            str(f)
            for f in fns
            if "_".join(f.name.split("_")[:-2]) + ".tif"
            in np.loadtxt(self.TEST_LIST, dtype=str)
        ]
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == "fit":
            self.OEM_train = OEMDataset(
                img_list=self.train_list, testing=False, augm=None
            )
            self.OEM_val = OEMDataset(img_list=self.val_list, testing=False, augm=None)
        elif stage == "test":
            self.OEM_test = OEMDataset(img_list=self.test_list, testing=True, augm=None)
        elif stage == "predict":
            self.OEM_pred = OEMDataset(img_list=self.val_list, testing=False, augm=None)

    def train_dataloader(self):
        return DataLoader(self.OEM_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.OEM_val, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.OEM_pred, batch_size=1)
