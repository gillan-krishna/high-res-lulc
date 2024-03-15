import itertools
import os
from pathlib import Path
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm
from PIL import Image
from patchify import patchify
import cv2

if __name__ == "__main__":
    DATA_DIR = "data/input/OpenEarthMap_wo_xBD"
    TRAIN_LIST = os.path.join(DATA_DIR, "train.txt")
    VAL_LIST = os.path.join(DATA_DIR, "val.txt")
    TEST_LIST = os.path.join(DATA_DIR, "test.txt")

    fns = [f for f in Path(DATA_DIR).rglob("*tif") if "/images/" in str(f)]
    PATCH_SIZE = 256
    PATCH_DIR = "data/processing"
    count = 0
    for fn in tqdm(fns):
        img = imread(fn)
        size_x = (img.shape[1] // PATCH_SIZE) * PATCH_SIZE
        size_y = (img.shape[0] // PATCH_SIZE) * PATCH_SIZE
        image = Image.fromarray(img)
        image = image.crop((0, 0, size_x, size_y))
        image = np.array(image)
        try:
            patches_img = patchify(image, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)

            for i, j in itertools.product(
                range(patches_img.shape[0]), range(patches_img.shape[1])
            ):
                single_patch_img = patches_img[i, j, :, :]
                single_patch_img = single_patch_img[0]
                cv2.imwrite(
                    PATCH_DIR + "/images/" + fn.name.split(".")[0] + f"_{i}_{j}.png",
                    single_patch_img,
                )
                count += 1
        except Exception:
            print(fn.name)

    fns = [f for f in Path(DATA_DIR).rglob("*tif") if "/labels/" in str(f)]
    for fn in tqdm(fns):
        img = imread(fn)
        size_x = (img.shape[1] // PATCH_SIZE) * PATCH_SIZE
        size_y = (img.shape[0] // PATCH_SIZE) * PATCH_SIZE
        image = Image.fromarray(img, "L")
        image = image.crop((0, 0, size_x, size_y))
        image = np.expand_dims(np.array(image), axis=-1)
        try:
            patches_img = patchify(image, (PATCH_SIZE, PATCH_SIZE, 1), step=PATCH_SIZE)

            for i, j in itertools.product(
                range(patches_img.shape[0]), range(patches_img.shape[1])
            ):
                single_patch_img = patches_img[i, j, :, :]
                single_patch_img = single_patch_img[0]
                cv2.imwrite(
                    PATCH_DIR + "/labels/" + fn.name.split(".")[0] + f"_{i}_{j}.png",
                    single_patch_img,
                )
        except Exception:
            print(fn.name)

    print("Total images count: ", count)
