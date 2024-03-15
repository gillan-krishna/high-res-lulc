import numpy as np
import torchvision.transforms.functional as TF


class ToTensor:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, sample):
        msks = [(sample["mask"] == v) * 1 for v in self.classes]
        sample["mask"] = TF.to_tensor(np.stack(msks, axis=-1))
        sample["image"] = TF.to_tensor(sample["image"])
        return sample
