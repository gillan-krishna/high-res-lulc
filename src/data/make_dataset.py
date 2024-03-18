import zipfile


def make_dataset(pth_datazip, dir_dataset):
    with zipfile.ZipFile(pth_datazip, "r") as zip_ref:
        zip_ref.extractall(dir_dataset)


if __name__ == "__main__":
    DATA = "data/raw/OpenEarthMap.zip"
    DATA_DIR = "data/input"
    make_dataset(DATA, DATA_DIR)
