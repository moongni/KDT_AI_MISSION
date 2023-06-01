import os
import random

import numpy as np
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt

from config import *


invTrans = T.Normalize((-1 * MEAN) / STD, (1.0 / STD))


def get_transform() -> T.Compose:
    transforms = []
    transforms.append(T.Resize((IMG_SIZE, IMG_SIZE))),
    transforms.append(T.ToTensor()),
    transforms.append(T.Normalize(MEAN, STD))
    return T.Compose(transforms)


def collate_fn(batch):
    """
    collate function for the ObjectDetectionDataSet.
    Only used by the dataloader.
    """
    return tuple(zip(*batch))


def set_seed():
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def plot_loss(train_losses, val_losses):
    ...


def make_dir(path) -> None:
    from pathlib import Path
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    import pandas as pd
    # from torch.utils.data import DataLoader

    # from config import *
    # from dataset import ObjDetectionDataset 
    
    
    # train_df = pd.read_csv(os.path.join(TRAIN_PATH, 'train_output.csv'))
    # test_df = pd.read_csv(os.path.join(TEST_PATH, 'test_output.csv'))

    # transform = get_transform()
    # train_dset = ObjDetectionDataset(TRAIN_PATH, train_df)