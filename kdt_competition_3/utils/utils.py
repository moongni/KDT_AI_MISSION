import os
import random

import numpy as np
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt

from config import *


invTrans = T.Normalize((-1 * MEAN) / STD, (1.0 / STD))


def get_common_transform() -> T.Compose:
    transforms = []
    transforms.append(T.Resize((IMG_SIZE, IMG_SIZE))),
    transforms.append(T.ToTensor()),
    transforms.append(T.Normalize(MEAN, STD))
    return T.Compose(transforms)


def get_augmented_transform() -> T.Compose:
    transforms = []
    transforms.append(T.RandomApply([
        T.ColorJitter(brightness=0.1, contrast=0.2),
    ], 0.3))
    transforms.append(T.RandomApply([
        T.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 2))
    ], 0.3))
    return T.Compose(transforms)


def transform(image, target, train:bool):
    transforms = get_common_transform()
    # augmented = get_augmented_transform()
    image = transforms(image)
    if train:
        # image = augmented(image)
        if random.random() > 0.5:
            image = T.RandomHorizontalFlip(1)(image)
            target['x_center'] = [1 - x for x in target['x_center']]
    
    return image, target


def collate_fn(batch):
    """
    collate function for the ObjectDetectionDataSet.
    Only used by the dataloader.
    """
    return tuple(zip(*batch))


def set_seed():
    """
    fix seed to control the random variable 
    """
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def visualize_losses(train_losses, test_losses):
    plt.figure(figsize=(12, 12))
    plt.plot(train_losses, c='b', label='train')
    plt.plot(test_losses, c='r', label='test')
    plt.title('Loss graph')
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.savefig(os.path.join(OUTPUT_PATH, 'loss_graph.png'))


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
    visualize_losses([random.random() for _ in range(10)], [random.random() for _ in range(10)])