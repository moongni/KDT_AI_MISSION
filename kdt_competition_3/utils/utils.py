import numpy as np
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
import scipy.misc

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


def save_loss_plot(OUT_DIR, train_loss, val_loss):
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    train_ax.plot(train_loss, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    valid_ax.plot(val_loss, color='tab:red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')
    figure_1.savefig(f"{OUT_DIR}/train_loss.png")
    figure_2.savefig(f"{OUT_DIR}/valid_loss.png")
    print('SAVING PLOTS COMPLETE...')
    plt.close('all')


if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader

    from config import *
    from dataset import ObjDetectionDataset 
    
    
    train_df = pd.read_csv(os.path.join(TRAIN_PATH, 'train_output.csv'))
    test_df = pd.read_csv(os.path.join(TEST_PATH, 'test_output.csv'))

    transform = get_transform()
    train_dset = ObjDetectionDataset(TRAIN_PATH, train_df)