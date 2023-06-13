import os
import random

import numpy as np
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt

from config import *

import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2


class TrainTransform:
    def __init__(self):
        self.transforms = A.Compose([
                            A.OneOf([
                                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.5),
                                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
                            ], p=0.9),                      
                            A.ToGray(p=0.05),
                            A.HorizontalFlip(p=0.5), 
                            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.2),
                            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
                            ToTensorV2(p=1.0)
                        ], p=1.0, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __call__(self, **kwargs):
        return self.transforms(**kwargs)
    

class TestTransform:
    def __init__(self):
        self.transforms = A.Compose([
                            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
                            ToTensorV2(p=1.0)
                          ], p=1.0, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    def __call__(self, **kwargs):
        return self.transforms(**kwargs)
    

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
    """
    save loss graph
    """
    plt.figure(figsize=(12, 12))
    plt.plot(train_losses, c='b', label='train')
    plt.plot(test_losses, c='r', label='test')
    plt.title('Loss graph')
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    make_dir(OUTPUT_PATH)
    plt.savefig(os.path.join(OUTPUT_PATH, 'loss_graph.png'))


def make_dir(path) -> None:
    from pathlib import Path

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ...
    import pandas as pd
    from torch.utils.data import DataLoader

    from config import *
    from dataset import ObjDetectionDataset 
    
    
    train_df = pd.read_csv(os.path.join(TRAIN_PATH, 'train_output.csv'))
    test_df = pd.read_csv(os.path.join(TEST_PATH, 'test_output.csv'))

    train_dset = ObjDetectionDataset(TRAIN_PATH, train_df, TrainTransform())
    test_dset = ObjDetectionDataset(TEST_PATH, test_df, TestTransform())
    image, label = next(iter(test_dset))
    print(f"Image shape:", image.size())
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.show()
    # visualize_losses([random.random() for _ in range(10)], [random.random() for _ in range(10)])