import cv2
import numpy as np
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
import scipy.misc

from config import (DEVICE, CLASSES_INFO, IMG_SIZE)


def get_transform() -> T.Compose:
    transforms = []
    transforms.append(T.Resize((IMG_SIZE, IMG_SIZE))),
    transforms.append(T.ToTensor()),
    transforms.append(T.Normalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225)))
    return T.Compose(transforms)


def collate_fn(batch):
    """
    collate function for the ObjectDetectionDataSet.
    Only used by the dataloader.
    """
    return tuple(zip(*batch))


def show_transformed_image(train_loader):
    """
    This function shows the transformed images from the `train_loader`
    Help to check whether teh transformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config/config.py
    """
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box_num, box in enumerate(boxes):
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255), 2)
                cv2.putText(sample, CLASSES_INFO[labels[box_num]],
                            (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 2)
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


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