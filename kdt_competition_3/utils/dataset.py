import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import matplotlib.pyplot as plt

from config import config


class ObjDetectionDataset(Dataset):
    def __init__(
        self,
        root: str,
        df: Optional[pd.DataFrame],
        transform: Optional[nn.Module] = None
    ) -> None:
        if not os.path.isdir(root):
            raise Exception(f"Invalid root path: {root}")
        if df is not None:
            df = self.preprocessing(df)

        self.root = root
        self.df = df
        self.transform = transform

    def preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        new_data = {
            'label_idx': [],
            'x_center': [],
            'y_center': [],
            'w': [],
            'h': []
        }

        for i in range(len(df)):
            label = df.iloc[i]['label']
            # multi labels
            if '\n' in label:
                label = label.split('\n')
            else:
                label = [label]

            label_idx = []
            x_center = []
            y_center = []
            w = []
            h = []
            for l in label:
                idx, x_c, y_c, width, height = l.split(' ')
                label_idx.append(int(idx))
                x_center.append(float(x_c))
                y_center.append(float(y_c))
                w.append(float(width))
                h.append(float(height))

            new_data['label_idx'].append(label_idx)
            new_data['x_center'].append(x_center)
            new_data['y_center'].append(y_center)
            new_data['w'].append(w)
            new_data['h'].append(h)

        new_data = pd.DataFrame(new_data)
        return pd.concat([df, new_data], axis=1)
        
    def __getitem__(self, idx):
        # get image from filepath
        data = self.df.iloc[idx]
        file_name = data['filename']
        img_path = os.path.join(self.root, str(file_name).zfill(4) + '.jpg')

        # open Image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255
        h, w, _ = image.shape

        # get labels
        labels = np.array(data['label_idx']) + 1  # 0 index -> background
        
        num_objs = len(labels)

        # get bounding box
        x_center = (np.array(data['x_center']).reshape(-1, 1) * w).astype(np.int32)
        y_center = (np.array(data['y_center']).reshape(-1, 1) * h).astype(np.int32)
        width = (np.array(data['w']) * w).reshape(-1, 1) // 2
        height = (np.array(data['h']) * h).reshape(-1, 1) // 2
        x_0 = x_center - width
        x_1 = x_center + width
        y_0 = y_center - height
        y_1 = y_center + height
        boxes = np.hstack((x_0, y_0, x_1, y_1))

        if self.transform:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            transformed = self.transform(**sample)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['labels'])
        else:
            image = np.asarray(image)
            image = ToTensorV2(p=1.0)(image)

        # get boxes area
        area = ((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]))
        
        # get iscrowd - 여러 인스턴스가 있는지
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target['boxes'] = torch.from_numpy(boxes).float()
        target['labels'] = torch.from_numpy(labels).to(torch.int64)
        target['image_id'] = torch.tensor([file_name])
        target['area'] = torch.from_numpy(area).float()
        target['iscrowd'] = iscrowd
        
        return image, target
    
    def __len__(self):
        return len(self.df)
    

if __name__ == "__main__":
    from config import *
    from utils import TrainTransform

    train_df = pd.read_csv(os.path.join(TRAIN_PATH, 'train_output.csv'))
    test_df = pd.read_csv(os.path.join(TEST_PATH, 'test_output.csv'))

    train_dset = ObjDetectionDataset(TRAIN_PATH, train_df, transform=TrainTransform())

    print(len(train_dset))
    iter_ = iter(train_dset)
    next(iter_)
    next(iter_)
    next(iter_)
    next(iter_)
    image, target = next(iter_)
    print(f"Image shape:", image.size())
    print(f"target:", target)

    import matplotlib.pyplot as plt

    
    plt.imshow(image.permute(1, 2, 0).numpy())
    for boxes in target['boxes']:
        x_0, y_0, x_1, y_1 = boxes.cpu().tolist()
        plt.plot([x_0, x_1], [y_0, y_0], c='b')
        plt.plot([x_0, x_0], [y_0, y_1], c='b')
        plt.plot([x_1, x_1], [y_1, y_0], c='b')
        plt.plot([x_0, x_1], [y_1, y_1], c='b')

    plt.show()
