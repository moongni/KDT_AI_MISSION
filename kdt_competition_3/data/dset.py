import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image


class ObjDetectionDataset(Dataset):
    def __init__(
        self,
        root: str,
        df: Optional[pd.DataFrame] = None,
        transform: Optional[nn.Module] = None,
        mode: str = 'train'
    ) -> None:
        root = os.path.join(root, mode)

        if mode not in ['train', 'test']:
            raise Exception(f"Invalid mode, mode should be in ['train', 'test'] but input is {mode}")
        if os.path.isdir(root):
            raise Exception(f"Invalid root path: {root}")
        if mode == 'train' and df is not None:
            df = self.preprocessing(df)

        self.root = root
        self.df = df
        self.transform = transform
        self.mode = mode
        
        self.imgs = list(sorted([filename for filename in os.listdir(root) if '.csv' not in filename]))        

    def preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        new_data = {
            'label_idx': [],
            'x_center': [],
            'y_center': [],
            'w': [],
            'h': [],
            'bbox': []
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
        target = None
        
        # get image from filepath
        if self.mode == 'train':
            data = self.df.iloc[idx]
            file_name = data['filename']
            img_path = os.path.join(self.root, str(file_name).zfill(4) + '.jpg')
            
        else:
            img_path = os.path.join(self.root, self.imgs[idx])

        image = np.asarray(Image.open(img_path))
        image_shape = image.shape
        
        # get bounding box
        boxes = []
        if self.mode == 'train':
            target = {}
            data = self.df.iloc[idx]
            for i in range(len(data['label_idx'])):
                y_center = int(image_shape[0] * data['y_center'])
                x_center = int(image_shape[1] * data['x_center'])
                
        if self.transform is not None:
            image = self.transform(image)
            
        return image, target