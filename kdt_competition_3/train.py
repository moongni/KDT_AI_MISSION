import os, sys, time, datetime, random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from utils import *
from config import *
from models import create_model


# Data 불러오기
train_df = pd.read_csv(TRAIN_DF_PATH)
test_df = pd.read_csv(TEST_DF_PATH)

train_dset = ObjDetectionDataset(TRAIN_PATH, train_df, get_transform())
test_dset = ObjDetectionDataset(TEST_PATH, test_df, get_transform())
print(len(train_dset), len(test_dset))

train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
dataloaders = {
    'train': train_loader,
    'test': test_loader
}


# Model 불러오기
model = create_model()
model = model.to(DEVICE)

# get the model params
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

best_model = train(model, optimizer, dataloaders, lr_scheduler)

