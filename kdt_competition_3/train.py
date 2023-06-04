import os, json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pprint import pprint

from utils import *
from config import *
from models import *


# fix seed
set_seed()
print(f"Set Seed: {SEED}")
print(f"Torch Device: {DEVICE}")

# Data 불러오기
train_df = pd.read_csv(TRAIN_DF_PATH)
test_df = pd.read_csv(TEST_DF_PATH)

train_dset = ObjDetectionDataset(TRAIN_PATH, train_df, transform, train=True)
test_dset = ObjDetectionDataset(TEST_PATH, test_df, transform, train=False)
print(f"Load Dataset from \n {TRAIN_DF_PATH} \n {TEST_DF_PATH}")
print(f"Train data size: {len(train_dset)}, Test data size: {len(test_dset)}")

train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
dataloaders = {
    'train': train_loader,
    'test': test_loader
}

# Model 불러오기
model = fasterrcnn_resnet_50()
model = model.to(DEVICE)

# get the model params
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# Train model
best_model, train_loss_list, test_loss_list = train(model, optimizer, dataloaders, lr_scheduler)

# # Save best model
make_dir(OUTPUT_PATH)
torch.save(best_model, os.path.join(OUTPUT_PATH, 'best_detector.pth'))

map_result = mean_average_precision(best_model, test_loader)

print(f"Train score metrix")
pprint(map_result)

with open(os.path.join(OUTPUT_PATH, 'metrics.txt'), 'w') as f:
    f.write(json.dumps(map_result, indent='\t'))