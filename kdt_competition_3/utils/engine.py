import os
import time
import copy

import torch
from tqdm import tqdm

from config import (
    DEVICE, EPOCHS, OUTPUT_PATH, MODEL_SAVE_PATH
)


def train(model, optimizer, dataloaders, lr_scheduler=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        print('=' * 20)

        train_one_epoch(model, optimizer, dataloaders['train'])
        if lr_scheduler is not None:
            lr_scheduler.step()

        val_loss = val_one_epoch(model, dataloaders['test'])

        if best_loss > val_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"Training compelet in {time_elapsed // 60}m {time_elapsed % 60}s")    
    print(f"Best Test Loss: {best_loss:.4f}")

    model.load_state_dict(best_model_wts)
    torch.save(model, os.path.join(MODEL_SAVE_PATH, 'best_detector.pth'))

    return model


def train_one_epoch(model, optimizer, dataloader):
    model.train()
    running_loss = 0.

    prog_bar = tqdm(dataloader, total=len(dataloader))
    for images, targets in prog_bar:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        losses_value = losses.item()

        # backward propagation 
        losses.backward()
        optimizer.step()
        running_loss += losses_value * len(images)
        prog_bar.set_description(desc=f"Loss: {losses_value:.4f}")

    epoch_loss = running_loss / len(dataloader)
    print(f"TRAIN LOSS: {epoch_loss:.4f}")


def val_one_epoch(model, dataloader):
    running_loss = 0.
    for images, targets in dataloader:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item() * len(images)
    
    epoch_loss = running_loss / len(dataloader)
    print(f"TEST LOSS: {epoch_loss:.4f}")

    return epoch_loss
