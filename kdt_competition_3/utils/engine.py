import os
import time
import copy

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import *
from .utils import invTrans

def train(model, optimizer, dataloaders, lr_scheduler=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    train_loss_list = []
    val_loss_list = []
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        train_loss = train_one_epoch(model, optimizer, dataloaders['train'])
        train_loss_list.append(train_loss)

        if lr_scheduler is not None:
            lr_scheduler.step()

        val_loss = evaluate(model, dataloaders['test'])
        val_loss_list.append(val_loss)
        patience += 1

        if best_loss > val_loss:
            print(f"Best Model detection Save model state")
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience = 0

        # Early Stop
        if patience > PATIENCE:
            print(f"Ealry Stoping current epoch: {epoch}")
            break
        
        print('=' * 20)
        
    time_elapsed = time.time() - since
    print(f"Training compelet in {time_elapsed // 60}m {time_elapsed % 60}s")    
    print(f"Best Test Loss: {best_loss:.4f}")

    model.load_state_dict(best_model_wts)
    torch.save(model, os.path.join(MODEL_SAVE_PATH, 'best_detector.pth'))

    if VISUALIZE_EVALUATED_IMAGE:
        detect_object(model, dataloaders['test'], CONFIDENCE)

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
        prog_bar.set_description(desc=f"Train Loss: {losses_value:.4f}")

    epoch_loss = running_loss / len(dataloader)
    print(f"TRAIN LOSS: {epoch_loss:.4f}")

    return epoch_loss


def evaluate(model, dataloader):
    # model.eval()
    running_loss = 0.
    prog_bar = tqdm(dataloader, total=len(dataloader))
    for images, targets in prog_bar:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            # pred = model(images)
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        losses_value = losses.item()
        running_loss += losses_value * len(images)
        prog_bar.set_description(desc=f"Test Loss: {losses_value:.4f}")
    
    epoch_loss = running_loss / len(dataloader)
    print(f"TEST LOSS: {epoch_loss:.4f}")

    return epoch_loss


def get_prediction(model, image, confidence):
    """
    get_prediction
        parameters:
        - model - nn.Module to detect object
        - image - input image
        - confidence - threshold value for prediction score
        method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
            are chosen.
    """
    model.eval()
    with torch.no_grad():
        pred = model([image])

    pred_class = [CLASS_INFO[i] for i in pred[0]['labels'].cpu().numpy()]
    pred_boxes = [(i[0], i[1], i[2], i[3]) for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())

    pred_t = [x for x in pred_score if x > confidence]
    t = len(pred_t) - 1

    pred_class = pred_class[:t+1]
    pred_boxes = pred_boxes[:t+1]
    pred_score = pred_score[:t+1]

    return pred_class, pred_boxes, pred_score


def detect_object(model, dataloader, confidence=0.7):
    """
    object_detection_api
        parameters:
        - model - nn.Module to detect object
        - dataloader - torch.utils.data.DataLoader, input image DataLoader
        - confidence - threshold value for prediction score
        method:
        - prediction is obtained from get_prediction method
        - for each prediction, bounding box is drawn and text is written 
            with opencv
        - the final image is displayed
    """
    print(f"Detect Object running ...")
    for images, targets in dataloader:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.cpu().numpy() for k, v in t.items()} for t in targets]

        for image, target in zip(images, targets):
            fig = plt.figure()
            pred_class, pred_boxes, pred_score = get_prediction(model, image, confidence)
            image = image.cpu()
            image = invTrans(image).permute(1, 2, 0)
            plt.axis('off')
            plt.imshow(image)
            for i, pred_box in enumerate(pred_boxes):
                # pred box color blue
                plt.plot([pred_box[0], pred_box[0]], [pred_box[1], pred_box[3]], c='b')
                plt.plot([pred_box[0], pred_box[2]], [pred_box[1], pred_box[1]], c='b')
                plt.plot([pred_box[0], pred_box[2]], [pred_box[3], pred_box[3]], c='b')
                plt.plot([pred_box[2], pred_box[2]], [pred_box[1], pred_box[3]], c='b')
                plt.text(pred_box[0], (pred_box[1] - 10), s=pred_class[i], c='b')
                plt.text(pred_box[0], (pred_box[1] + 10), s=pred_score[i], c='b')
            for i, box in enumerate(target['boxes']):
                # real box color red
                plt.plot([box[0], box[0]], [box[1], box[3]], c='r')
                plt.plot([box[0], box[2]], [box[1], box[1]], c='r')
                plt.plot([box[0], box[2]], [box[3], box[3]], c='r')
                plt.plot([box[2], box[2]], [box[1], box[3]], c='r')
                plt.text(box[2], (box[1] - 10), s=CLASS_INFO[target['labels'][i]], c='r')
            
            title = str(target['image_id'].item()).zfill(4)
            plt.savefig(os.path.join(OUTPUT_PATH, title + '.png'))