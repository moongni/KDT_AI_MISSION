import os
import time
import copy
from typing import Tuple, List, Dict

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import *
from .utils import make_dir, visualize_losses


def train(model, optimizer, dataloaders, lr_scheduler=None):
    """Full train method"""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    train_loss_list = []
    test_loss_list = []
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        train_loss, test_loss = train_one_epoch(model, optimizer, dataloaders)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        patience += 1
        
        # Save Best model weight
        if best_loss > test_loss:
            print(f"Best Model detection Save model state")
            best_loss = test_loss
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

    # 모델이 추론한 결과 이미지 저장
    if VISUALIZE_EVALUATED_IMAGE:
        visualize_object_detection(model, dataloaders['test'], CONFIDENCE)
    if VISUALIZE_LOSS_GRAPH:
        visualize_losses(train_loss_list, test_loss_list)
        
    return model, train_loss_list, test_loss_list


def train_one_epoch(model, optimizer, dataloaders):
    """Train model in epoch"""
    model.train()
    train_loss = 0.
    test_loss = 0.
    # 각 에폭은 train, test 검증과정을 가짐
    for phase in ['train', 'test']:
        running_loss = 0.
        dataloader = dataloaders[phase]

        prog_bar = tqdm(dataloader, total=len(dataloader))
        for images, targets in prog_bar:
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            # train 과정일 때만 torch tensor 미분 계산
            with torch.set_grad_enabled(phase == 'train'):
                # forward propagation
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses_value = losses.item()
                # backward propagation 
                if phase == 'train':
                    losses.backward()
                    optimizer.step()

            running_loss += losses_value * len(images)
            prog_bar.set_description(desc=f"{phase} Loss: {losses_value:.4f}")

        epoch_loss = running_loss / len(dataloader)
        print(f"{phase} LOSS: {epoch_loss:.4f}")

        if phase == 'train':
            train_loss = epoch_loss
        else:
            test_loss = epoch_loss

    return train_loss, test_loss


def get_prediction(model, image, confidence) -> Tuple[List, List, List]:
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
        - class, box coordinates are obtained, but only prediction score > threshold are chosen.
    """
    model.eval()
    with torch.no_grad():
        pred = model([image])

    pred_class = list(pred[0]['labels'].cpu().numpy())
    pred_boxes = [(i[0], i[1], i[2], i[3]) for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())

    pred_t = [x for x in pred_score if x > confidence]
    t = len(pred_t) - 1

    pred_class = pred_class[:t+1]
    pred_boxes = pred_boxes[:t+1]
    pred_score = pred_score[:t+1]

    return pred_class, pred_boxes, pred_score


def visualize_object_detection(model, dataloader, confidence=0.7) -> None:
    """
    object_detection_api
        parameters:
        - model - nn.Module to detect object
        - dataloader - torch.utils.data.DataLoader, input image DataLoader
        - confidence - threshold value for prediction score
        method:
        - save prediction image is obtained from get_prediction method
        - for each prediction, bounding box is drawn and text is written with matplotlib
    """
    print(f"Detect Object ...")
    since = time.time()

    # 이미지 결과 Path 디렉토리 생성
    output_path = os.path.join(OUTPUT_PATH, 'images')
    if not os.path.isdir(output_path):
        make_dir(output_path)

    model.to(DEVICE)
    for images, targets in dataloader:
        images = list(image.to(DEVICE) for image in images)
        print(targets)
        targets = [{k: v.cpu().numpy() for k, v in t.items()} for t in targets]
        
        for image, target in zip(images, targets):
            fig = plt.figure()
            pred_class, pred_boxes, pred_score = get_prediction(model, image, confidence)
            # class idx -> class name
            pred_class = [CLASS_INFO[i] for i in pred_class]
            image = image.cpu()
            image = image.permute(1, 2, 0).numpy()
            plt.axis('off')
            plt.imshow(image)
            for i, pred_box in enumerate(pred_boxes):
                # pred box color blue
                plt.plot([pred_box[0], pred_box[0]], [pred_box[1], pred_box[3]], c='b')
                plt.plot([pred_box[0], pred_box[2]], [pred_box[1], pred_box[1]], c='b')
                plt.plot([pred_box[0], pred_box[2]], [pred_box[3], pred_box[3]], c='b')
                plt.plot([pred_box[2], pred_box[2]], [pred_box[1], pred_box[3]], c='b')
                plt.text(pred_box[0], (pred_box[1] - 10), s=pred_class[i], c='b')
                plt.text(pred_box[0], (pred_box[1] + 20), s=pred_score[i], c='b')
            for i, box in enumerate(target['boxes']):
                # real box color red
                plt.plot([box[0], box[0]], [box[1], box[3]], c='r')
                plt.plot([box[0], box[2]], [box[1], box[1]], c='r')
                plt.plot([box[0], box[2]], [box[3], box[3]], c='r')
                plt.plot([box[2], box[2]], [box[1], box[3]], c='r')
                plt.text(box[2], (box[1] - 10), s=CLASS_INFO[target['labels'][i]], c='r')
            
            title = str(target['image_id'].item()).zfill(4)
            
            plt.savefig(os.path.join(output_path, title + '.png'))
    
    time_elapsed = time.time() - since
    print(f"Inference Time per sample: {time_elapsed / (len(dataloader) * BATCH_SIZE):.6f}s")


def mean_average_precision(model, dataloader) -> Dict:
    """
    Return mAP result from the dataloader
    """
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    preds = []
    y_true = []
    for images, targets in tqdm(dataloader):
        images = [image.to(DEVICE) for image in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        for image, target in zip(images, targets):
            pred_class, pred_boxes, pred_score = get_prediction(model, image, 0)
            preds.append(
                dict(
                    boxes=torch.tensor(pred_boxes, dtype=torch.float32).to(DEVICE),
                    scores=torch.tensor(pred_score, dtype=torch.float32).to(DEVICE),
                    labels=torch.tensor(pred_class, dtype=torch.int64).to(DEVICE)
                )
            )
            y_true.append(
                dict(
                    boxes=target['boxes'],
                    labels=target['labels']
                )
            )

    metric = MeanAveragePrecision()
    metric.update(preds, y_true)
    metric_dict = dict(metric.compute())

    result = dict()
    for key, value in metric_dict.items():
        result[key] = value.cpu().numpy().tolist()

    return result
