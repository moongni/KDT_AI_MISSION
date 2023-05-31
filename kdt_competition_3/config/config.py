import os
import torch
import numpy as np

# TRAIN SETTING
IMG_SIZE = 416
BATCH_SIZE = 2
EPOCHS = 100
CLASS_INFO = {0:'Buffalo', 1:'Elephant', 2:'Rhinoceros', 3:'Zebra'}
CLASSES = CLASS_INFO.keys()
NUM_CLASSES = len(CLASSES)
PATIENCE = 10
CONFIDENCE = 0.7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VISUALIZE_EVALUATED_IMAGE = True

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.299, 0.224, 0.225])

# PATH SETTING
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, 'data')

TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TRAIN_DF_PATH = os.path.join(TRAIN_PATH, 'train_output.csv')

TEST_PATH = os.path.join(DATA_PATH, 'test')
TEST_DF_PATH = os.path.join(TEST_PATH, 'test_output.csv')

OUTPUT_PATH = os.path.join(ROOT_DIR, 'result', 'images')
MODEL_SAVE_PATH = '/content/'


# Optional: GeneralizedRCNNTransform SETTING
RCNNTransform = {
    'min_size': 1024,
    'max_size': 1024,
    'image_mean': (0.485, 0.456, 0.406),
    'image_std': (0.229, 0.224, 0.225)
}