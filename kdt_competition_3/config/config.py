import os
import torch


# TRAIN SETTING
IMG_SIZE = 416
BATCH_SIZE = 2
EPOCHS = 100
CLASSES_INFO = {0:'Buffalo', 1:'Elephant', 2:'Rhinoceros', 3:'Zebra'}
CLASSES = CLASSES_INFO.keys()
NUM_CLASSES = len(CLASSES)
PATIENCE = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VISUALIZE_TRANSFORMED_IMAGES = True


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