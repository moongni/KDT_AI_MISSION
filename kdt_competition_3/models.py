import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

from config import NUM_CLASSES


def fasterrcnn_resnet_50():
    """
    load FasterRCNN pretrained model
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    return model


if __name__ == "__main__":
    model = fasterrcnn_resnet_50()
    print(model)

    import pandas as pd
    from torch.utils.data import DataLoader
    from config import *
    from utils import *

    train_df = pd.read_csv(os.path.join(TRAIN_PATH, 'train_output.csv'))
    test_df = pd.read_csv(os.path.join(TEST_PATH, 'test_output.csv'))

    train_dset = ObjDetectionDataset(TRAIN_PATH, train_df, TrainTransform())
    test_dset = ObjDetectionDataset(TEST_PATH, test_df, TestTransform())
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    visualize_object_detection(model, test_loader)