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
    model = fasterrcnn_resnet_50('resnet18')
    print(model)