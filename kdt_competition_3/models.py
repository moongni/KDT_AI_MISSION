import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import NUM_CLASSES


def create_model():
    # load FasterRCNN pretrained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    return model


if __name__ == "__main__":
    model = create_model()
    print(model)