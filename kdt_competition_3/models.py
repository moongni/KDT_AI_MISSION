import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.rpn import AnchorGenerator

from config import NUM_CLASSES


def create_diff_backbone(backbone_name: str) -> nn.Module:
    if backbone_name == 'resnet18':
        backbone = backbone_resnet18()
    elif backbone_name == 'mobilenet':
        backbone = backbone_mobilnet()

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to be [0]. 
    # More generally, the backbone should return an OrderedDict[Tensor], 
    # and in featmap_names you can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                       num_classes=NUM_CLASSES,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    
    return model


def backbone_resnet18():
    resnet_net = torchvision.models.resnet18(pretrained=True)
    modules = list(resnet_net.children())[:-2]
    backbone = nn.Sequential(*modules)
    backbone.out_channels = 512
    
    return backbone

def backbone_resnet50():
    # load FasterRCNN pretrained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    return model


def backbone_mobilnet():
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    return backbone


if __name__ == "__main__":
    model = create_diff_backbone('resnet18')
    print(model)