from typing import Dict

import torch.nn as nn
import torchvision
import torch.nn.functional as F

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PairWiseRanker(nn.Module):
    def __init__(self, pretrained):
        """
        ResNet-50 model.

        Args:
            num_classes: Number of classes to predict
        """
        super().__init__()

        if pretrained == 'yes':
            pretrained = "IMAGENET1K_V1"
        elif pretrained == 'no':
            pretrained = None
        else:
            raise ValueError("pretrained should be 'yes' or 'no' ")

        self.resnet50 = torchvision.models.resnet50(pretrained=pretrained)
        logger.info(f"Loading ResNet50 model... with pretrained weight {pretrained}")
        for param in self.resnet50.parameters():
            param.requires_grad = True

        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, 128)
        self.activation = nn.Sigmoid()
        self.final_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_anchor, x_anchor_pair, inference=False):
        
        enc_x_anchor = self.resnet50(x_anchor)
        enc_x_anchor_pair = self.resnet50(x_anchor_pair)

        diff = enc_x_anchor - enc_x_anchor_pair
        x = self.final_layer(diff)
        x = self.activation(x)
        return x


def get_model(model_config: Dict):
    if model_config['model_name'].lower() == 'pairwiseranker':
        model = PairWiseRanker(pretrained=model_config['pretrained'])
    else:
        raise ValueError(f"Model {model_config['model_name']} not supported, use 'pairwiseranker'")
    return model
