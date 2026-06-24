from __future__ import annotations

import torch
from torchvision.models import ResNet18_Weights, resnet18


def create_resnet18_classifier(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features, num_classes),
    )

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model
