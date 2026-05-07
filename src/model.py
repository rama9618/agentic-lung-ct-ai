from __future__ import annotations

import torch
from torch import nn
from torchvision import models


CLASS_NAMES = (
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma",
)


def create_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


if __name__ == "__main__":
    model = create_model(num_classes=len(CLASS_NAMES), pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
