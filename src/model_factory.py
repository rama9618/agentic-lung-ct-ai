from __future__ import annotations

import torch
from torch import nn
from torchvision import models


SUPPORTED_MODELS = ("resnet50", "densenet121", "efficientnet_b0")


def create_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
) -> nn.Module:
    model_key = model_name.lower()

    if model_key == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_key == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if model_key == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    supported = ", ".join(SUPPORTED_MODELS)
    raise ValueError(f"Unsupported model_name: {model_name}. Choose one of: {supported}")


if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)

    for name in SUPPORTED_MODELS:
        model = create_model(model_name=name, num_classes=4, pretrained=False)
        output = model(dummy_input)
        print(f"{name}: output shape = {output.shape}")
