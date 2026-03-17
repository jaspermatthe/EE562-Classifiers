import torch.nn as nn
import timm # pip install timm

def get_vit(num_classes, pretrained=False):
    # 'mobilevitv2_050' is the 0.5 width variant (~1.4M params)
    # Default to no pretrained weights
    model = timm.create_model(
        'mobilevitv2_050', 
        pretrained=pretrained, 
        num_classes=num_classes
    )
    return model