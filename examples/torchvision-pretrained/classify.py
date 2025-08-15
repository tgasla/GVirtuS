import torch
from torchvision import models
from PIL import Image

from torchvision import models

# Change this to the desired model name
model_selector = "vit_h_14"

# all these models were tested and are supported by GVirtuS
# Mapping of model names to (constructor, weights enum)
model_registry = {
    "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
    "inception_v3": (models.inception_v3, models.Inception_V3_Weights.DEFAULT),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
    "efficientnet_b7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
    "squeezenet1_0": (models.squeezenet1_0, models.SqueezeNet1_0_Weights.DEFAULT),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
    "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT),
    "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights.DEFAULT),
    "swin_v2_b": (models.swin_v2_b, models.Swin_V2_B_Weights.DEFAULT), # 3rd best model
    "resnext101_64x4d": (models.resnext101_64x4d, models.ResNeXt101_64X4D_Weights.DEFAULT), # # 2nd best model
    "vit_h_14": (models.vit_h_14, models.ViT_H_14_Weights.DEFAULT), # 1st best model
}

# Load model
if model_selector not in model_registry:
    raise ValueError(f"Model '{model_selector}' is not supported.")

model_fn, weights = model_registry[model_selector]

# For inception_v3, disable aux_logits
if model_selector == "inception_v3":
    model = model_fn(weights=weights, aux_logits=False)
else:
    model = model_fn(weights=weights)

model.eval()

# Use the preprocessing transforms provided by the weights
preprocess = weights.transforms()

# Load and preprocess your image
img = Image.open('image.png').convert('RGB')
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]

# Inference
with torch.no_grad():
    output = model(input_batch)

# Get predicted class index and label
_, predicted_idx = torch.max(output, 1)
predicted_label = weights.meta['categories'][predicted_idx]

print(f'Predicted class: {predicted_label}')
