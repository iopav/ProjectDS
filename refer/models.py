from torchvision import models
from torch import nn
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

def get_inceptionv3(num_classes=2, print_model=False):
    model = models.inception_v3(pretrained= True) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    if print_model:
        print(model)
    return model

def get_maxvit(num_classes=5, print_model=False):
    model = models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT)
    model.classifier[-1]=nn.Linear(512, num_classes,bias=False)
    if print_model:
        print(model)
    return model

def get_efficientNetB3(num_classes=5, print_model=False):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Linear(1536, num_classes, bias=True)
    )
    if print_model:
        print(model)
    return model

def get_efficientNetB0(num_classes=5, print_model=False):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Linear(1280, num_classes, bias=True)
    )
    if print_model:
        print(model)
    return model

def get_regnet_y_128gf(num_classes=2, print_model=False):
    model = models.regnet_y_128gf(weights=models.RegNet_Y_128GF_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(7392, 2048, bias=True),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(2048, num_classes, bias=True)
    )
    if print_model:
        print(model)
    return model

def get_regnet_y_32gf(num_classes=2, print_model=False, pretrained = True):
    if pretrained:
        model = models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1)
    else:
        model = models.regnet_y_32gf(weights= None)
    model.fc = nn.Sequential(
        nn.Linear(3712, 1024, bias=True),
        nn.ReLU(),
        nn.Dropout(0.35),
        nn.Linear(1024, num_classes, bias=True)
    )
    if print_model:
        print(model)
    return model

def get_regnet_y_16gf(num_classes=2, print_model=False):
    model = models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(3024, 1024, bias=True),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, num_classes, bias=True)
    )
    if print_model:
        print(model)
    return model

def get_convnext_tiny(num_classes=2, print_model=False, drop_path_rate = 0.5):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT, drop_path_rate = drop_path_rate)
    model.classifier = nn.Sequential(
        LayerNorm2d((768),eps=1e-06, elementwise_affine=True),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(in_features=768, out_features=num_classes, bias=True)
    )   
    if print_model:
        print(model)
    return model 


def get_convnext_tiny_AvgPolling(print_model=False, drop_path_rate = 0.5):
    def get_convnext_tiny_AvgPolling(print_model=False, drop_path_rate=0.5):
        """
        加载预训练的ConvNeXt Tiny模型，并修改其分类器。
        参数:
        print_model (bool): 是否打印模型结构。默认值为False。
        drop_path_rate (float): Drop path rate的值。默认值为0.5。
        返回:
        torch.nn.Module: 修改后的ConvNeXt Tiny模型。
        功能:
        1. 加载预训练的ConvNeXt Tiny模型，并设置drop path rate。
        2. 修改模型的分类器，添加LayerNorm2d层、卷积层、自适应平均池化层和展平层。
        3. 如果print_model为True，打印模型结构。
        4. 返回修改后的模型。
        """
    # Load the pre-trained ConvNeXt Tiny model with specified drop path rate
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT, drop_path_rate = drop_path_rate)
    
    # Modify the classifier of the model
    model.classifier = nn.Sequential(
        # Apply LayerNorm2d with specified parameters
        LayerNorm2d((768), eps=1e-06, elementwise_affine=True),
        
        # Add a convolutional layer with 2 output channels and 1x1 kernel
        nn.Conv2d(in_channels=768, out_channels=2, kernel_size=(1, 1), stride=(1, 1), bias=False),
        
        # Apply adaptive average pooling to get output size of 1x1
        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        
        # Flatten the output tensor starting from the first dimension
        nn.Flatten(start_dim=1, end_dim=-1)
    )
    
    # Print the model if print_model is True
    if print_model:
        print(model)
    
    # Return the modified model
    return model