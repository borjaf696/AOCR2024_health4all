import os
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
import time
import psutil
from torchsummary import summary
import timm

class ModifiedR3D18(nn.Module):
    def __init__(self):
        super(ModifiedR3D18, self).__init__()
        self.r3d_18 = models.video.r3d_18(
            weights=models.video.R3D_18_Weights.DEFAULT
        )
        num_ftrs = self.r3d_18.fc.in_features
        self.r3d_18.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1)         
        )

    def to_device(self, device):
        self.r3d_18 = self.r3d_18.to(device)

    def load_weights(self, path_file: str):
        self.load_state_dict(torch.load(path_file))

    def forward(self, x):
        return self.r3d_18(x)
    
    def summary(self, input_size):
        summary(self, input_size=input_size)

    def unfreeze(self):
        for param in self.r3d_18.parameters():
            param.requires_grad = True

class Modified2plus1(nn.Module):
    def __init__(self):
        super(Modified2plus1, self).__init__()
        self.r2plus1d_18 = models.video.r2plus1d_18(
            weights=models.video.R2Plus1D_18_Weights.DEFAULT
        )
        num_ftrs = self.r2plus1d_18.fc.in_features
        self.r2plus1d_18.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1)         
        )

    def to_device(self, device):
        self.r2plus1d_18 = self.r2plus1d_18.to(device)

    def load_weights(self, path_file: str):
        self.load_state_dict(torch.load(path_file))

    def forward(self, x):
        return self.r2plus1d_18(x)
    
    def summary(self, input_size):
        summary(self, input_size=input_size)

    def unfreeze(self):
        for param in self.r2plus1d_18.parameters():
            param.requires_grad = True

class ModifiedMC3_18(nn.Module):
    def __init__(self):
        super(ModifiedMC3_18, self).__init__()
        self.mc3_18 = models.video.mc3_18(
            weights=models.video.MC3_18_Weights.DEFAULT
        )
        num_ftrs = self.mc3_18.fc.in_features
        self.mc3_18.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1)         
        )

    def to_device(self, device):
        self.mc3_18 = self.mc3_18.to(device)

    def load_weights(self, path_file: str):
        self.load_state_dict(torch.load(path_file))

    def forward(self, x):
        return self.mc3_18(x)
    
    def summary(self, input_size):
        summary(self, input_size=input_size)

    def unfreeze(self):
        for param in self.mc3_18.parameters():
            param.requires_grad = True

## Too slow :(
class TwoPlusOneModel3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(TwoPlusOneModel3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3_depthwise = nn.Conv3d(128, 128, kernel_size=3, padding=1, groups=128)
        self.conv3_pointwise = nn.Conv3d(128, 128, kernel_size=1)
        self.relu3 = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3_depthwise(x)
        x = self.conv3_pointwise(x)
        x = self.relu3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def to_device(self, device):
        self.to(device)

    def load_weights(self, path_file: str):
        self.load_state_dict(torch.load(path_file))

    def summary(self, input_size):
        summary(self, input_size=input_size)

    def unfreeze(self):
        pass

class ModifiedEfficientNetv2(nn.Module):
    def __init__(self):
        super(ModifiedEfficientNetv2, self).__init__()
        # Model: efficientnetv2
        # Model variant: rw
        # Size: m
        # Training algo: agc_in1k (adaptive gradient clipping with 1000 classes)
        self.en2 = timm.create_model('efficientnetv2_rw_m.agc_in1k', pretrained = True)
        num_features = self.en2.get_classifier().in_features
        self.en2.classifier = nn.Linear(num_features, 1)

    def to_device(self, device):
        self.en2 = self.en2.to(device)

    def load_weights(self, path_file: str):
        self.en2.load_state_dict(torch.load(path_file))

    def forward(self, x):
        return self.en2(x)
    
    def summary(self, input_size):
        pass

    def unfreeze(self):
        pass