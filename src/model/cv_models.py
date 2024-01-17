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
        weights = self.r3d_18.state_dict()
        for name, param in weights.items():
            if param.is_distributed:
                weights[name] = param.to(device)
        self.r3d_18.load_state_dict(weights)

    def forward(self, x):
        return self.r3d_18(x)
    
    def summary(self, input_size):
        print(f"Summary: {summary(self, input_size=input_size)}")

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
        weights = self.mc3_18.state_dict()
        for name, param in weights.items():
            if param.is_distributed:
                weights[name] = param.to(device)
        self.mc3_18.load_state_dict(weights)

    def forward(self, x):
        return self.mc3_18(x)
    
    def summary(self, input_size):
        print(f"Summary: {summary(self, input_size=input_size)}")