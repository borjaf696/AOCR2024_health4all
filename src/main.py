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

from dataset.lazy_dataset import (
    LazyImageDataset,
    DataLoaderCustom
)
from trainer.trainer import (
    DefaultTrainer
)

filter_masks = False

selected_model = "r3d18"
batch_size = 8
device = torch.device("cpu")

if __name__ == "__main__":
    print(f"Filtering masks: {filter_masks}")
    if filter_masks:
        print(f"Add filtering masks step")
    print(f"Running the model: {selected_model}")
    # Create the loaders
    train_dataset, val_dataset = DataLoaderCustom.get_train_val_datasets(
        image_dir="aocr2024/preprocessed_images/",
        labels_file = "aocr2024/TrainValid_ground_truth.csv",
        validation_file = "aocr2024/TrainValid_split.csv"
    )
    train_loader, val_loader = DataLoaderCustom.get_train_val_dataloaders(
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        batch_size = batch_size
    )
    # Train de model
    trainer = DefaultTrainer(device = device)
    trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        selected_model = selected_model
    )

