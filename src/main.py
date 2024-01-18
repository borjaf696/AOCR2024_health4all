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
import argparse

from dataset.lazy_dataset import (
    LazyImageDataset,
    DataLoaderCustom
)
from trainer.trainer import (
    DefaultTrainer
)
from src.utils.utils import (
    PreprocessUtilities
)

filter_masks = False

selected_model_volumes = "r3d18"
selected_model_slices = "efficientnet_v2"
batch_size = 8
device_volumes = torch.device("cpu")
device_slices = torch.device("mps")

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments to control the code flow")
    parser.add_argument('--preprocess', action='store_true', default = True)
    parser.add_argument('--train_model', action='store_true', default = True)
    parser.add_argument('--test_model', action='store_true', default = True)

    return parser.parse_args()

# TODO: Do better
def train_volumes(selected_model = selected_model_volumes):
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
    trainer = DefaultTrainer(device = device_volumes)
    model = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        selected_model = selected_model
    )
    return model

def train_slices(selected_model = selected_model_slices):
    train_dataset, val_dataset = DataLoaderCustom.get_train_val_datasets(
        image_dir="aocr2024/preprocessed_images_2d/",
        labels_file = "aocr2024/TrainValid_ground_truth_slices.csv",
        validation_file = "aocr2024/TrainValid_split_slices.csv"
    )
    train_loader, val_loader = DataLoaderCustom.get_train_val_dataloaders(
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        batch_size = batch_size
    )
    # Train de model
    trainer = DefaultTrainer(device = device_volumes)
    model = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        selected_model = selected_model
    )
    return model

if __name__ == "__main__":
    args = parse_args()
    print(f"Args: {args}")
    if args.preprocess:
        print(f"Using the masks to filter the images")
        # PreprocessUtilities.crop_images_with_calculated_bounds()
        print(f"Transforming 3D images into a set of 2D images, and adjusting the labels")
        PreprocessUtilities.images_from_3d_to_2d()
        import sys
        sys.exit()

    print(f"Running the model: {selected_model}")
    # Create the loaders
    if args.train_model:
        model_volumes = train_volumes()
        model_slices = train_slices()

    if args.test_model:
        # Test dataset
        test_dataset = LazyImageDataset(
            image_dir = "../aocr2024/preprocessed_images_test/",
            labels_file = "../aocr2024/TrainValid_ground_truth.csv",
            validation_file = "../aocr2024/TrainValid_split.csv",
            split = "Test",
        )
        # Test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size = batch_size,
            shuffle = False,
        )
        classifications = []
        ids = []
        for images, labels in test_loader:
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predicted = list((probabilities > 0.5).float())
            classifications += predicted
            ids += list(labels)
        # Submission for scan level class
        submission = pd.DataFrame(
            zip(ids, classifications),
            columns = [
                "id",
                "label"
            ]
        )
        submission["label"] = submission.label.astype(int)
        submission.to_csv("../aocr2024/submission.csv", index = False)

