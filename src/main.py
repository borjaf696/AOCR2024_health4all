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

from typing import (
    Any
)
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

selected_model_volumes = "custom_twoplusone"
selected_model_slices = "efficientnet_v2"
batch_size_volumes = 8
batch_size_slices = 64
device_volumes = torch.device("cpu")
device_slices = torch.device("mps")

last_model_files = {
    "scan": "tmp/scan_r3d18_execution_3.pth",
    "slices": "tmp/slices_execution_5.pth"
}

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments to control the code flow")
    parser.add_argument('--preprocess', type = str2bool, default = True)
    parser.add_argument('--train_model', type = str2bool, default = True)
    parser.add_argument('--test_model', type = str2bool, default = True)

    return parser.parse_args()

# Load model
def load_model(
        device: Any = None,
        selected_model: str = None,
        type_of_model: str = "scan",
):
    # Volume trainer/model
    trainer = DefaultTrainer(
        device = device,
        selected_model = selected_model,
        preffix_model = type_of_model,
        weights_file = last_model_files[type_of_model]
    )
    trainer.load_weights()
    return trainer.get_model()

# TODO: Do better
def train_volumes(
        selected_model = selected_model_volumes, 
        type_of_model: str = "scan"
    ):
    train_dataset, val_dataset = DataLoaderCustom.get_train_val_datasets(
        image_dir="aocr2024/preprocessed_images/",
        labels_file = "aocr2024/TrainValid_ground_truth.csv",
        validation_file = "aocr2024/TrainValid_split.csv"
    )
    train_loader, val_loader = DataLoaderCustom.get_train_val_dataloaders(
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        batch_size = batch_size_volumes
    )
    # Train de model
    trainer = DefaultTrainer(
        device = device_volumes,
        selected_model = selected_model,
        preffix_model = type_of_model,
        weights_file = last_model_files[type_of_model]
    )
    model = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader
    )
    return model

def train_slices(selected_model = selected_model_slices):
    start_time = time.time()
    train_dataset, val_dataset = DataLoaderCustom.get_train_val_datasets(
        image_dir="aocr2024/preprocessed_images_2d/",
        labels_file = "aocr2024/TrainValid_ground_truth_slices.csv",
        validation_file = "aocr2024/TrainValid_split_slices.csv"
    )
    end_time = time.time()
    print(f"Time to get the datasets: {end_time - start_time:.2f}s")
    start_time = time.time()
    train_loader, val_loader = DataLoaderCustom.get_train_val_dataloaders(
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        batch_size = batch_size_slices,
        type_of_loader = "slices",
    )
    end_time = time.time()
    print(f"Time to create the loader: {end_time - start_time:.2f}s")
    # Train de model
    trainer = DefaultTrainer(
        device = device_slices,
        selected_model=selected_model,
        preffix_model = "slices",
        weights_file = "tmp/slices_execution_5.pth"
    )
    model = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader
    )
    return model

# Tester:
def test(
    model: Any = None,
    device: Any = None,
    image_dir: str = None,
    labels_file: str = None,
    validation_file: str = None,
    split: str = None, 
    batch_size: int = 4
):
    test_dataset = LazyImageDataset(
        image_dir = image_dir,
        labels_file = labels_file,
        validation_file = validation_file,
        split = split
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
    )
    classifications = []
    ids = []
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for _, (images, labels) in progress_bar:
        images = images.to(device)
        outputs = model(images)
        probabilities = torch.sigmoid(outputs)
        predicted = list((probabilities > 0.5).float())
        classifications += predicted
        ids += list(labels.split(".")[0])
    # Submission for scan level class
    predictions = pd.DataFrame(
        zip(ids, classifications),
        columns = [
            "id",
            "label"
        ]
    )
    predictions["label"] = predictions.label.astype(int) 
    return predictions

if __name__ == "__main__":
    args = parse_args()
    print(f"Args: {args}")
    if args.preprocess:
        print(f"Using the masks to filter the images")
        PreprocessUtilities.crop_images_with_calculated_bounds()
        print(f"Transforming 3D images into a set of 2D images, and adjusting the labels")
        # PreprocessUtilities.images_from_3d_to_2d()

    # Create the loaders
    if args.train_model:
        print(f"Training the volumes model: {selected_model_volumes}")
        model_volumes = train_volumes()
        print(f"Training the slices model: {selected_model_slices}")
        model_slices = train_slices()
        import sys
        print(f"Exiting")
        sys.exit()

    else:
        # Volume trainer/model
        model_volumes = load_model(
            device = device_volumes,
            selected_model = selected_model_volumes,
            type_of_model = "scan"
        )
        # Change to eval mode
        model_volumes.eval()
        # Slices trainer/model
        model_slices = load_model(
            device = device_slices,
            selected_model = selected_model_slices,
            type_of_model = "slices"
        )
        # Switch to eval mode
        model_slices.eval()

    if args.test_model:
        # Prediction volumes
        volume_predictions_df = test(
            model = model_volumes,
            device = device_volumes,
            image_dir = "aocr2024/preprocessed_images_test/",
            labels_file = "aocr2024/TrainValid_ground_truth.csv",
            validation_file = "aocr2024/TrainValid_split.csv",
            split = "Test"
        )
        # Predictions slices
        slices_predictions_df = test(
            model = model_slices,
            device = device_slices,
            image_dir = "aocr2024/preprocessed_images_test_2d/",
            labels_file = "aocr2024/TrainValid_ground_truth_slices.csv",
            validation_file = "aocr2024/TrainValid_split_slices.csv",
            split = "Test",
            batch_size = 64
        )
        # Concat dfs
        submission = pd.concat(
            [
                volume_predictions_df,
                slices_predictions_df
            ]
        )
        submission.to_csv("aocr2024/submission/submission.csv", index = False)

