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

from torch.utils.data import Sampler
import numpy as np
import torch
import time

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, n_classes, n_samples, strategy: str = "over"):
        self.labels = [dataset[i][1] for i in range(len(dataset))]
        self.labels_set = list(set(self.labels))
        self.used_labels_indices_count = {label: 0 for label in self.labels_set}
        self.count = len(dataset)
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_samples * self.n_classes
        # Labels to index
        self.label_to_indices = {
            label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set
        }
        # Sampling strategy to fight imbalance
        self.__strategy = strategy

    def __iter__(self):
        self.used_labels_indices_count = {label: 0 for label in self.labels_set}

        if self.__strategy == "over":
            total_batches = len(self.labels) // self.batch_size
            for _ in range(total_batches):
                indices = self._sample_indices_oversampling()
                np.random.shuffle(indices)
                yield indices
        elif self.__strategy == "under":
            min_samples = min(len(self.label_to_indices[label]) for label in self.labels_set)
            total_batches = min_samples * len(self.labels_set) // self.batch_size
            for _ in range(total_batches):
                indices = self._sample_indices_undersampling(min_samples)
                np.random.shuffle(indices)
                yield indices
        else:
            raise ValueError

    def __len__(self):
        return self.count // self.batch_size
    
    # Sampling strategies
    def _sample_indices_oversampling(self):
        indices = []
        for _ in range(self.n_samples):
            for label in self.labels_set:
                if self.used_labels_indices_count[label] >= len(self.label_to_indices[label]):
                    np.random.shuffle(self.label_to_indices[label])
                    self.used_labels_indices_count[label] = 0

                selected_indices = self.label_to_indices[label][self.used_labels_indices_count[label]:self.used_labels_indices_count[label] + 1].tolist()
                indices.extend(selected_indices)
                self.used_labels_indices_count[label] += 1
        return indices

    def _sample_indices_undersampling(self, min_samples):
        indices = []
        for label in self.labels_set:
            selected_indices = np.random.choice(self.label_to_indices[label], min_samples, replace=False).tolist()
            indices.extend(selected_indices)
        return indices

class LazyImageDataset(Dataset):
    def __init__(
            self, 
            image_dir: str, 
            labels_file: str, 
            validation_file: str,
            split: str = "Train",
            debug: bool = False
        ):
        self.image_dir = image_dir
        self.__type_of_data = split
        self.image_filenames = set()
        if debug:
            for i, filename in enumerate(os.listdir(image_dir)):
                if i > 1000:
                    break
                self.image_filenames.add(filename.split(".")[0])
        else:
            for i, filename in enumerate(os.listdir(image_dir)):
                self.image_filenames.add(filename.split(".")[0])
        if self.__type_of_data != "Test":
            self.__df_validation = pd.read_csv(
                validation_file,
            )
            self.df_labels = pd.read_csv(
                labels_file
            )
            self.image_filenames = self.image_filenames.intersection(
                set(
                    self.__df_validation.loc[
                        self.__df_validation.group == self.__type_of_data,
                        "id"
                    ]
                )
            )
        self.image_filenames = [f"{filename}.nii.gz" for filename in self.image_filenames]  
        # Store the labels per image
        if split != "Test":
            self.labels = {
                row["id"]: int(row["label"])
                for _, row in self.df_labels.iterrows()
                if f"{row['id']}.nii.gz" in self.image_filenames
            }
        else:
            self.labels = {
                id.split(".")[0]: id for id in self.image_filenames
            }  

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image_nifti = nib.load(img_name)
        image = torch.from_numpy(
            image_nifti.get_fdata()
        ).float()
        # Get label
        id = self.image_filenames[idx].split(".")[0]
        label = self.labels[id]

        return image, label
    
    def get_labels(self):
        return [
            self.labels[filename.split(".")[0]] for filename in self.image_filenames
        ]
    
class DataLoaderCustom:
    
    @staticmethod
    def get_train_val_datasets(
        image_dir = "../aocr2024/preprocessed_images/",
        labels_file = "../aocr2024/TrainValid_ground_truth.csv",
        validation_file = "../aocr2024/TrainValid_split.csv"
    ):
        # Load the dataset
        train_dataset = LazyImageDataset(
            image_dir = image_dir,
            labels_file = labels_file,
            validation_file = validation_file,
            split = "Train"
        )
        val_dataset = LazyImageDataset(
            image_dir = image_dir,
            labels_file = labels_file,
            validation_file = validation_file,
            split = "Valid"
        )
        return train_dataset, val_dataset
    
    @staticmethod
    def get_train_val_dataloaders(
        train_dataset: Dataset = None, 
        val_dataset: Dataset = None, 
        batch_size: int = 32, 
        type_of_loader: str = "scan",
        n_classes: int = 2
    ):
        if type_of_loader == "scan":
            train_loader = DataLoader( 
                train_dataset, 
                batch_size = batch_size, 
                shuffle = False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size = batch_size,
                shuffle = False
            )
        elif type_of_loader == "slices":
            start_time = time.time()
            balanced_batch_sampler_train = BalancedBatchSampler(
                dataset = train_dataset, 
                n_classes = n_classes, 
                n_samples = batch_size // n_classes
            )
            end_time = time.time()
            print(f"Balanced batch sampler (train): {end_time - start_time:.2f}s")
            start_time = time.time()
            balanced_batch_sampler_val = BalancedBatchSampler(
                dataset = val_dataset, 
                n_classes = n_classes, 
                n_samples = batch_size // n_classes
            )
            end_time = time.time()
            print(f"Balanced batch sampler (val): {end_time - start_time:.2f}s")
            start_time = time.time()
            train_loader = DataLoader(
                train_dataset, 
                batch_sampler = balanced_batch_sampler_train
            )
            end_time = time.time()
            print(f"Dataloader time (train): {end_time - start_time:.2f}s")
            start_time = time.time()
            val_loader = DataLoader(
                val_dataset,
                batch_sampler = balanced_batch_sampler_val
            )
            end_time = time.time()
            print(f"Dataloader time (val): {end_time - start_time:.2f}s")
        return train_loader, val_loader