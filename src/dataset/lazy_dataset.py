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


class LazyImageDataset(Dataset):
    def __init__(self, image_dir, labels_file, validation_file, split = "Train"):
        self.image_dir = image_dir
        self.__type_of_data = split
        self.image_filenames = set([filename.split(".")[0] for filename in os.listdir(image_dir)])
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

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image_nifti = nib.load(img_name)
        image = torch.from_numpy(
            image_nifti.get_fdata()
        ).float()
        # Get label
        id = img_name.split("/")[-1].split(".")[0]
        if self.__type_of_data == "Test":
            label = id
        else:
            label = int(
                    self.df_labels.loc[
                        self.df_labels.id == id,
                        "label"
                    ].iloc[
                        0
                    ]
                )
        return image, label
    
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
            split = "Train",
        )
        val_dataset = LazyImageDataset(
            image_dir = image_dir,
            labels_file = labels_file,
            validation_file = validation_file,
            split = "Valid",
        )
        return train_dataset, val_dataset
    
    @staticmethod
    def get_train_val_dataloaders(train_dataset, val_dataset, batch_size):
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
        return train_loader, val_loader