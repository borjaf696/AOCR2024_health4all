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

from model.cv_models import *

class DefaultTrainer:
    def __init__(self, device: str = "cpu"):
        # TODO: add a factory here
        self.__device = device

    def fit(self, train_loader, val_loader, selected_model: str = "mc3_18"):
        # Model
        if selected_model == "r3d18":
            model = ModifiedR3D18().to(self.__device)
        elif selected_model == "mc3_18":
            model = ModifiedMC3_18().to(self.__device)
        else:
            raise RuntimeError
        # Report model summary
        model.summary(input_size = (3, 40, 178, 150))
        # Train loop steps
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        # Move the model if possible
        if self.__device == torch.device("mps"):
            model.to_device(self.__device)
            print(f"Model and weights moved to GPU (MPS Mac)")
        model.train()
        # Load model if exists
        try:
            file_name = "tmp/tmp_execution_17.pth"
            model.load_state_dict(torch.load(file_name))
            original_epochs = file_name.split(".")[0].split("_")[-1]
            print(f"Loaded the model from {file_name}")
        except Exception as e:
            original_epochs = 0
            print(f"Exception {e}")

        # Number of epochs
        num_epochs = 30
        print(f"Continue training after {original_epochs} for epochs {num_epochs}")
        for epoch in range(num_epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, (images, labels) in progress_bar:
                images, labels = images.to(self.__device), labels.reshape((labels.size(0),1)).to(self.__device)
                start = time.time()
                outputs = model(images)
                end = time.time()
                loss = criterion(outputs.float(), labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # Accuracy
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
                average_accuracy = correct_predictions / total_predictions
                # Memory
                memory_info = psutil.virtual_memory()
                progress_bar.set_postfix(
                    {
                        'loss': total_loss / (batch_idx + 1), 
                        'accuracy': average_accuracy,
                        'memory_used': f"{memory_info.used / (1024**2):.2f}",
                        'memory_availabls': f"{memory_info.available / (1024**2):.2f}MB",
                        'prediction_time': f"{end - start:.2f}s"
                    }
                )
            # Validation
            valid_correct_predictions = 0
            valid_total_predictions = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    labels = labels.reshape((labels.size(0),1))
                    outputs = model(images)
                    probabilities = torch.sigmoid(outputs)
                    predicted = (probabilities > 0.5).float()
                    valid_correct_predictions += (predicted == labels).sum().item()
                    valid_total_predictions += labels.size(0)
                average_accuracy = valid_correct_predictions / valid_total_predictions
                print(f"Validation accuracy: {average_accuracy * 100:.2f}% Validation items: {valid_total_predictions}")
            # Store the current model just in case of failure
            torch.save(model.state_dict(), f'tmp/tmp_execution_{int(epoch) + int(original_epochs) + 1}.pth')
        
        return model
