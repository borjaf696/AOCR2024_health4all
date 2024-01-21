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

from utils.utils import (
    MetricUtils,
    ClassificationMetrics
)
from model.cv_models import *

class DefaultTrainer:
    def __init__(
            self, 
            device: str = "cpu", 
            selected_model: str = "mc3_18", 
            show_summary: bool = False,
            weights_file: str = None,
            preffix_model: str = "mc3_18"
        ):
        # TODO: add a factory here
        self.__device = device
        # Model
        if selected_model == "r3d18":
            self.model = ModifiedR3D18().to(self.__device)
            print(f"Created model: {selected_model}")
        elif selected_model == "mc3_18":
            self.model = ModifiedMC3_18().to(self.__device)
        elif selected_model == "efficientnet_v2":
            self.model = ModifiedEfficientNetv2().to(self.__device)
        else:
            raise RuntimeError
        # Weights file:
        self.__weights_file = weights_file
        self.__preffix_model = preffix_model
        # Metrics:
        self.__metrics = ClassificationMetrics(0, 0, 0, 0)
        self.__validation_metrics = ClassificationMetrics(0, 0, 0, 0)
        # Selected model
        self.__selected_model = selected_model
        # Report model summary
        if show_summary:
            self.model.summary(input_size = (3, 40, 178, 150))
    
    def load_weights(self):
        # Load model if exists
        try:
            self.model.load_weights(self.__weights_file)
            original_epochs = self.__weights_file.split(".")[0].split("_")[-1]
            print(f"Loaded the model from {self.__weights_file}")
        except FileNotFoundError as e:
            original_epochs = 0
            print(f"Exception {e}")
        except Exception as e:
            original_epochs = 0
            print(f"Exception {e} form path file: {self.__weights_file}")

        return original_epochs
    
    def get_model(self):
        return self.model

    def fit(
            self, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            num_epochs: int = 20
        ):
        # Train loop steps
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        # Move the model if possible
        if self.__device == torch.device("mps"):
            start_time = time.time()
            self.model.to_device(self.__device)
            end_time = time.time()
            print(f"Model and weights moved to GPU (MPS Mac), time spent: {end_time - start_time:.2f}s")
        self.model.train()
        # Load the weights
        original_epochs = self.load_weights()
        # Number of epochs
        num_epochs = num_epochs
        print(f"Continue training after {original_epochs} for epochs {num_epochs}")
        for epoch in range(num_epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, (images, labels) in progress_bar:
                images, labels = images.to(self.__device), labels.reshape((labels.size(0),1)).to(self.__device)
                start = time.time()
                outputs = self.model(images)
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
                # Add extra metrics
                self.__metrics += MetricUtils.calculate_metrics(predicted, labels)
                # Memory
                memory_info = psutil.virtual_memory()
                progress_bar.set_postfix(
                    {
                        'loss': total_loss / (batch_idx + 1), 
                        'accuracy': average_accuracy,
                        'f1_score': self.__metrics.f1_score,
                        'memory_used': f"{memory_info.used / (1024**2):.2f}",
                        'prediction_time': f"{end - start:.2f}s",
                        'epoch labels balance': labels.float().mean(dim = 0)
                    }
                )
            # Validation
            valid_correct_predictions = 0
            valid_total_predictions = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.__device), labels.reshape((labels.size(0),1)).to(self.__device)
                    outputs = self.model(images)
                    probabilities = torch.sigmoid(outputs)
                    predicted = (probabilities > 0.5).float()
                    valid_correct_predictions += (predicted == labels).sum().item()
                    valid_total_predictions += labels.size(0)
                    # Calculate validation metrics
                    self.__validation_metrics += MetricUtils.calculate_metrics(predicted, labels)
                average_accuracy = valid_correct_predictions / (valid_total_predictions + 1e-5)
                print(f"Validation accuracy: {average_accuracy * 100:.2f}% F1-Score: {self.__validation_metrics.f1_score} Validation items: {valid_total_predictions}")
            # Store the current model just in case of failure
            torch.save(self.model.state_dict(), f'tmp/{self.__preffix_model}_{self.__selected_model}_execution_{int(epoch) + int(original_epochs) + 1}.pth')
        
        return self.model
