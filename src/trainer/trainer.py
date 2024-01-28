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
        elif selected_model == "2plus1":
            self.model = Modified2plus1().to(self.__device)
        elif selected_model == "efficientnet_v2":
            self.model = ModifiedEfficientNetv2().to(self.__device)
        elif selected_model == "custom_twoplusone":
            self.model = TwoPlusOneModel3D(
                in_channels = 3,
                num_classes = 2
            ).to(self.__device)
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
        # Train loop steps
        self.__criterion = nn.BCEWithLogitsLoss()
        # Set the seed to create reproducible experiments
        torch.manual_seed(6543210)
        # Report model summary
        if show_summary:
            self.model.summary(input_size = (3, 40, 178, 150))
    
    def load_weights(self):
        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=1e-3
        )
        # Load model if exists
        try:
            checkpoint = torch.load(self.__weights_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            original_epochs = checkpoint['epoch']
            print(f"Loaded the model from {self.__weights_file}")
        except FileNotFoundError as e:
            original_epochs = 0
            print(f"Exception {e}")
        except Exception as e:
            original_epochs = 0
            print(f"Exception {e} form path file: {self.__weights_file}")

        return original_epochs, optimizer
    
    def get_model(self):
        return self.model

    def fit(
            self, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            num_epochs: int = 20,
            num_layers_to_unfreeze: int = 9
        ):
        # Move the model if possible
        if self.__device == torch.device("mps"):
            start_time = time.time()
            self.model.to_device(self.__device)
            end_time = time.time()
            print(f"Model and weights moved to GPU (MPS Mac), time spent: {end_time - start_time:.2f}s")
        # Load the weights
        original_epochs, optimizer = self.load_weights()
        # Unfreeze the model layers for training
        self.model.unfreeze(num_layers_to_unfreeze = num_layers_to_unfreeze)
        # Load the scheduler for the lr
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # Number of epochs
        num_epochs = num_epochs
        print(f"Continue training after {original_epochs} for epochs {num_epochs}")
        for epoch in range(original_epochs, num_epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
            # Set model to train:
            self.model.train()
            for batch_idx, (images, labels) in progress_bar:
                images, labels = images.to(self.__device), labels.reshape((labels.size(0),1)).to(self.__device)
                start = time.time()
                outputs = self.model(images)
                end = time.time()
                loss = self.__criterion(outputs.float(), labels.float())
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
                # Count number of matches
                total_matches = (predicted.float() == labels.float()).sum().item()
                # Memory
                memory_info = psutil.virtual_memory()
                progress_bar.set_postfix(
                    {
                        'loss': total_loss / (batch_idx + 1), 
                        'total_matches': int(total_matches),
                        'accuracy': average_accuracy,
                        'f1_score': self.__metrics.f1_score,
                        'memory_used': f"{memory_info.used / (1024**2):.2f}",
                        'labels balance': float(labels.float().mean(dim = 0)),
                        'prediction labels balance': float(predicted.float().mean(dim = 0))
                    }
                )
            # Validation
            valid_correct_predictions = 0
            valid_total_predictions = 0
            # Set model to eval:
            self.model.eval()
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
                print(f"Validation accuracy: {average_accuracy * 100:.2f}% F1-Score: {self.__validation_metrics.f1_score}")
                print(f"Validation number of positive predictions: {float(predicted.float().mean(dim = 0))} out of {valid_total_predictions}")
            # Store the current model just in case of failure
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                }, 
                f"tmp/{self.__preffix_model}_{self.__selected_model}_execution_{int(epoch) + 1}.pth"
            )
            # Update the scheduler
            scheduler.step()
        
        return self.model
