"""Model trainer and evaluation module."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                       confusion_matrix, f1_score, precision_score,
                       recall_score)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class ModelTrainer:
    """Trainer for Myanmar text classification models."""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = "cuda",
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with metrics
        """
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_x, batch_y in tqdm(train_loader, desc="Training"):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)
        
        return {"loss": avg_loss, "acc": acc}
    
    def evaluate(self, val_loader: DataLoader) -> dict:
        """Evaluate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc="Evaluating"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="weighted"),
            "recall": recall_score(all_labels, all_preds, average="weighted"),
            "f1": f1_score(all_labels, all_preds, average="weighted"),
        }
        
        return metrics, all_preds, all_labels
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 10,
              save_dir: Optional[str] = None) -> dict:
        """Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            save_dir: Directory to save model
            
        Returns:
            Training history
        """
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            
            # Validate
            val_metrics, _, _ = self.evaluate(val_loader)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['acc']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                if save_dir:
                    self.save_checkpoint(save_dir, "best_model.pt")
        
        return self.history
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Predict on new data.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices
        """
        self.model.eval()
        
        x = x.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x)
            preds = torch.argmax(outputs, dim=1)
        
        return preds.cpu().numpy()
    
    def save_checkpoint(self, save_dir: str, filename: str = "model.pt") -> None:
        """Save model checkpoint.
        
        Args:
            save_dir: Directory to save to
            filename: Filename
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, save_dir / filename)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)


def create_dataloader(X: np.ndarray,
                    y: Optional[np.ndarray] = None,
                    batch_size: int = 32,
                    shuffle: bool = True,
                    device: str = "cuda") -> DataLoader:
    """Create dataloader from numpy arrays.
    
    Args:
        X: Input features
        y: Labels (optional)
        batch_size: Batch size
        shuffle: Whether to shuffle
        device: Device
        
    Returns:
        DataLoader
    """
    if y is not None:
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(y)
        )
    else:
        dataset = TensorDataset(torch.FloatTensor(X))
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_classification_report(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              labels: Optional[list[str]] = None) -> str:
    """Get classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        
    Returns:
        Classification report string
    """
    return classification_report(y_true, y_pred, target_names=labels)


def get_confusion_matrix(y_true: np.ndarray,
                        y_pred: np.ndarray) -> np.ndarray:
    """Get confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pro


def save_metrics(metrics: dict, save_path: str, history: Optional[dict] = None) -> None:
    """Save metrics to JSON.
    
    Args:
        metrics: Metrics dictionary
        save_path: Path to save
        history: Training history (optional)
    """
    output = {"metrics": metrics}
    
    if history:
        output["history"] = history
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)