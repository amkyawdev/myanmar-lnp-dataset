"""Evaluation metrics for classification."""

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                       recall_score)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)


def compute_precision(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     average: str = "weighted") -> float:
    """Compute precision score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method
        
    Returns:
        Precision score
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def compute_recall(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  average: str = "weighted") -> float:
    """Compute recall score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method
        
    Returns:
        Recall score
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_f1(y_true: np.ndarray,
              y_pred: np.ndarray,
              average: str = "weighted") -> float:
    """Compute F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method
        
    Returns:
        F1 score
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_per_class_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              labels: list[str]) -> dict:
    """Compute per-class metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        
    Returns:
        Dictionary with per-class metrics
    """
    metrics = {}
    
    for label in labels:
        mask = y_true == label
        
        if mask.sum() == 0:
            continue
        
        label_true = y_true[mask]
        label_pred = y_pred[mask]
        
        metrics[label] = {
            "precision": precision_score(label_true, label_pred, zero_division=0),
            "recall": recall_score(label_true, label_pred, zero_division=0),
            "f1": f1_score(label_true, label_pred, zero_division=0),
            "support": mask.sum(),
        }
    
    return metrics


def compute_all_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       average: str = "weighted") -> dict:
    """Compute all classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method
        
    Returns:
        Dictionary with all metrics
    """
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "precision": compute_precision(y_true, y_pred, average),
        "recall": compute_recall(y_true, y_pred, average),
        "f1": compute_f1(y_true, y_pred, average),
    }


class MetricsTracker:
    """Track metrics during training and evaluation."""
    
    def __init__(self):
        """Initialize tracker."""
        self.metrics_history = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }
    
    def update(self, metrics: dict) -> None:
        """Update with new metrics.
        
        Args:
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def get_best(self, metric: str = "f1") -> tuple[float, int]:
        """Get best metric value and epoch.
        
        Args:
            metric: Metric name
            
        Returns:
            Tuple of (best_value, epoch)
        """
        if metric not in self.metrics_history:
            raise ValueError(f"Unknown metric: {metric}")
        
        values = self.metrics_history[metric]
        
        if not values:
            return 0.0, -1
        
        best_idx = np.argmax(values)
        return values[best_idx], best_idx
    
    def get_summary(self) -> dict:
        """Get summary of all metrics.
        
        Returns:
            Dictionary with summary
        """
        summary = {}
        
        for key, values in self.metrics_history.items():
            if values:
                summary[key] = {
                    "best": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                }
        
        return summary