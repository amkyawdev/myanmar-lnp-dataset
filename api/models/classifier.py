"""Model class definitions."""

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


class MyanmarTextClassifier(nn.Module):
    """PyTorch classifier for Myanmar text.
    
    Simple feedforward neural network.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
        """Initialize classifier.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Class probabilities
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


class CNNTextClassifier(nn.Module):
    """CNN-based text classifier.
    
    For more complex text classification tasks.
    """
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_classes: int,
                 num_filters: int = 100,
                 filter_sizes: list[int] = None,
                 dropout: float = 0.5):
        """Initialize classifier.
        
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            num_classes: Number of output classes
            num_filters: Number of filters per size
            filter_sizes: List of filter kernel sizes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        filter_sizes = filter_sizes or [2, 3, 4, 5]
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len)
            
        Returns:
            Output logits
        """
        # Embedding: (batch, seq_len, embed_dim)
        x = self.embedding(x)
        
        # Transpose for conv1d: (batch, embed_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            c = torch.max_pool1d(c, c.size(2)).squeeze(2)
            conv_outputs.append(c)
        
        # Concatenate: (batch, num_filters * len(filter_sizes))
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class LSTMTextClassifier(nn.Module):
    """LSTM-based text classifier."""
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 num_layers: int = 1,
                 dropout: float = 0.5):
        """Initialize classifier.
        
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_classes: Number of output classes
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len)
            
        Returns:
            Output logits
        """
        # Embedding: (batch, seq_len, embed_dim)
        x = self.embedding(x)
        
        # LSTM: output (batch, seq_len, hidden_dim*2)
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Concatenate final forward and backward hidden states
        # hidden = (num_layers*2, batch, hidden_dim)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        x = self.dropout(hidden)
        x = self.fc(x)
        
        return x


# sklearn classifiers for quick prototyping
ClassifierTypes = {
    "logistic_regression": LogisticRegression,
    "multinomial_nb": MultinomialNB,
    "pytorch": MyanmarTextClassifier,
}