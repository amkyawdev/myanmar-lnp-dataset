"""Feature extraction module for Myanmar NLP.

Provides TF-IDF, Word2Vec, and BERT-based vectorization.
"""

from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class MyanmarVectorizer:
    """Vectorizer for Myanmar text.
    
    Supports multiple vectorization methods.
    """
    
    def __init__(self,
                method: str = "tfidf",
                max_features: int = 10000,
                ngram_range: tuple[int, int] = (1, 2),
                min_df: int = 2,
                max_df: float = 0.95):
        """Initialize vectorizer.
        
        Args:
            method: Vectorization method (tfidf, word2vec, bert)
            max_features: Maximum number of features
            ngram_range: N-gram range
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.is_fitted = False
    
    def fit(self, texts: list[str]) -> "MyanmarVectorizer":
        """Fit vectorizer on texts.
        
        Args:
            texts: List of training texts
            
        Returns:
            Self
        """
        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                analyzer="char_wb" if self._is_myanmar(texts[0]) else "word"
            )
            self.vectorizer.fit(texts)
        
        self.is_fitted = True
        return self
    
    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts to vectors.
        
        Args:
            texts: List of texts
            
        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        if self.method == "tfidf":
            return self.vectorizer.transform(texts).toarray()
        elif self.method == "word2vec":
            # Placeholder for Word2Vec
            raise NotImplementedError("Word2Vec not implemented. Use pre-trained models.")
        elif self.method == "bert":
            # Placeholder for BERT
            raise NotImplementedError("BERT not implemented. Use transformers library.")
        
        raise ValueError(f"Unknown method: {self.method}")
    
    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit and transform texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Feature matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    def _is_myanmar(self, text: str) -> bool:
        """Check if text contains Myanmar characters.
        
        Args:
            text: Input text
            
        Returns:
            True if text contains Myanmar characters
        """
        for char in text:
            if "\u1000" <= char <= "\u109F":
                return True
        return False
    
    def get_feature_names(self) -> list[str]:
        """Get feature names.
        
        Returns:
            List of feature names
        """
        if self.vectorizer:
            return self.vectorizer.get_feature_names_out().tolist()
        return []


class BERTVectorizer:
    """BERT-based vectorizer for Myanmar text.
    
    Requires transformers library with pre-trained Myanmar BERT.
    """
    
    def __init__(self,
                model_name: str = "bert-base-multilingual-cased",
                max_length: int = 512,
                device: str = "cpu"):
        """Initialize BERT vectorizer.
        
        Args:
            model_name: Name of pre-trained model
            max_length: Maximum sequence length
            device: Device to use (cpu, cuda)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.tokenizer = None
        self.model = None
    
    def _lazy_load(self):
        """Lazy load model and tokenizer."""
        if self.model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
            except ImportError:
                raise ImportError("transformers library required. Install with: pip install transformers")
    
    def fit(self, texts: list[str]) -> "BERTVectorizer":
        """Fit vectorizer (no-op for BERT).
        
        Args:
            texts: List of texts (not used)
            
        Returns:
            Self
        """
        self._lazy_load()
        return self
    
    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts to BERT embeddings.
        
        Args:
            texts: List of texts
            
        Returns:
            Feature matrix
        """
        import torch
        
        self._lazy_load()
        
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit and transform texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Feature matrix
        """
        self.fit(texts)
        return self.transform(texts)


def create_vectorizer(method: str = "tfidf", **kwargs) -> MyanmarVectorizer:
    """Create vectorizer by method name.
    
    Args:
        method: Vectorization method
        kwargs: Additional arguments for vectorizer
        
    Returns:
        Vectorizer instance
    """
    if method == "bert":
        return BERTVectorizer(**kwargs)
    return MyanmarVectorizer(method=method, **kwargs)