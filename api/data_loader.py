"""Data loader module for Myanmar LNP dataset.

Supports loading data from multiple formats: JSONL, CSV, and Parquet.
"""

import json
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd


class LNPDataLoader:
    """Data loader for Myanmar LNP dataset.
    
    Supports multiple file formats and provides consistent interface
    for loading labeled Myanmar text data.
    """
    
    SUPPORTED_FORMATS = {".jsonl", ".json", ".csv", ".parquet"}
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize data loader.
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir) if data_dir else None
    
    def load_file(self, filepath: str) -> pd.DataFrame:
        """Load data from file.
        
        Args:
            filepath: Path to data file
            
        Returns:
            DataFrame with columns: text, label, metadata
        """
        path = Path(filepath)
        suffix = path.suffix.lower()
        
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}")
        
        if suffix in {".jsonl", ".json"}:
            return self._load_json(path)
        elif suffix == ".csv":
            return self._load_csv(path)
        elif suffix == ".parquet":
            return self._load_parquet(path)
    
    def _load_json(self, path: Path) -> pd.DataFrame:
        """Load JSON/JSONL file."""
        if path.suffix == ".jsonl":
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
        
        return pd.DataFrame(records)
    
    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Load CSV file."""
        return pd.read_csv(path, encoding="utf-8")
    
    def _load_parquet(self, path: Path) -> pd.DataFrame:
        """Load Parquet file."""
        return pd.read_parquet(path)
    
    def iter_file(self, filepath: str) -> Iterator[dict[str, Any]]:
        """Iterate over JSONL file line by line (memory efficient).
        
        Args:
            filepath: Path to JSONL file
            
        Yields:
            Dictionary for each record
        """
        path = Path(filepath)
        
        if path.suffix != ".jsonl":
            raise ValueError("iter_file only supports JSONL format")
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    
    def save_jsonl(self, data: list[dict], filepath: str) -> None:
        """Save data to JSONL file.
        
        Args:
            data: List of dictionaries
            filepath: Output file path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def save_csv(self, data: pd.DataFrame, filepath: str) -> None:
        """Save data to CSV file.
        
        Args:
            data: DataFrame to save
            filepath: Output file path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(path, index=False, encoding="utf-8")
    
    def save_parquet(self, data: pd.DataFrame, filepath: str) -> None:
        """Save data to Parquet file.
        
        Args:
            data: DataFrame to save
            filepath: Output file path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_parquet(path, index=False)
    
    def get_file_format(self, filepath: str) -> str:
        """Get file format from filepath.
        
        Args:
            filepath: Path to file
            
        Returns:
            File format extension
        """
        return Path(filepath).suffix.lower()


def load_data(filepath: str) -> pd.DataFrame:
    """Convenience function to load data.
    
    Args:
        filepath: Path to data file
        
    Returns:
        DataFrame with data
    """
    loader = LNPDataLoader()
    return loader.load_file(filepath)


def save_data(data: pd.DataFrame, filepath: str, format: Optional[str] = None) -> None:
    """Convenience function to save data.
    
    Args:
        data: DataFrame to save
        filepath: Output file path
        format: Output format (csv, jsonl, parquet). Inferred from filepath if None
    """
    loader = LNPDataLoader()
    
    if format == "jsonl" or filepath.endswith(".jsonl"):
        loader.save_jsonl(data.to_dict("records"), filepath)
    elif format == "parquet" or filepath.endswith(".parquet"):
        loader.save_parquet(data, filepath)
    else:
        loader.save_csv(data, filepath)