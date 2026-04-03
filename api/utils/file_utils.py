"""Utility functions for file operations."""

import json
import os
from pathlib import Path
from typing import Any, Optional


def get_project_root() -> Path:
    """Get project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def ensure_dir(path: str) -> Path:
    """Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_name: str = "settings") -> dict:
    """Load configuration from YAML.
    
    Args:
        config_name: Config name (without .yaml)
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    config_path = get_project_root() / "config" / f"{config_name}.yaml"
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_json(filepath: str) -> dict:
    """Load JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        JSON data
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save JSON file.
    
    Args:
        data: Data to save
        filepath: Output path
        indent: JSON indentation
    """
    ensure_dir(Path(filepath).parent)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_labels() -> dict:
    """Load label mapping.
    
    Returns:
        Label mapping dictionary
    """
    return load_config("labels")["label_to_id"]


def get_data_dir(subdir: Optional[str] = None) -> Path:
    """Get data directory path.
    
    Args:
        subdir: Subdirectory (raw, processed, external)
        
    Returns:
        Path to data directory
    """
    data_dir = get_project_root() / "data"
    
    if subdir:
        data_dir = data_dir / subdir
    
    return ensure_dir(str(data_dir))


def get_checkpoint_dir(subdir: Optional[str] = None) -> Path:
    """Get checkpoint directory path.
    
    Args:
        subdir: Subdirectory name
        
    Returns:
        Path to checkpoint directory
    """
    checkpoint_dir = get_project_root() / "checkpoints"
    
    if subdir:
        checkpoint_dir = checkpoint_dir / subdir
    
    return ensure_dir(str(checkpoint_dir))


def list_files(directory: str, extension: Optional[str] = None) -> list[Path]:
    """List files in directory.
    
    Args:
        directory: Directory path
        extension: File extension filter (e.g., ".jsonl")
        
    Returns:
        List of file paths
    """
    dir_path = Path(directory)
    
    if extension:
        return list(dir_path.glob(f"*{extension}"))
    
    return [f for f in dir_path.iterdir() if f.is_file()]


def get_file_size(filepath: str) -> int:
    """Get file size in bytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(filepath)


def format_size(size_bytes: int) -> str:
    """Format file size for display.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    
    return f"{size_bytes:.1f} TB"