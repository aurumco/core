"""Configuration definitions for the project.

This module defines configuration classes for model loading, surgery parameters,
and file paths, ensuring type safety and clarity.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the model to be processed.

    Args:
        model_name (str): HuggingFace model ID or local path.
        device_map (str): Device mapping strategy (e.g., "auto").
        quantization_bit (int): Quantization level (e.g., 4).
    """

    model_name: str
    device_map: str = "auto"
    quantization_bit: int = 4


@dataclass
class SurgeryConfig:
    """Configuration for the surgery/extraction process.

    Args:
        truncation_ratio (float): Ratio of singular values to retain (0.0 to 1.0).
        target_modules (Optional[list[str]]): List of module names to target.
            If None, targets all Linear layers.
    """

    truncation_ratio: float = 0.2
    target_modules: Optional[list[str]] = None


@dataclass
class PathConfig:
    """Configuration for file system paths.

    Args:
        output_dir (Path): Root directory for outputs.
        extraction_dir (Path): Directory for extracted subspace tensors.
        backup_dir (Path): Directory for backups/archives.
    """

    output_dir: Path

    @property
    def base_model_dir(self) -> Path:
        """Returns the directory for base model configuration."""
        return self.output_dir / "base_model"

    @property
    def extraction_dir(self) -> Path:
        """Returns the directory for extracted tensors."""
        return self.output_dir / "extracted_subspace"

    @property
    def adapters_dir(self) -> Path:
        """Returns the directory for adapter configurations."""
        return self.output_dir / "adapters"

    @property
    def backup_dir(self) -> Path:
        """Returns the directory for zipped archives."""
        return self.output_dir / "backup"


@dataclass
class AppConfig:
    """Master configuration aggregating all sub-configs.

    Args:
        model (ModelConfig): Model configuration.
        surgery (SurgeryConfig): Surgery configuration.
        paths (PathConfig): Path configuration.
    """

    model: ModelConfig
    surgery: SurgeryConfig
    paths: PathConfig
