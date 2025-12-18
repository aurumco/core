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
        energy_threshold (float): Energy threshold for dynamic truncation (e.g., 0.99).
        target_modules (Optional[list[str]]): List of module names to target.
            If None, targets all Linear layers.
    """

    energy_threshold: float = 0.99
    target_modules: Optional[list[str]] = None


@dataclass
class PathConfig:
    """Configuration for file system paths.

    Args:
        output_dir (Path): Root directory for outputs.
    """

    output_dir: Path

    @property
    def model_4bit_dir(self) -> Path:
        """
        Directory path for the finalized 4-bit model.
        
        Returns:
            Path: Path to the "model_4bit" subdirectory inside the configured output directory.
        """
        return self.output_dir / "model_4bit"

    @property
    def extraction_dir(self) -> Path:
        """
        Path to the directory where extracted tensors are stored (legacy support).
        
        Returns:
            Path: Directory for extracted tensors, computed as `output_dir / "extracted_subspace"`. Kept for backward compatibility.
        """
        return self.output_dir / "extracted_subspace"

    @property
    def adapters_dir(self) -> Path:
        """
        Directory path for adapter configurations (legacy support).
        
        Provided for backward compatibility; returns the path to the "adapters" subdirectory under the configured output_dir.
        
        Returns:
            Path: Path to the adapters directory.
        """
        return self.output_dir / "adapters"

    @property
    def analytics_dir(self) -> Path:
        """
        Provide the path to the analytics reports directory.
        
        Returns:
            Path: Path to the analytics reports directory (equivalent to output_dir / "analytics").
        """
        return self.output_dir / "analytics"

    @property
    def metadata_dir(self) -> Path:
        """
        Provide the path to the surgery metadata directory.
        
        Returns:
            metadata_dir (Path): Path pointing to the `metadata` subdirectory inside `output_dir`.
        """
        return self.output_dir / "metadata"

    @property
    def backup_dir(self) -> Path:
        """
        Get the path to the backup directory used for zipped archives.
        
        Returns:
            backup_dir (Path): Path to the directory where backup (zipped) archives are stored.
        """
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