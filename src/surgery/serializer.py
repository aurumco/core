"""Serialization module.

This module handles saving the extracted subspaces to disk using SafeTensors.
"""

from typing import Any, Dict
import torch
from safetensors.torch import save_file
from src.config import PathConfig
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class ModelSerializer:
    """Handles serialization of extracted model components.

    Attributes:
        config (PathConfig): Path configuration.
    """

    def __init__(self, config: PathConfig) -> None:
        """Initialize the serializer.

        Args:
            config (PathConfig): Path configuration.
        """
        self.config = config

    def save_layer_subspace(
        self, layer_name: str, subspace: Dict[str, torch.Tensor]
    ) -> None:
        """Saves the extracted subspace for a single layer.

        Args:
            layer_name (str): The name of the layer.
            subspace (Dict[str, torch.Tensor]): The extracted matrices (U, S, V).
        """
        # Create directory for the layer
        # Sanitize layer name for filesystem
        safe_name = layer_name.replace(".", "_")
        layer_dir = self.config.extraction_dir / safe_name
        layer_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving subspace for {layer_name} to {layer_dir}")

        # Save using safetensors
        # Flatten dictionary or save as individual files?
        # A single safetensors file per layer is cleaner.
        save_file(subspace, layer_dir / "subspace.safetensors")

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Saves metadata about the extraction process.

        Args:
             metadata (Dict[str, Any]): Metadata to save.
        """
        # TODO: Implement metadata saving (JSON) if needed
        pass
