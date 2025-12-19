"""Analytics engine for visualization and reporting.

This module generates visualizations for subspace energy, scree plots, and
rank distributions.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Dict, Any
from src.config import PathConfig
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class AnalyticsEngine:
    """Handles generation of analytics charts and reports."""

    def __init__(self, config: PathConfig) -> None:
        """Initialize the analytics engine.

        Args:
            config (PathConfig): Path configuration.
        """
        self.config = config
        self.config.analytics_dir.mkdir(parents=True, exist_ok=True)
        (self.config.analytics_dir / "layers").mkdir(exist_ok=True)

    def generate_layer_charts(
        self, layer_name: str, S: torch.Tensor, k: int, energy_ratio: float
    ) -> None:
        """Generates Scree Plot and Cumulative Energy chart for a layer.

        Args:
            layer_name (str): Name of the layer.
            S (torch.Tensor): Singular values (1D tensor).
            k (int): Selected rank.
            energy_ratio (float): Preserved energy ratio.
        """
        S_np = S.float().cpu().numpy()
        total_energy = np.sum(S_np**2)
        cumulative_energy = np.cumsum(S_np**2) / total_energy

        # Sanitize filename
        safe_name = layer_name.replace(".", "_")

        # 1. Scree Plot
        plt.figure(figsize=(10, 6))
        plt.plot(S_np, label="Singular Values")
        plt.axvline(x=k, color="r", linestyle="--", label=f"Cutoff (k={k})")
        plt.title(f"Scree Plot: {layer_name}")
        plt.xlabel("Index")
        plt.ylabel("Singular Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.config.analytics_dir / "layers" / f"{safe_name}_scree.png")
        plt.close()

        # 2. Cumulative Energy
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_energy, label="Cumulative Energy")
        plt.axhline(
            y=energy_ratio,
            color="g",
            linestyle="--",
            label=f"Threshold ({energy_ratio})",
        )
        plt.axvline(x=k, color="r", linestyle="--", label=f"Cutoff (k={k})")
        plt.title(f"Cumulative Energy: {layer_name}")
        plt.xlabel("Index")
        plt.ylabel("Energy Ratio")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.config.analytics_dir / "layers" / f"{safe_name}_energy.png")
        plt.close()

    def generate_rank_distribution(self, metadata_list: List[Dict[str, Any]]) -> None:
        """Generates a bar chart of rank distribution across layers.

        Args:
            metadata_list (List[Dict[str, Any]]): List of metadata dicts.
        """
        layer_names = [m["layer_name"] for m in metadata_list]
        ranks = [m["rank"] for m in metadata_list]

        short_names = [n if "." not in n else ".".join(n.split(".")[-2:]) for n in layer_names]

        plt.figure(figsize=(15, 8))
        plt.bar(short_names, ranks)
        plt.xticks(rotation=90)
        plt.title("Rank Distribution Across Layers")
        plt.xlabel("Layer")
        plt.ylabel("Selected Rank")
        plt.tight_layout()
        plt.savefig(self.config.analytics_dir / "rank_distribution.png")
        plt.close()
