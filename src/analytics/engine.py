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
        """
        Prepare analytics output directories using the provided path configuration.
        
        Parameters:
            config (PathConfig): Configuration containing `analytics_dir`; the constructor ensures `analytics_dir` and a `layers` subdirectory exist, creating them if necessary.
        """
        self.config = config
        self.config.analytics_dir.mkdir(parents=True, exist_ok=True)
        (self.config.analytics_dir / "layers").mkdir(exist_ok=True)

    def generate_layer_charts(
        self, layer_name: str, S: torch.Tensor, k: int, energy_ratio: float
    ) -> None:
        """
        Generate scree and cumulative energy plots for a given layer and save them to disk.
        
        Parameters:
            layer_name (str): Layer identifier used in plot titles and filenames; dots are replaced with underscores for filenames.
            S (torch.Tensor): 1D tensor of singular values for the layer.
            k (int): Cutoff index (selected rank) shown as a vertical line on both plots.
            energy_ratio (float): Target cumulative energy threshold (0.0–1.0) shown as a horizontal line on the energy plot.
        
        Notes:
            Saves two PNG files to the `analytics_dir/layers` directory in the configured PathConfig:
            `{safe_name}_scree.png` and `{safe_name}_energy.png`.
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
        """
        Generate a bar chart showing selected rank per layer and save it to analytics_dir/rank_distribution.png.
        
        Parameters:
            metadata_list (List[Dict[str, Any]]): List of metadata dictionaries each containing at least "layer_name" (str) and "rank" (int). Layer names are abbreviated to the last two dot-separated components for x-axis labels.
        """
        layer_names = [m["layer_name"] for m in metadata_list]
        ranks = [m["rank"] for m in metadata_list]

        # Abbreviate names for X-axis
        short_names = [n.split(".")[-2] + "." + n.split(".")[-1] for n in layer_names]

        plt.figure(figsize=(15, 8))
        plt.bar(short_names, ranks)
        plt.xticks(rotation=90)
        plt.title("Rank Distribution Across Layers")
        plt.xlabel("Layer")
        plt.ylabel("Selected Rank")
        plt.tight_layout()
        plt.savefig(self.config.analytics_dir / "rank_distribution.png")
        plt.close()