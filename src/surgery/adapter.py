"""Adapter generation module.

This module converts SVD components into LoRA-compatible adapters.
"""

from typing import Dict
import torch
from src.config import SurgeryConfig
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class AdapterGenerator:
    """Generates LoRA adapters from SVD components.

    Attributes:
        config (SurgeryConfig): Surgery configuration.
    """

    def __init__(self, config: SurgeryConfig) -> None:
        """Initialize the generator.

        Args:
            config (SurgeryConfig): Surgery configuration.
        """
        self.config = config

    def generate(self, subspace: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Converts U, S, Vh into LoRA A and B matrices.

        Formula:
            W_approx = U @ diag(S) @ Vh
            Let A = diag(sqrt(S)) @ Vh
            Let B = U @ diag(sqrt(S))
            Then W_approx = B @ A

        Args:
            subspace (Dict[str, torch.Tensor]): Contains "U", "S", "Vh".

        Returns:
            Dict[str, torch.Tensor]: Contains "lora_A", "lora_B".
        """
        U = subspace["U"].float()  # Calculations in float32 for precision
        S = subspace["S"].float()
        Vh = subspace["Vh"].float()

        # Sqrt of singular values for balanced decomposition
        sqrt_S = torch.sqrt(S)
        diag_sqrt_S = torch.diag(sqrt_S)

        # Calculate A and B
        # lora_A shape: (r, in_features) -> corresponds to Vh (r, in)
        # lora_B shape: (out_features, r) -> corresponds to U (out, r)

        # A = diag(sqrt(S)) @ Vh
        lora_A = torch.matmul(diag_sqrt_S, Vh)

        # B = U @ diag(sqrt(S))
        lora_B = torch.matmul(U, diag_sqrt_S)

        # Cast back to half for storage/usage
        return {"lora_A": lora_A.half(), "lora_B": lora_B.half()}
