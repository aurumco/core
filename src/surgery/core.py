"""Core extraction logic module.

This module implements the subspace extraction via SVD and truncation
on quantized model layers.
"""

from typing import Dict, Generator, Tuple
import torch
import torch.nn as nn
from bitsandbytes.nn import Linear4bit  # type: ignore
import bitsandbytes.functional as F
from src.config import SurgeryConfig
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class SubspaceExtractor:
    """Performs layer-wise subspace extraction using SVD.

    Attributes:
        config (SurgeryConfig): Configuration for extraction.
    """

    def __init__(self, config: SurgeryConfig) -> None:
        """Initialize the extractor.

        Args:
            config (SurgeryConfig): Surgery configuration.
        """
        self.config = config

    def extract(
        self, model: nn.Module
    ) -> Generator[Tuple[str, Dict[str, torch.Tensor]], None, None]:
        """Iterates over model layers and yields extracted subspaces.

        Args:
            model (nn.Module): The model to process.

        Yields:
            Tuple[str, Dict[str, torch.Tensor]]: (layer_name, {U, S, V}).
        """
        for name, module in model.named_modules():
            if self._should_process(name, module):
                logger.info(f"Processing layer: {name}")
                try:
                    subspace = self._process_layer(module)
                    yield name, subspace

                    # Explicit cleanup to manage VRAM
                    del subspace
                    torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"Failed to process layer {name}: {e}", exc_info=True)
                    # Continue to next layer instead of crashing entire pipeline?
                    # Constitution says "Silence is unacceptable", raising is better
                    # but for a long process, logging error and moving on might be preferred
                    # if possible. However, Article V says "Actionable".
                    # We will re-raise to ensure integrity.
                    raise

    def _should_process(self, name: str, module: nn.Module) -> bool:
        """Determines if a layer should be processed.

        Args:
            name (str): Layer name.
            module (nn.Module): Layer module.

        Returns:
            bool: True if should be processed.
        """
        is_linear = isinstance(module, (nn.Linear, Linear4bit))

        if not is_linear:
            return False

        if self.config.target_modules:
            return any(target in name for target in self.config.target_modules)

        return True

    def _process_layer(self, layer: nn.Module) -> Dict[str, torch.Tensor]:
        """Performs SVD and truncation on a single layer.

        Complexity: O(min(m,n)^2 * max(m,n)) for SVD.

        Args:
            layer (nn.Module): The layer to process.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing U, S, V tensors.
        """
        # 1. Dequantize / Get Weights
        # Move to float16 for precision during SVD
        if isinstance(layer, Linear4bit):
            # bitsandbytes stores weights in .weight which is a Params4bit object.
            # We must use dequantize_4bit to get the actual float values.
            # layer.weight.quant_state holds the quantization parameters.
            weight = F.dequantize_4bit(
                layer.weight.data, layer.weight.quant_state  # type: ignore
            ).float()
        elif isinstance(layer, nn.Linear):
            weight = layer.weight.data.float()
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

        # 2. SVD
        # Use float32 for stability in SVD if possible, or float16 if VRAM is tight.
        # Given dual T4, we have some room, but large layers (4096*4096) are big.
        # SVD on GPU is faster.
        try:
            # Perform SVD
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        except RuntimeError:
            # Fallback to CPU if OOM
            logger.warning("GPU SVD failed (likely OOM), falling back to CPU")
            weight_cpu = weight.cpu()
            U, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)
            U, S, Vh = U.to(weight.device), S.to(weight.device), Vh.to(weight.device)

        # 3. Truncation
        k = int(len(S) * self.config.truncation_ratio)
        if k < 1:
            k = 1

        U_k = U[:, :k]
        S_k = S[:k]
        V_k = Vh[:k, :]  # Vh is V transpose (usually denoted VT)

        return {
            "U": U_k.half(),  # Store as half to save space
            "S": S_k.half(),
            "V": V_k.half(),  # This is V^T actually, consistent with torch.linalg.svd output
        }
