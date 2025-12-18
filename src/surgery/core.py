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
        # Strict type check or Duck typing for Unsloth layers
        is_linear = isinstance(module, (nn.Linear, Linear4bit))
        if not is_linear:
            # Check for Unsloth or other custom Linear layers via class name and attributes
            class_name = module.__class__.__name__
            has_weight = hasattr(module, "weight")
            if "Linear" in class_name and has_weight:
                is_linear = True

        if not is_linear:
            return False

        if self.config.target_modules:
            return any(target in name for target in self.config.target_modules)

        return True

    def _process_layer(self, layer: nn.Module) -> Dict[str, torch.Tensor]:
        """Performs SVD and truncation on a single layer using optimal device.

        Complexity: O(min(m,n)^2 * max(m,n)) for SVD.

        Args:
            layer (nn.Module): The layer to process.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing U, S, V tensors.
        """
        # 1. Dequantize / Get Weights
        if isinstance(layer, Linear4bit):
            # bitsandbytes stores weights in .weight which is a Params4bit object.
            # We must use dequantize_4bit to get the actual float values.
            # layer.weight.quant_state holds the quantization parameters.
            weight = F.dequantize_4bit(
                layer.weight.data, layer.weight.quant_state  # type: ignore
            ).half()
        elif hasattr(layer, "weight"):
            # Handle standard Linear and Unsloth/Custom layers via duck typing
            # Cast to half to ensure we don't blow up memory with float32
            # Mypy sees layer.weight as Any or Module usually, we need to be explicit about .data
            weight = layer.weight.data.half()  # type: ignore
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

        # 2. SVD Strategy: Offload to Free GPU
        torch.cuda.empty_cache()

        # Default to current device
        surgery_device = weight.device

        # If multiple GPUs, try to offload to the other one
        if torch.cuda.device_count() > 1 and weight.device.type == "cuda":
            current_idx = weight.device.index if weight.device.index is not None else 0
            # Switch between 0 and 1 (assuming 2 GPUs context)
            target_idx = 1 - current_idx
            surgery_device = torch.device(f"cuda:{target_idx}")

        try:
            # Move weight to surgery device
            # We cast to float32 because SVD is generally more stable on float32
            # and CPU requires it anyway.
            weight_for_svd = weight.to(surgery_device).float()

            U, S, Vh = torch.linalg.svd(weight_for_svd, full_matrices=False)

            # Move results back to original device (to clear surgery device VRAM)
            U = U.to(weight.device)
            S = S.to(weight.device)
            Vh = Vh.to(weight.device)

            # Cleanup surgery device
            del weight_for_svd
            torch.cuda.empty_cache()

        except RuntimeError as e:
            # Fallback to CPU if OOM even on second GPU
            logger.warning(
                f"GPU SVD failed on {surgery_device}: {e}. Falling back to CPU"
            )
            weight_cpu = weight.cpu().float()
            U, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)
            device = weight.device
            U = U.to(device)
            S = S.to(device)
            Vh = Vh.to(device)

        # 3. Truncation
        k = int(len(S) * self.config.truncation_ratio)
        if k < 1:
            k = 1

        # NOTE: Must be contiguous for safetensors saving
        U_k = U[:, :k].contiguous()
        S_k = S[:k].contiguous()
        V_k = Vh[:k, :].contiguous()  # Vh is V transpose (usually denoted VT)

        return {
            "U": U_k.half(),
            "S": S_k.half(),
            "Vh": V_k.half(),  # Renamed key to Vh as requested
        }
