"""Core extraction logic module.

This module implements the subspace extraction via SVD and truncation
on quantized model layers.
"""

from typing import Dict, Generator, Tuple, Any
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

    def get_target_layers(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        """Identifies all layers to be processed.

        Args:
            model (nn.Module): The model to scan.

        Returns:
            list[tuple[str, nn.Module]]: List of (name, module) tuples.
        """
        targets = []
        for name, module in model.named_modules():
            if self._should_process(name, module):
                targets.append((name, module))
        return targets

    def extract(
        self, layers: list[tuple[str, nn.Module]]
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Iterates over provided layers and yields extracted subspaces.

        Args:
            layers (list[tuple[str, nn.Module]]): List of layers to process.

        Yields:
            Tuple[str, Dict[str, Any]]: (layer_name, result_dict).
        """
        for name, module in layers:
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

    def _process_layer(self, layer: nn.Module) -> Dict[str, Any]:
        """Performs SVD and truncation on a single layer using dynamic energy threshold.

        Args:
            layer (nn.Module): The layer to process.

        Returns:
            Dict[str, Any]: Contains "U", "S", "Vh", "rank", "energy_preserved".
        """
        # 1. Get Weights (Always cast to float32 for SVD stability)
        # Note: If layer is Linear4bit, we dequantize. If it's already bf16/fp16, we cast.

        if hasattr(layer, "weight"):
            # Check if it's 4-bit (shouldn't be in this new pipeline but good to handle)
            if isinstance(layer, Linear4bit):
                weight = F.dequantize_4bit(
                    layer.weight.data, layer.weight.quant_state  # type: ignore
                ).float()
            else:
                weight = layer.weight.data.float()  # type: ignore
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

        # 2. SVD Strategy: Offload to Free GPU
        torch.cuda.empty_cache()
        surgery_device = weight.device

        # Ping-Pong Strategy
        if torch.cuda.device_count() > 1 and weight.device.type == "cuda":
            current_idx = weight.device.index if weight.device.index is not None else 0
            target_idx = 1 - current_idx
            surgery_device = torch.device(f"cuda:{target_idx}")

        try:
            if weight.numel() > 10000000:
                logger.info(f"  -> Offloading SVD to {surgery_device}...")

            weight_for_svd = weight.to(surgery_device)
            U, S, Vh = torch.linalg.svd(weight_for_svd, full_matrices=False)

            # Move results back
            U = U.to(weight.device)
            S = S.to(weight.device)
            Vh = Vh.to(weight.device)

            del weight_for_svd
            torch.cuda.empty_cache()

        except RuntimeError:
            logger.warning("  -> GPU SVD failed. Fallback to CPU.")
            weight_cpu = weight.cpu()
            U, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)
            device = weight.device
            U = U.to(device)
            S = S.to(device)
            Vh = Vh.to(device)

        # 3. Dynamic Truncation (Energy Threshold)
        # E = sum(sigma^2)
        S_squared = S**2
        total_energy = torch.sum(S_squared)
        cumulative_energy = torch.cumsum(S_squared, dim=0)

        # Find k where cumulative energy >= threshold * total
        threshold_energy = self.config.energy_threshold * total_energy
        # torch.searchsorted requires sorted sequence. cumulative is sorted ascending.
        k_val = torch.searchsorted(cumulative_energy, threshold_energy).item()
        # Ensure k is int and 1-based index
        k = int(k_val) + 1

        # Safety bounds
        k = max(1, min(k, len(S)))

        # Reconstruct Weight W' = U_k * S_k * Vh_k
        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]

        # Reconstruct approximation
        W_approx = torch.matmul(U_k, torch.matmul(torch.diag(S_k), Vh_k))

        # 4. In-Place Update of Model Weights
        # We replace the layer's weight with the approximated weight.
        # This prepares the model for QAT/Quantization.
        # Convert back to original dtype (likely bf16 or fp16)
        # Type ignore: we assume these properties exist on the layer.weight
        original_dtype = (
            layer.weight.dtype if not isinstance(layer, Linear4bit) else torch.bfloat16
        )  # Assumption

        if isinstance(layer, nn.Linear):
            # Explicit cast to dtype required by mypy
            # Cast original_dtype to torch.dtype just to be safe for mypy if it inferred Union
            layer.weight.data = W_approx.to(dtype=original_dtype)  # type: ignore
        else:
            # If complex layer (Unsloth), try assigning to .weight.data
            # Be precise with arguments for to()
            target_device = layer.weight.device
            target_dtype = layer.weight.dtype
            layer.weight.data = W_approx.to(device=target_device, dtype=target_dtype)  # type: ignore

        return {
            "U": U_k.contiguous(),  # Keep if we want to save subspace
            "S": S_k.contiguous(),
            "Vh": Vh_k.contiguous(),
            "rank": k,
            "energy_preserved": float((cumulative_energy[k - 1] / total_energy).item()),
            "original_size": weight.numel(),
            "compressed_size": U_k.numel() + S_k.numel() + Vh_k.numel(),
        }
