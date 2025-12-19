"""Core extraction logic module.

This module implements the subspace extraction via SVD and truncation
on quantized model layers.
"""

from typing import Dict, Generator, Tuple, Any
import gc
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

        # 2. SVD Strategy: Offload to Surgery GPU (e.g., cuda:1)
        torch.cuda.empty_cache()

        # Determine target device for SVD (prefer cuda:1 if available, else cuda:0)
        # Assuming we loaded model on CPU as requested to save VRAM.
        if torch.cuda.device_count() > 1:
            surgery_device = torch.device("cuda:1")
        elif torch.cuda.is_available():
            surgery_device = torch.device("cuda:0")
        else:
            surgery_device = torch.device("cpu")

        try:
            if weight.numel() > 10000000:
                logger.debug(f"  -> Offloading SVD to {surgery_device}...")

            # Move weight to surgery device for SVD
            weight_for_svd = weight.to(surgery_device)
            U, S, Vh = torch.linalg.svd(weight_for_svd, full_matrices=False)

            # 3. Dynamic Truncation on Surgery Device
            # E = sum(sigma^2)
            S_squared = S**2
            total_energy = torch.sum(S_squared)
            cumulative_energy = torch.cumsum(S_squared, dim=0)

            # Find k where cumulative energy >= threshold * total
            threshold_energy = self.config.energy_threshold * total_energy
            k_val = torch.searchsorted(cumulative_energy, threshold_energy).item()
            k = int(k_val) + 1
            k = max(1, min(k, len(S)))

            # Truncate
            U_k = U[:, :k]
            S_k = S[:k]
            Vh_k = Vh[:k, :]

            # Reconstruct approximation on Surgery Device
            W_approx = torch.matmul(U_k, torch.matmul(torch.diag(S_k), Vh_k))

            # Move results to CPU immediately to free VRAM
            # As per strict instruction: contiguous() on GPU, then move to CPU
            U_cpu = U_k.contiguous().cpu()
            S_cpu = S_k.contiguous().cpu()
            Vh_cpu = Vh_k.contiguous().cpu()
            W_approx_cpu = W_approx.cpu()

            energy_preserved = float((cumulative_energy[k - 1] / total_energy).item())
            original_size = weight.numel()
            compressed_size = U_k.numel() + S_k.numel() + Vh_k.numel()

            # Cleanup VRAM immediately
            del (
                weight_for_svd,
                U,
                S,
                Vh,
                U_k,
                S_k,
                Vh_k,
                W_approx,
                cumulative_energy,
                S_squared,
            )
            torch.cuda.empty_cache()

        except RuntimeError as e:
            logger.warning(f"  -> GPU SVD failed: {e}. Fallback to CPU.")
            # Fallback
            weight_cpu = weight.cpu()
            U, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)

            # Truncation Logic (Duplicated for fallback - could be refactored)
            S_squared = S**2
            total_energy = torch.sum(S_squared)
            cumulative_energy = torch.cumsum(S_squared, dim=0)
            threshold_energy = self.config.energy_threshold * total_energy
            k_val = torch.searchsorted(cumulative_energy, threshold_energy).item()
            k = int(k_val) + 1
            k = max(1, min(k, len(S)))

            U_k = U[:, :k]
            S_k = S[:k]
            Vh_k = Vh[:k, :]
            W_approx_cpu = torch.matmul(U_k, torch.matmul(torch.diag(S_k), Vh_k))

            U_cpu = U_k.contiguous()
            S_cpu = S_k.contiguous()
            Vh_cpu = Vh_k.contiguous()

            energy_preserved = float((cumulative_energy[k - 1] / total_energy).item())
            original_size = weight.numel()
            compressed_size = U_k.numel() + S_k.numel() + Vh_k.numel()

        # 4. In-Place Update of Model Weights
        # Determine original dtype
        original_dtype = (
            layer.weight.dtype if not isinstance(layer, Linear4bit) else torch.bfloat16
        )

        # Convert W_approx to target dtype (bf16/fp16) on CPU
        W_new = W_approx_cpu.to(dtype=original_dtype)

        if isinstance(layer, nn.Linear):
            # Assign to layer on CPU (since device_map="cpu")
            layer.weight.data = W_new  # type: ignore
        else:
            # For Unsloth/custom layers
            layer.weight.data = W_new  # type: ignore

        # Aggressive Cleanup
        del weight, W_approx_cpu, W_new
        gc.collect()

        return {
            "U": U_cpu,
            "S": S_cpu,
            "Vh": Vh_cpu,
            "rank": k,
            "energy_preserved": energy_preserved,
            "original_size": original_size,
            "compressed_size": compressed_size,
        }
