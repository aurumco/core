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
            # Check if it's 4-bit
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

            # Reconstruct approximation
            # W_approx = U_k @ diag(S_k) @ Vh_k
            W_approx = torch.matmul(U_k, torch.matmul(torch.diag(S_k), Vh_k))

            # Move results to CPU
            U_cpu = U_k.contiguous().cpu()
            S_cpu = S_k.contiguous().cpu()
            Vh_cpu = Vh_k.contiguous().cpu()
            W_approx_cpu = W_approx.cpu()  # Keep on CPU for assignment/quantization

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

            # Truncation Logic
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

        # 4. In-Place Update of Model Weights (The "Surgery")
        # We replace the original high-rank weights with the low-rank approximation.
        # This is mathematically correct for compression: W <- W_approx

        # Determine original dtype
        original_dtype = torch.bfloat16 if isinstance(layer, Linear4bit) else layer.weight.dtype

        # Convert W_approx to target dtype
        W_new = W_approx_cpu.to(dtype=original_dtype)  # type: ignore

        if isinstance(layer, Linear4bit):
            # For 4-bit layers, we must re-quantize the new approximated weights
            # to maintain the 4-bit structure and memory benefits.
            # Using nf4 as it's standard for Qwen/Unsloth

            # Ensure input is on same device as layer for quantization if needed,
            # or CPU if bitsandbytes supports it.
            # Ideally we quantize on GPU if possible for speed, but CPU saves VRAM.
            # W_new is on CPU.

            W_new_cuda = W_new.to(layer.weight.device)
            q_weight, q_state = F.quantize_4bit(
                W_new_cuda, quant_type="nf4", compress_statistics=True
            )

            # Update the layer in-place
            layer.weight.data = q_weight
            layer.weight.quant_state = q_state  # type: ignore

            # Cleanup
            del W_new_cuda

        elif isinstance(layer, nn.Linear):
            # Standard assignment
            layer.weight.data = W_new  # type: ignore
        else:
             # Fallback
            layer.weight.data = W_new # type: ignore

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
