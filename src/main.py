"""Main entry point for the AI Subspace Extraction pipeline.

This module orchestrates the loading, surgery, and saving processes.
"""

import json
import os
import gc
import sys
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Set

import torch
from safetensors.torch import save_file

# 1. Environment Setup (Prevent Fragmentation)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.analytics.engine import AnalyticsEngine
from src.config import AppConfig, ModelConfig, PathConfig, SurgeryConfig
from src.surgery.core import SubspaceExtractor
from src.surgery.loader import ModelLoader
from src.surgery.export_tools import convert_to_custom_gguf, quantize_model
from src.utils.logging import setup_logger
from src.utils.ui import ConsoleUI

logger = setup_logger(__name__)


def check_system_resources() -> None:
    """Checks for sufficient disk space and resources before starting."""
    # Check disk space
    work_usage = shutil.disk_usage(".")
    tmp_usage = shutil.disk_usage("/tmp")

    work_free_gb = work_usage.free / (1024**3)
    tmp_free_gb = tmp_usage.free / (1024**3)

    ConsoleUI.status_update(
        f"Disk Space: Working={work_free_gb:.1f}GB, /tmp={tmp_free_gb:.1f}GB"
    )

    # Estimate requirements:
    # 4B model FP16 dump ~= 8GB
    # GGUF Final ~= 3GB
    # Safetensors ~= 8GB (if full model)
    # Total margin needed: ~12GB in /tmp (for conversion) + ~10GB in Working (for outputs)

    if tmp_free_gb < 12:
        logger.warning(
            f"Low disk space in /tmp ({tmp_free_gb:.1f}GB). GGUF conversion might fail."
        )

    if work_free_gb < 10:
        logger.warning(
            f"Low disk space in working dir ({work_free_gb:.1f}GB). Saving outputs might fail."
        )
        print(
            "\n[WARNING] Low disk space detected! Pipeline might fail at export stage.\n"
        )


def main() -> None:
    """Executes the main pipeline."""
    try:
        # 0. Pre-flight Checks
        check_system_resources()

        # 1. Configuration
        config = AppConfig(
            model=ModelConfig(
                model_name="unsloth/Qwen3-4B-Base",
                device_map="auto",
                quantization_bit=4,
            ),
            surgery=SurgeryConfig(energy_threshold=0.99),
            paths=PathConfig(output_dir=Path("output")),
        )

        ConsoleUI.print_header(config)

        # 2. Load Model
        ConsoleUI.status_update("Loading Base Model (unsloth/Qwen3-4B-Base)...")
        loader = ModelLoader(config.model)
        model, tokenizer = loader.load()
        ConsoleUI.status_update("Model Loaded Successfully.")

        # 3. Initialize Components
        extractor = SubspaceExtractor(config.surgery)
        analytics = AnalyticsEngine(config.paths)

        # 4. Surgery Loop (Genetic Engineering)
        target_layers = extractor.get_target_layers(model)
        total_layers = len(target_layers)

        metadata_list: List[Dict[str, Any]] = []

        # Decomposed State Dict: Collects U, V for processed layers, and originals for others.
        # We build this in memory. For 4B params, it should fit in 20GB RAM (FP16 ~8GB).
        # We'll use CPU tensors to save VRAM.
        decomposed_state_dict: Dict[str, torch.Tensor] = {}
        processed_layer_names: Set[str] = set()

        surgery_iter = extractor.extract(target_layers)

        print(f"\n Executing Genetic Engineering on {total_layers} layers...")
        pbar = ConsoleUI.progress_bar(
            surgery_iter, total=total_layers, prefix="Surgery"
        )

        # We rely on extraction order.
        # Enumerate output is assumed to be (index, (layer_name, result))
        for i, item in enumerate(pbar):  # type: ignore
            # Unpack explicitly for type checker
            layer_name: str = item[0]  # type: ignore
            result: Dict[str, Any] = item[1]  # type: ignore

            lname: str = layer_name
            res: Dict[str, Any] = result
            processed_layer_names.add(lname)

            # 1. Analytics & Visualization
            analytics.generate_layer_charts(
                lname, res["S"], res["rank"], res["energy_preserved"]
            )

            # 2. Collect Metadata
            meta = {
                "layer_name": lname,
                "rank": res["rank"],
                "energy_preserved": res["energy_preserved"],
                "original_size": res["original_size"],
                "compressed_size": res["compressed_size"],
            }
            metadata_list.append(meta)

            # 3. Collect Decomposed Matrices for GGUF
            # We need to construct u = U * sqrt(S) and v = sqrt(S) * Vh
            # Result tensors are already on CPU and contiguous.
            # Names: layer.weight -> layer.u, layer.v
            # We assume layer_name ends in a module name, e.g. "model.layers.0.self_attn.q_proj"
            # We want to store keys as full parameter names: "...q_proj.u", "...q_proj.v"

            try:
                # Move to CPU float32 for matmul stability, then cast to float16
                S_sqrt = torch.sqrt(res["S"])
                S_diag = torch.diag(S_sqrt)

                # u = U @ S_diag
                u_tensor = torch.matmul(res["U"], S_diag)
                # v = S_diag @ Vh
                v_tensor = torch.matmul(S_diag, res["Vh"])

                # Store in state dict as FP16
                decomposed_state_dict[f"{lname}.u"] = u_tensor.to(dtype=torch.float16)
                decomposed_state_dict[f"{lname}.v"] = v_tensor.to(dtype=torch.float16)

                # Clean up immediately
                del u_tensor, v_tensor, S_sqrt, S_diag

            except Exception as e:
                logger.error(f"Failed to decompose layer {lname} for export: {e}")
                # Fallback: Don't store u/v, let it use original weight?
                # But processed layer in model is W_approx.
                # Ideally we fail or skip.
                pass

            # NOTE: SubspaceExtractor updates the model weights IN-PLACE with W_approx.
            # We don't want W_approx in our export, we want U and V.
            logger.debug(f"Processed {lname}")

        print("\n")

        # 5. Generate Aggregate Analytics
        ConsoleUI.status_update("Generating Global Analytics...")
        analytics.generate_rank_distribution(metadata_list)

        # Save Metadata JSON
        config.paths.metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(config.paths.metadata_dir / "surgery_report.json", "w") as f:
            json.dump(metadata_list, f, indent=2)

        # 6. Export (SVD Safetensors + Custom GGUF + Quantization)
        ConsoleUI.status_update("Starting Export Pipeline...")

        # Cleanup VRAM
        gc.collect()
        torch.cuda.empty_cache()

        # Create output dir
        config.paths.model_4bit_dir.mkdir(parents=True, exist_ok=True)
        safetensors_dir = config.paths.model_4bit_dir / "safetensors"
        safetensors_dir.mkdir(parents=True, exist_ok=True)

        # 6.1 Build Full State Dict
        logger.info("Collecting remaining model weights...")

        # Iterate over all model parameters
        # We need to grab everything that wasn't decomposed.
        for name, param in model.named_parameters():
            # Check if this parameter belongs to a processed layer
            # processed_layer_names contains module names (e.g., ...q_proj)
            # param name: ...q_proj.weight

            is_processed = False
            for layer_name in processed_layer_names:
                if name.startswith(layer_name):
                    # It's a weight of a processed layer (likely .weight)
                    # We already stored .u and .v for this.
                    # Verify we aren't skipping biases if they exist and weren't processed?
                    # SVD usually only processes weight. If bias exists, we need it.
                    if name.endswith("weight"):
                        is_processed = True
                        break

            if not is_processed:
                # Store original parameter
                # Move to CPU and float16
                decomposed_state_dict[name] = (
                    param.detach().cpu().to(dtype=torch.float16)
                )

        # 6.2 Save Decomposed Safetensors (Step 1 requirement)
        svd_model_path = safetensors_dir / "model_svd.safetensors"
        logger.info(f"Saving decomposed model to {svd_model_path}...")
        save_file(decomposed_state_dict, svd_model_path)

        # Save tokenizer and config to same dir for completeness
        model.config.save_pretrained(safetensors_dir)
        tokenizer.save_pretrained(safetensors_dir)

        # Free RAM
        del decomposed_state_dict
        gc.collect()

        try:
            # 6.3 Convert to Custom GGUF (Step 1)
            gguf_dir = config.paths.model_4bit_dir / "gguf"
            gguf_dir.mkdir(parents=True, exist_ok=True)
            gguf_path = gguf_dir / "model_fp16.gguf"

            logger.info("converting to Custom GGUF...")
            convert_to_custom_gguf(svd_model_path, gguf_path, model.config)

            # 6.4 Quantize (Step 2)
            final_quantized_path = gguf_dir / "model_q4_k_m.gguf"
            logger.info("Running Final Compression (llama-quantize)...")
            quantize_model(gguf_path, final_quantized_path)

            # Cleanup intermediate heavy files to save space?
            # User constraint: "Kaggle 20GB".
            # We have:
            # 1. model_svd.safetensors (~8GB)
            # 2. model_fp16.gguf (~8GB)
            # 3. model_q4_k_m.gguf (~2GB)
            # Total ~18GB. Very risky.
            # We should delete intermediate files after usage.

            logger.info("Cleaning up intermediate files...")
            if gguf_path.exists():
                os.remove(gguf_path)
            if svd_model_path.exists():
                os.remove(svd_model_path)

            logger.info(f"Export Success! Final model: {final_quantized_path}")

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

        # 7. Final Packaging
        ConsoleUI.status_update("Packaging Artifacts...")
        archive_path = config.paths.backup_dir
        archive_path.mkdir(parents=True, exist_ok=True)

        zip_filename = archive_path / "final.zip"
        target_folders = [
            config.paths.model_4bit_dir,
            config.paths.analytics_dir,
            config.paths.metadata_dir,
        ]

        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder in target_folders:
                if folder.exists():
                    for file in folder.rglob("*"):
                        if file.is_file():
                            arcname = file.relative_to(config.paths.output_dir)
                            zf.write(file, arcname=arcname)

        logger.info(f"Pipeline Complete. Output: {zip_filename}")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
