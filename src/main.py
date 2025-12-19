"""Main entry point for the AI Subspace Extraction pipeline.

This module orchestrates the loading, surgery, and saving processes.
"""

import json
import os
import gc
import sys
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import torch

# 1. Environment Setup (Prevent Fragmentation)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.analytics.engine import AnalyticsEngine
from src.config import AppConfig, ModelConfig, PathConfig, SurgeryConfig
from src.surgery.core import SubspaceExtractor
from src.surgery.loader import ModelLoader
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
        print("\n[WARNING] Low disk space detected! Pipeline might fail at export stage.\n")


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

            # NOTE: SubspaceExtractor now updates the model weights IN-PLACE.
            logger.debug(f"Processed {lname}")

        print("\n")

        # 5. Generate Aggregate Analytics
        ConsoleUI.status_update("Generating Global Analytics...")
        analytics.generate_rank_distribution(metadata_list)

        # Save Metadata JSON
        config.paths.metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(config.paths.metadata_dir / "surgery_report.json", "w") as f:
            json.dump(metadata_list, f, indent=2)

        # 6. Unsloth Quantization & Export
        ConsoleUI.status_update("Quantizing & Exporting (Safetensors + GGUF)...")

        # Cleanup before export
        gc.collect()
        torch.cuda.empty_cache()

        # Create output dir
        config.paths.model_4bit_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Save Safetensors (Modified Full Model)
            safetensors_path = config.paths.model_4bit_dir / "safetensors"
            logger.info("Saving Safetensors format...")
            model.save_pretrained(safetensors_path)
            tokenizer.save_pretrained(safetensors_path)
            logger.info(f"Safetensors saved to {safetensors_path}")

            # 2. GGUF Export (Native Unsloth) - Optimized for Kaggle Disk Limits
            # We use a temporary directory in /tmp for the intermediate conversion steps
            # to avoid filling up the working directory (20GB limit).

            logger.info("Starting GGUF conversion (using /tmp for intermediate storage)...")

            with tempfile.TemporaryDirectory(dir="/tmp") as temp_gguf_dir:
                logger.debug(f"Temporary GGUF staging: {temp_gguf_dir}")

                # BUGFIX: Explicitly save config and tokenizer to temp dir first.
                # Unsloth's save_pretrained_gguf can sometimes fail to find config.json
                # if the initial save step hiccups or if looking for existing config.
                # Pre-saving ensures it exists.
                model.config.save_pretrained(temp_gguf_dir)
                tokenizer.save_pretrained(temp_gguf_dir)

                if hasattr(model, "save_pretrained_gguf"):
                    # Save GGUF to temp dir
                    model.save_pretrained_gguf(
                        temp_gguf_dir,
                        tokenizer,
                        quantization_method="q4_k_m",
                    )

                    # Move the result to the final output directory
                    final_gguf_dir = config.paths.model_4bit_dir / "gguf"
                    final_gguf_dir.mkdir(parents=True, exist_ok=True)

                    # Find generated GGUF files
                    temp_path = Path(temp_gguf_dir)
                    found_files = list(temp_path.glob("*.gguf"))

                    if not found_files:
                        raise RuntimeError("No GGUF files found after conversion!")

                    for file in found_files:
                        shutil.move(str(file), str(final_gguf_dir / file.name))
                        logger.info(f"Moved {file.name} to {final_gguf_dir}")

                else:
                    logger.error("Unsloth GGUF export method not found on model.")
                    raise RuntimeError("save_pretrained_gguf missing")

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
