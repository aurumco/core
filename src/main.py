"""Main entry point for the AI Subspace Extraction pipeline.

This module orchestrates the loading, surgery, and saving processes.
"""

import json
import os
import gc
import sys
import zipfile
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


def main() -> None:
    """Executes the main pipeline."""
    try:
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
            # We do not generate adapters here because SVD compression is a replacement,
            # not an addition. The modified model (now low-rank) is resident in memory
            # (re-quantized to 4-bit).

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
            # We save the in-place modified model as safetensors first.
            # This satisfies the requirement for "two main formats".
            safetensors_path = config.paths.model_4bit_dir / "safetensors"
            logger.info("Saving Safetensors format...")
            model.save_pretrained(safetensors_path)
            tokenizer.save_pretrained(safetensors_path)
            logger.info(f"Safetensors saved to {safetensors_path}")

            # 2. GGUF Export (Native Unsloth)
            # The model already contains the SVD-compressed weights (in-place).
            # save_pretrained_gguf will handle saving this modified state.

            if hasattr(model, "save_pretrained_gguf"):
                gguf_path = config.paths.model_4bit_dir / "gguf"
                model.save_pretrained_gguf(
                    str(gguf_path),
                    tokenizer,
                    quantization_method="q4_k_m",
                )
                logger.info(f"GGUF saved to {gguf_path}")
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
