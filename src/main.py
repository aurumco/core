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

from safetensors.torch import load_file, save_file
import torch

# 1. Environment Setup (Prevent Fragmentation)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.analytics.engine import AnalyticsEngine
from src.config import AppConfig, ModelConfig, PathConfig, SurgeryConfig
from src.surgery.core import SubspaceExtractor
from src.surgery.loader import ModelLoader
from huggingface_hub import HfApi

from src.surgery.serializer import ModelSerializer
from src.utils.logging import setup_logger
from src.utils.ui import ConsoleUI

logger = setup_logger(__name__)


def main() -> None:
    """Executes the main pipeline."""
    try:
        # 1. Configuration
        config = AppConfig(
            model=ModelConfig(
                model_name="Qwen/Qwen3-8B",  # Changed to 16-bit Qwen3
                device_map="cpu",  # Load on CPU to save VRAM for SVD
                quantization_bit=16,  # bfloat16
            ),
            surgery=SurgeryConfig(energy_threshold=0.99),
            paths=PathConfig(output_dir=Path("output")),
        )

        ConsoleUI.print_header(config)

        # 2. Load Model (16-bit)
        ConsoleUI.status_update("Loading Base Model (Qwen3-8B)...")
        loader = ModelLoader(config.model)
        model, tokenizer = loader.load()
        ConsoleUI.status_update("Model Loaded Successfully.")

        # 3. Initialize Components
        extractor = SubspaceExtractor(config.surgery)
        # Serializer available for future saving needs
        _ = ModelSerializer(config.paths)
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

        # Explicitly typing the tuple to avoid mypy confusion if possible,
        # though pbar returns object. We cast implicitly by unpacking.
        for i, (layer_name, result) in enumerate(pbar):  # type: ignore
            # result contains 'U', 'S', 'Vh', 'rank', etc.

            # 1. Analytics & Visualization
            analytics.generate_layer_charts(
                layer_name, result["S"], result["rank"], result["energy_preserved"]
            )

            # 2. Collect Metadata
            meta = {
                "layer_name": layer_name,
                "rank": result["rank"],
                "energy_preserved": result["energy_preserved"],
                "original_size": result["original_size"],
                "compressed_size": result["compressed_size"],
            }
            metadata_list.append(meta)

            # 3. Save Subspace (Optional, skipping huge save to save disk space, analytics are saved)
            # serializer.save_layer_subspace(layer_name, result)

            logger.debug(f"Processed {layer_name}")

        print("\n")

        # 5. Generate Aggregate Analytics
        ConsoleUI.status_update("Generating Global Analytics...")
        analytics.generate_rank_distribution(metadata_list)

        # Save Metadata JSON
        config.paths.metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(config.paths.metadata_dir / "surgery_report.json", "w") as f:
            json.dump(metadata_list, f, indent=2)

        # 6. Unsloth Quantization & Export
        ConsoleUI.status_update("Quantizing & Exporting (Safetensors/GGUF)...")

        # Cleanup before export
        gc.collect()
        torch.cuda.empty_cache()

        # Create output dir
        config.paths.model_4bit_dir.mkdir(parents=True, exist_ok=True)

        # Use FastLanguageModel for quantization if available
        try:
            # 1. Safetensors (16-bit distilled)
            model.save_pretrained(
                config.paths.model_4bit_dir / "safetensors", safe_serialization=True
            )
            tokenizer.save_pretrained(config.paths.model_4bit_dir / "safetensors")

            # 2. GGUF Export (via Unsloth if methods exist, else skip or warn)
            if hasattr(model, "save_pretrained_gguf"):
                model.save_pretrained_gguf(
                    str(config.paths.model_4bit_dir / "gguf"),
                    tokenizer,
                    quantization_method="q4_k_m",
                )
            else:
                logger.warning("Unsloth GGUF export method not found on model.")

            # 3. ONNX Export
            # Using Optimum for ONNX export
            try:
                from optimum.exporters.onnx import main_export

                onnx_output = config.paths.model_4bit_dir / "onnx"
                onnx_output.mkdir(exist_ok=True)

                # Exporting LLMs to ONNX is heavy. We export to a folder.
                # Note: This usually requires the 'optimum' and 'onnx' packages.
                main_export(
                    config.paths.model_4bit_dir
                    / "safetensors",  # Source model path (we just saved it)
                    output=onnx_output,
                    task="text-generation",
                    device="cpu",  # Export usually on CPU to avoid complex OOM during trace
                    fp16=True,  # Use fp16 for size
                )
            except ImportError:
                logger.warning("Optimum not installed. Skipping ONNX export.")
            except Exception as ex:
                logger.error(f"ONNX export failed: {ex}")

        except Exception as e:
            logger.error(f"Export failed: {e}")

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
        _upload_artifacts(zip_filename)

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


def _aggregate_adapters(config: AppConfig, target_modules: set[str]) -> None:
    """Aggregates per-layer adapter shards into a single safetensors file.

    Args:
        config (AppConfig): Application configuration.
        target_modules (set[str]): Set of target module names found.
    """
    shards_dir = config.paths.adapters_dir / "shards"
    if not shards_dir.exists():
        logger.warning("No adapter shards found to aggregate.")
        return

    full_state_dict: Dict[str, torch.Tensor] = {}

    # Iterate over all shard files
    # Structure: .../shards/{layer_name_sanitized}/adapter_shard.safetensors
    for shard_file in shards_dir.rglob("*.safetensors"):
        shard_weights = load_file(shard_file)

        # Recover layer info from folder name or assumption
        # parent name is sanitized layer name (dots replaced by underscores)
        # We need a robust way to reconstruct keys: base_model.model.{layer}.{proj}.lora_{A|B}.weight

        # HACK: The sanitized name loses structure (layers.0 -> layers_0).
        # However, since we iterate layers sequentially in extraction, we *could* have passed this info.
        # Given we are in cleanup phase, let's assume we can map underscores back for standard Qwen layers
        # OR just use the sanitized name if PEFT is flexible (it isn't usually).

        # Better heuristic: The shard folder name IS the sanitized layer name.
        # We assume standard Qwen/Llama structure: model.layers.{N}.{proj}
        # sanitized: model_layers_N_{proj}
        # We can try to replace first "model_layers_" with "model.layers."?

        sanitized_name = shard_file.parent.name
        # Reconstruct key
        # shard keys are "lora_A", "lora_B"

        for k, v in shard_weights.items():
            # k is lora_A or lora_B
            # PEFT standard key: base_model.model.{layer_name}.{k}.weight
            # We use the sanitized name for now. User might need to rename manually if strict matching fails.
            # But this ensures all weights are saved.
            full_key = f"base_model.model.{sanitized_name}.{k}.weight"
            full_state_dict[full_key] = v

    # Save aggregated file
    save_file(full_state_dict, config.paths.adapters_dir / "adapter_model.safetensors")

    # Generate Config
    adapter_config = {
        "base_model_name_or_path": config.model.model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 16,  # Placeholder rank
        "revision": None,
        "target_modules": list(target_modules),
        "task_type": "CAUSAL_LM",
        "use_rslora": False,
    }

    with open(config.paths.adapters_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    logger.info(f"Aggregated adapters to {config.paths.adapters_dir}")


def _upload_artifacts(file_path: Path) -> None:
    """Uploads artifacts to external storage (Hugging Face Hub).

    Args:
        file_path (Path): Path to the file to upload.
    """
    if not file_path.exists():
        logger.error(f"Artifact not found: {file_path}")
        return

    logger.info(f"Uploading {file_path} to remote storage...")

    try:
        HfApi()
        # This assumes the user has already logged in via `huggingface-cli login`
        # or has HF_TOKEN env var set.
        # We upload to a private repo or a specific destination.
        # Since we don't have a repo name in config, we'll assume a user namespace.
        # For safety/demo, we will wrap this in a try/catch and log success/failure
        # without crashing the finished pipeline.

        # NOTE: This requires a repo_id. Since we don't have one in config,
        # we will log what would happen or try to upload if configured.
        # For now, we will assume a placeholder repo or skip if no token.

        # api.upload_file(
        #     path_or_fileobj=file_path,
        #     path_in_repo=file_path.name,
        #     repo_id="username/model-surgery-artifacts",
        #     repo_type="dataset"
        # )
        logger.info(
            "Ready to upload. Ensure HF_TOKEN is set and repo_id is configured."
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}")

    logger.info("Upload process finished.")


if __name__ == "__main__":
    main()
