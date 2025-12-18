"""Main entry point for the AI Subspace Extraction pipeline.

This module orchestrates the loading, surgery, and saving processes.
"""

import sys
import zipfile
from pathlib import Path

from src.config import AppConfig, ModelConfig, PathConfig, SurgeryConfig
from src.surgery.core import SubspaceExtractor
from src.surgery.loader import ModelLoader
from huggingface_hub import HfApi

from src.surgery.serializer import ModelSerializer
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    """Executes the main pipeline."""
    try:
        # 1. Configuration
        # In a real app, these might come from CLI args or env vars
        config = AppConfig(
            model=ModelConfig(
                model_name="unsloth/Qwen3-8B-unsloth-bnb-4bit",
                device_map="auto",
                quantization_bit=4,
            ),
            surgery=SurgeryConfig(truncation_ratio=0.2),
            paths=PathConfig(output_dir=Path("output")),
        )

        logger.info("Starting AI Subspace Extraction Pipeline")

        # 2. Load Model
        loader = ModelLoader(config.model)
        model, tokenizer = loader.load()

        # 3. Initialize Components
        extractor = SubspaceExtractor(config.surgery)
        serializer = ModelSerializer(config.paths)

        # 4. Surgery Loop
        logger.info("Beginning surgery...")
        for layer_name, subspace in extractor.extract(model):
            serializer.save_layer_subspace(layer_name, subspace)

        # 5. Save Model Configuration (Base Model)
        logger.info("Saving base model configuration...")
        config.paths.base_model_dir.mkdir(parents=True, exist_ok=True)
        model.config.save_pretrained(config.paths.base_model_dir)
        tokenizer.save_pretrained(config.paths.base_model_dir)

        # 6. Create Adapters Directory (Structure compliance)
        config.paths.adapters_dir.mkdir(parents=True, exist_ok=True)

        # 7. Antifragile Export (Compression)
        logger.info("Surgery complete. Compressing artifacts...")
        archive_path = config.paths.backup_dir
        archive_path.mkdir(parents=True, exist_ok=True)

        zip_filename = archive_path / "subspace_extraction.zip"
        target_folders = [
            config.paths.base_model_dir,
            config.paths.extraction_dir,
            config.paths.adapters_dir,
        ]

        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder in target_folders:
                if folder.exists():
                    # Preserve structure: /base_model/config.json -> base_model/config.json
                    # We walk relative to output_dir so base_model is at root of zip
                    for file in folder.rglob("*"):
                        if file.is_file():
                            # Ensure we don't zip the zip file itself if paths overlap (unlikely here)
                            if file == zip_filename:
                                continue

                            arcname = file.relative_to(config.paths.output_dir)
                            zf.write(file, arcname=arcname)

        logger.info(f"Artifacts compressed to {zip_filename}")

        # 8. Dummy Upload Hook
        _upload_artifacts(zip_filename)

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


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
