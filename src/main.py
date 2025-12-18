"""Main entry point for the AI Subspace Extraction pipeline.

This module orchestrates the loading, surgery, and saving processes.
"""

import shutil
import sys
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
                model_name="Qwen/Qwen1.5-7B-Chat-GPTQ-Int4",  # Example 4-bit model
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

        # We want to zip the content of output_dir excluding backup_dir
        # Since shutil.make_archive zips a root_dir, we need to be careful not to zip the zip itself if it's inside.
        # Here we zip the entire output_dir structure.

        # Create a temporary directory or zip carefully.
        # Simpler approach: Zip the output directory content.
        # But wait, config.paths.output_dir contains "backup" folder which we don't want to include recursively.
        # We should zip specifically the 3 folders: base_model, extracted_subspace, adapters.

        # However, making one archive of the parent `output_dir` is cleaner if we move backup outside or exclude it.
        # Let's adjust: The backup goes to `output_dir/backup`.
        # We will create a zip that contains the folders.

        # Better approach: Create a temporary staging dir for the zip content
        staging_dir = config.paths.output_dir / "staging_package"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)

        # Move/Copy logic or just zip the directories we created
        # Actually, `shutil.make_archive` with `root_dir` and `base_dir` is the way.
        # We want the zip to contain:
        # /base_model
        # /extracted_subspace
        # /adapters

        # We can pass root_dir=config.paths.output_dir, but we need to exclude "backup".
        # It's easier to zip the output_dir and ignore the fact that backup is inside initially,
        # OR put backup outside of output_dir.
        # But PathConfig defines backup_dir inside output_dir.

        # Let's just iterate and zip the specific folders.
        # OR use python zipfile module for control.
        # For simplicity in this script, we'll use shutil but accept that we might need to structure it differently.
        # Let's change backup_dir location in the archive step? No, PathConfig is fixed.

        # We will use zipfile to create the archive manually to exclude "backup"
        import zipfile

        zip_filename = archive_path / "subspace_extraction.zip"
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder in [
                config.paths.base_model_dir,
                config.paths.extraction_dir,
                config.paths.adapters_dir,
            ]:
                if folder.exists():
                    for file in folder.rglob("*"):
                        if file.is_file():
                            zf.write(
                                file, arcname=file.relative_to(config.paths.output_dir)
                            )

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
