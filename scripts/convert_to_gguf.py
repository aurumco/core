"""Script to convert a pre-existing Safetensors model to GGUF.

This script skips the SVD decomposition and runs the export pipeline
directly on an existing 'model_svd.safetensors' file.

Usage:
    python scripts/convert_to_gguf.py --input output/model_4bit/safetensors/model_svd.safetensors --output output/model_4bit/gguf/model_fp16.gguf --config_dir output/model_4bit/safetensors
"""

import argparse
import sys
from pathlib import Path

from transformers import AutoConfig

# Add src to python path if running from root
sys.path.append(str(Path.cwd()))

from src.surgery.export_tools import convert_to_custom_gguf, quantize_model
from src.utils.logging import setup_logger
from src.utils.ui import ConsoleUI

logger = setup_logger(__name__)


def main() -> None:
    """Main entry point for standalone conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Safetensors model to Custom GGUF."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input safetensors file (e.g., model_svd.safetensors)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output GGUF file (e.g., model_fp16.gguf)",
    )
    parser.add_argument(
        "--config_dir",
        type=Path,
        required=True,
        help="Directory containing config.json and tokenizer files",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Run quantization (Q4_K_M) after conversion",
    )

    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output
    config_dir: Path = args.config_dir

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    if not config_dir.exists():
        logger.error(f"Config directory not found: {config_dir}")
        sys.exit(1)

    try:
        ConsoleUI.status_update("Loading Configuration...")
        config = AutoConfig.from_pretrained(config_dir)

        ConsoleUI.status_update(f"Converting {input_path} to GGUF...")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Call conversion tool
        convert_to_custom_gguf(
            output_path=output_path, config=config, safetensors_path=input_path
        )

        if args.quantize:
            final_quantized_path = output_path.parent / "model_q4_k_m.gguf"
            ConsoleUI.status_update("Running Quantization...")
            quantize_model(output_path, final_quantized_path)
            logger.info(f"Quantization complete: {final_quantized_path}")

        logger.info("Conversion Pipeline Complete.")

    except Exception as e:
        logger.critical(f"Conversion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
