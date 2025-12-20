"""Export tools for GGUF conversion and quantization.

This module handles the creation of custom GGUF files and the execution
of binary quantization tools.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import gguf
import torch
from safetensors.torch import load_file

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def get_qwen_tensor_name(name: str) -> str:
    """Maps PyTorch tensor name to GGUF tensor name for Qwen2 architecture.

    Args:
        name: PyTorch tensor name.

    Returns:
        GGUF tensor name.
    """
    # Base mappings
    if "embed_tokens" in name:
        return "token_embd"
    if "lm_head" in name:
        return "output"
    if "model.norm" in name:
        return "output_norm"

    # Layer mappings
    if "layers" in name:
        parts = name.split(".")
        # Expected format: model.layers.N.submodule...
        try:
            # Find the index after 'layers'
            idx_loc = parts.index("layers") + 1
            layer_idx = parts[idx_loc]
        except (ValueError, IndexError):
            return name  # Fallback

        prefix = f"blk.{layer_idx}"

        if "input_layernorm" in name:
            return f"{prefix}.attn_norm"
        if "post_attention_layernorm" in name:
            return f"{prefix}.ffn_norm"

        if "self_attn" in name:
            if "q_proj" in name:
                return f"{prefix}.attn_q"
            if "k_proj" in name:
                return f"{prefix}.attn_k"
            if "v_proj" in name:
                return f"{prefix}.attn_v"
            if "o_proj" in name:
                return f"{prefix}.attn_output"

        if "mlp" in name:
            if "gate_proj" in name:
                return f"{prefix}.ffn_gate"
            if "up_proj" in name:
                return f"{prefix}.ffn_up"
            if "down_proj" in name:
                return f"{prefix}.ffn_down"

    return name


def convert_to_custom_gguf(
    safetensors_path: Path, output_path: Path, config: Any
) -> None:
    """Converts a decomposed safetensors model to a custom GGUF file.

    Args:
        safetensors_path: Path to the input safetensors file.
        output_path: Path to save the GGUF file.
        config: Model configuration object (transformers config).
    """
    logger.info(f"Loading tensors from {safetensors_path}...")
    state_dict = load_file(safetensors_path)

    gguf_writer = gguf.GGUFWriter(output_path, "qwen2")

    # Write Headers
    # Qwen3-4B-Base standard headers
    logger.info("Writing GGUF headers...")

    # Architecture
    gguf_writer.add_architecture()  # Adds 'qwen2'
    gguf_writer.add_name("Qwen3-4B-Base")

    # Dimensions
    # Using config values if available, else strict defaults for Qwen3-4B
    gguf_writer.add_block_count(getattr(config, "num_hidden_layers", 36))
    gguf_writer.add_context_length(getattr(config, "max_position_embeddings", 32768))
    gguf_writer.add_embedding_length(getattr(config, "hidden_size", 0))
    gguf_writer.add_feed_forward_length(getattr(config, "intermediate_size", 0))
    gguf_writer.add_head_count(getattr(config, "num_attention_heads", 32))
    gguf_writer.add_head_count_kv(getattr(config, "num_key_value_heads", 8))

    # RoPE
    # freq_base defaults to 1000000.0 for Qwen2.5/3 usually
    gguf_writer.add_rope_dimension_count(
        getattr(config, "hidden_size", 0) // getattr(config, "num_attention_heads", 1)
    )
    gguf_writer.add_rope_freq_base(getattr(config, "rope_theta", 1000000.0))

    # Layer Norm
    gguf_writer.add_layer_norm_rms_eps(getattr(config, "rms_norm_eps", 1e-6))

    # Tensors
    logger.info("Writing tensors...")

    for key, tensor in state_dict.items():
        # Determine tensor name
        # Key examples: "model.layers.0.self_attn.q_proj.weight", "...q_proj.u", "...q_proj.v"

        # 1. Map base name
        base_name = key
        suffix = ""

        if key.endswith(".weight"):
            base_name = key[:-7]  # remove .weight
            suffix = ".weight"
        elif key.endswith(".bias"):
            base_name = key[:-5]
            suffix = ".bias"
        elif key.endswith(".u"):
            base_name = key[:-2]
            suffix = ".u"
        elif key.endswith(".v"):
            base_name = key[:-2]
            suffix = ".v"

        gguf_name_base = get_qwen_tensor_name(base_name)
        gguf_name = f"{gguf_name_base}{suffix}"

        # 2. Convert to FP16 (if not already) and numpy
        # GGUF writer expects numpy arrays
        # Use float16 to match user requirement (FP16 structure)
        if tensor.dtype != torch.float16:
            tensor = tensor.to(dtype=torch.float16)
        data = tensor.numpy()

        # 3. Add to GGUF
        gguf_writer.add_tensor(gguf_name, data)

    logger.info("Saving GGUF file...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    logger.info(f"GGUF saved to {output_path}")


def quantize_model(gguf_path: Path, final_path: Path) -> None:
    """Builds llama.cpp and runs quantization.

    Args:
        gguf_path: Path to the input GGUF file (FP16).
        final_path: Path to the output quantized file.
    """
    logger.info("Starting quantization process...")

    # Working in /tmp to save space/performance
    # We use a known subdirectory to avoid random temp names making debugging hard if it fails
    # But mkdtemp is safer for concurrency.
    with tempfile.TemporaryDirectory(dir="/tmp") as build_dir:
        build_path = Path(build_dir)
        repo_path = build_path / "llama.cpp"

        # 1. Clone llama.cpp
        logger.info("Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp", str(repo_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        # 2. Build
        logger.info("Building llama.cpp...")
        build_bin_path = repo_path / "build"
        build_bin_path.mkdir()

        # Simple build
        subprocess.run(
            [
                "cmake",
                "-B",
                str(build_bin_path),
                "-S",
                str(repo_path),
                "-DGROK_BUILD=OFF",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["cmake", "--build", str(build_bin_path), "--config", "Release", "-j4"],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        quantize_bin = build_bin_path / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            # Try fallback location
            quantize_bin = build_bin_path / "llama-quantize"

        if not quantize_bin.exists():
            raise FileNotFoundError(
                f"Could not find llama-quantize executable at {quantize_bin}"
            )

        # 3. Run Quantize
        logger.info(f"Running quantization on {gguf_path}...")
        subprocess.run(
            [str(quantize_bin), str(gguf_path), str(final_path), "Q4_K_M"], check=True
        )

    logger.info(f"Quantization complete. Saved to {final_path}")
