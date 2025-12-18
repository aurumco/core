# AI Subspace Extraction Pipeline

Extracts the "AI Core" (Subspace) from 4-bit Quantized Language Models using SVD.

[![CI](https://img.shields.io/github/actions/workflow/status/USER/REPO/ci.yml?style=flat&logo=github&label=CI)](link)
[![Coverage](https://img.shields.io/codecov/c/github/USER/REPO?style=flat&logo=codecov&label=Coverage)](link)
[![License](https://img.shields.io/github/license/USER/REPO?style=flat&label=License)](link)
[![Version](https://img.shields.io/github/v/release/USER/REPO?style=flat&label=Version)](link)

This project performs "Model Surgery" on LLMs (like Qwen) by dequantizing layers, decomposing them via Singular Value Decomposition (SVD), and truncating singular values to retain only the essential information (subspace), effectively compressing the model's knowledge into a low-rank format.

---

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.main import main

# Run the full pipeline
if __name__ == "__main__":
    main()
```

## Features

- **Quantized Loading**: Efficiently loads 4-bit models using `unsloth` or `bitsandbytes`.
- **Layer-wise Surgery**: Processes layers sequentially to respect VRAM limits.
- **Subspace Extraction**: Uses SVD to isolate and truncate essential features.
- **Antifragile Export**: Automatically compresses and prepares artifacts for upload.

## Documentation

See [docs/](docs/) for detailed guides and API reference (Coming soon).

## License

Licensed under MIT - see [LICENSE](LICENSE) file.
