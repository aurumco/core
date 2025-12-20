# AI Subspace Extraction Pipeline
Extracts the Core (Subspace) from Language Models using ASVD.

<p align="center">

[![Version](https://img.shields.io/badge/Version-0.1.0--beta-000000?style=flat&colorA=000000&colorB=000000&label=Version)](https://github.com/aurumco/core/releases)
[![License](https://img.shields.io/badge/License-AGPL--3.0-000000?style=flat&colorA=000000&colorB=000000&label=License)](https://www.gnu.org/licenses/agpl-3.0.en.html)
[![Security](https://img.shields.io/badge/Security-Audited-000000?style=flat&colorA=000000&colorB=000000&logo=osv&label=Security)](https://github.com/aurumco/core/security)
[![Activity](https://img.shields.io/github/commit-activity/m/aurumco/core?style=flat&colorA=000000&colorB=000000&label=Velocity)](https://github.com/aurumco/core/graphs/commit-activity)

</p>
  


This project performs "Model Surgery" on LLMs (like Qwen, Gemma, Llama) by dequantizing layers, decomposing them via Singular Value Decomposition (SVD), and truncating singular values to retain only the essential information (subspace), effectively compressing the model's knowledge into a low-rank format.

---

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.main import main

if __name__ == "__main__":
    main()
```

## Features

- **Quantized Loading**: Efficiently loads 4-bit models using `unsloth` or `bitsandbytes`.
- **Layer-wise Surgery**: Processes layers sequentially to respect VRAM limits.
- **Subspace Extraction**: Uses SVD to isolate and truncate essential features.
- **Antifragile Export**: Automatically compresses and prepares artifacts for upload.


## License

Licensed under AGPL-3.0.
