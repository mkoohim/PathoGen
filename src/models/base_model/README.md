# Base Model Setup

This directory contains the configuration for the base diffusion model used in PathoGen.

## Required Files

You need to download the UNet weights for the stable diffusion inpainting model:

### Option 1: Download from Hugging Face (Recommended)

Download the `diffusion_pytorch_model.bin` file from:
https://huggingface.co/booksforcharlie/stable-diffusion-inpainting/tree/main

Place it in the `unet/` subfolder:
```
src/models/base_model/
├── config.json
├── scheduler/
│   └── scheduler_config.json
├── unet/
│   └── diffusion_pytorch_model.bin  ← Download this file
└── README.md
```

### Option 2: Using Python

```python
from huggingface_hub import hf_hub_download

# Download the UNet weights
hf_hub_download(
    repo_id="booksforcharlie/stable-diffusion-inpainting",
    filename="unet/diffusion_pytorch_model.bin",
    local_dir="src/models/base_model"
)
```

### Option 3: Using wget

```bash
mkdir -p src/models/base_model/unet
wget -O src/models/base_model/unet/diffusion_pytorch_model.bin \
    "https://huggingface.co/booksforcharlie/stable-diffusion-inpainting/resolve/main/unet/diffusion_pytorch_model.bin"
```

## File Structure

After setup, the directory should contain:
- `config.json` - UNet model configuration
- `scheduler/scheduler_config.json` - DDIM scheduler configuration
- `unet/diffusion_pytorch_model.bin` - Pre-trained UNet weights (~3.5GB)

