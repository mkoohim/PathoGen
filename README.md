# 🔬 PathoGen - Histopathology Image Inpainting

PathoGen is a diffusion-based model for histopathology image inpainting. It enables training and inference of models that can fill in masked regions of pathology images with realistic tissue patterns.

<img width="1385" height="523" alt="fig1" src="https://github.com/user-attachments/assets/0dbd7246-5636-4a60-988b-bbe844bbd7bd" />

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/PathoGen.git
cd PathoGen
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download base model weights

Follow the instructions in `src/models/base_model/README.md` to download the required UNet weights from Hugging Face.

```bash
# Quick download using Python
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='booksforcharlie/stable-diffusion-inpainting',
    filename='unet/diffusion_pytorch_model.bin',
    local_dir='src/models/base_model'
)
"
```

## Dataset Preparation

PathoGen expects your dataset to be organized with three files per sample:

```
data/
├── sample001_wsi_crop.jpg      # Original whole slide image crop
├── sample001_extended_mask.jpg  # Binary mask (white = region to inpaint)
├── sample001_masked_crop.jpg    # Source image with pattern to transfer
├── sample002_wsi_crop.jpg
├── sample002_extended_mask.jpg
├── sample002_masked_crop.jpg
└── ...
```

### File naming convention:
- `{sample_name}_wsi_crop.jpg` - The target image to be inpainted
- `{sample_name}_extended_mask.jpg` - Grayscale mask indicating inpainting region
- `{sample_name}_masked_crop.jpg` - Source image providing the fill pattern

## Training

### Basic training

```bash
cd PathoGen
python train.py
```

### Configuration

Edit `configs/config.yaml` to customize:

```yaml
dataset:
  data_dir: "/path/to/your/data"
  batch_size: 2
  image_size: [512, 512]

training:
  learning_rate: 1e-5

trainer_params:
  max_epochs: 100
  devices: 1  # Number of GPUs
```

### Multi-GPU training

```bash
# Edit config.yaml:
# trainer_params:
#   devices: 4
#   strategy: "ddp"

python train.py
```

Checkpoints will be saved to `./checkpoints/` by default.

## Inference

### Run inference on test samples

```bash
python inference.py \
    --config_path configs/config.yaml \
    --test_sample_dir ./test_samples \
    --output_dir ./outputs \
    --checkpoint_epoch 99 \
    --gpu_id 0
```

## ⚙️ Configuration

The main configuration file is `configs/config.yaml`:

```yaml
# Paths
code_base_path: "."
artifacts_path: "./checkpoints"
dataset_path: "./data"

# Model settings
model:
  base_model_path: ${code_base_path}/src/models/base_model
  num_inference_steps: 100
  num_train_timesteps: 1000

# Training settings
training:
  learning_rate: 1e-5
  use_dream_in_training: False

# Dataset settings
dataset:
  batch_size: 2
  image_size: [512, 512]
  train_split: 0.8
```

## Project Structure

```
PathoGen/
├── configs/
│   └── config.yaml           # Main configuration
├── train.py                  # Training script
├── inference.py              # Inference script
├── src/
│   ├── dataset/
│   │   ├── datamodule.py     # PyTorch Lightning DataModule
│   │   └── dataset.py        # Dataset class
│   ├── models/
│   │   ├── base_model/       # Base diffusion model configs
│   │   ├── attn_processor.py # Attention processors
│   │   └── pathogen.py       # Main model (PathoGenModel)
│   └── tools/
│       └── utils.py          # Utility functions
├── checkpoints/              # Saved model weights
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Pretrained Weights

The pretrained attention weights can be found in our [HuggingFace](https://huggingface.co/mkoohim/PathoGen) page.

## Citation

If you use PathoGen in your research, please cite:

```bibtex
@misc{pathogen2025,
  title={PathoGen: Diffusion-Based Synthesis of Realistic Lesions in Histopathology Images},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License.
