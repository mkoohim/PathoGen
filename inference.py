"""
PathoGen Inference Script

Run inference on histopathology test samples using a trained model.

Usage:
    python inference.py --config_path configs/config.yaml --test_sample_dir ./test_samples --output_dir ./outputs

Options:
    --config_path: Path to configuration file (default: configs/config.yaml)
    --test_sample_dir: Directory containing test samples
    --output_dir: Directory to save output images
    --checkpoint_epoch: Epoch number of the checkpoint to use (default: 99)
    --sample_name: Specific sample name to process (optional)
    --gpu_id: GPU ID to use for inference (default: 0, use -1 for CPU)
"""
import click
import sys
import os

# Add the current directory to the path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf
import glob
import numpy as np
from PIL import Image
from src.models.pathogen import PathoGenModel
import matplotlib.pyplot as plt
from pytorch_lightning.utilities.model_summary import ModelSummary
import torch


@click.command()
@click.option(
    "--config_path",
    default="configs/config.yaml",
    help="Configuration file (default: configs/config.yaml)",
)
@click.option(
    "--test_sample_dir",
    default="test_samples",
    help="Directory containing test samples"
)
@click.option(
    "--output_dir",
    default="outputs",
    help="Directory to save output images"
)
@click.option(
    "--checkpoint_epoch",
    default=99,
    help="Epoch number of the checkpoint to use (default: 99)"
)
@click.option(
    "--sample_name",
    default=None,
    help="Specific sample name to process (if not provided, processes all samples)"
)
@click.option(
    "--gpu_id",
    default=0,
    help="GPU ID to use for inference (default: 0, use -1 for CPU)"
)
def inference(
    config_path: str,
    test_sample_dir: str,
    output_dir: str,
    checkpoint_epoch: int,
    sample_name: str,
    gpu_id: int,
):
    """Run inference on histopathology test samples using trained model."""
    # Load configuration
    config = OmegaConf.load(config_path)
    config = dict(config)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Update checkpoint path to use the specified epoch
    checkpoint_path = os.path.join(
        config["artifacts_path"],
        f"histo_attention_modules_ep_{checkpoint_epoch}.pt"
    )

    print(f"Loading model with checkpoint: {checkpoint_path}")

    # Initialize model
    model = PathoGenModel(config)

    # Set device based on user preference
    if gpu_id == -1:
        device = torch.device("cpu")
    elif torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
        props = torch.cuda.get_device_properties(gpu_id)
        memory_total = props.total_memory / 1024**3
        print(f"GPU {gpu_id} ({props.name}): {memory_total:.1f}GB total")
    else:
        print(f"Warning: GPU {gpu_id} not available. Falling back to CPU.")
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Move model to the correct device
    model = model.to(device)

    # Load checkpoint with proper device handling
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.attn_modules.load_state_dict(checkpoint)

    # Ensure model is in eval mode
    model.eval()

    # Convert all model components to consistent data types
    model.vae = model.vae.to(device=device, dtype=model.weight_dtype)
    model.unet = model.unet.to(device=device, dtype=model.weight_dtype)
    model.attn_modules = model.attn_modules.to(device=device, dtype=model.weight_dtype)

    print(f"Model device: {device}, weight_dtype: {model.weight_dtype}")
    print(ModelSummary(model))

    # Find test samples
    if sample_name:
        sample_names = [sample_name]
    else:
        wsi_files = glob.glob(os.path.join(test_sample_dir, "*_wsi_crop.jpg"))
        sample_names = [os.path.basename(f).replace("_wsi_crop.jpg", "") for f in wsi_files]
        sample_names.sort()

    print(f"Found {len(sample_names)} samples to process")

    # Process each sample
    for i, name in enumerate(sample_names):
        print(f"\nProcessing sample {i+1}/{len(sample_names)}: {name}")
        
        wsi_path = os.path.join(test_sample_dir, f"{name}_wsi_crop.jpg")
        mask_path = os.path.join(test_sample_dir, f"{name}_extended_mask.jpg")
        crop_path = os.path.join(test_sample_dir, f"{name}_masked_crop.jpg")

        # Check if all required files exist
        if not all(os.path.exists(p) for p in [wsi_path, mask_path, crop_path]):
            print(f"  Skipping {name}: Missing required files")
            continue

        try:
            # Load images
            wsi_image = Image.open(wsi_path).convert("RGB")
            extended_mask = Image.open(mask_path).convert("L")
            masked_crop = Image.open(crop_path).convert("RGB")

            print(f"  Images loaded - WSI: {wsi_image.size}")

            # Run inference
            with torch.no_grad():
                model.eval()
                result_image = model.forward(wsi_image, extended_mask, masked_crop)

                # Convert float16 to float32 for matplotlib compatibility
                if result_image.dtype == np.float16:
                    result_image = result_image.astype(np.float32)

            # Create visualization
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(wsi_image)
            axes[0].set_title("WSI Image")
            axes[0].axis("off")

            axes[1].imshow(extended_mask, cmap='gray')
            axes[1].set_title("Extended Mask")
            axes[1].axis("off")

            axes[2].imshow(masked_crop)
            axes[2].set_title("Masked Crop")
            axes[2].axis("off")

            axes[3].imshow(result_image)
            axes[3].set_title("Result")
            axes[3].axis("off")

            fig.tight_layout()

            # Save visualization
            output_path = os.path.join(output_dir, f"{name}_result.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Save individual result image
            result_only_path = os.path.join(output_dir, f"{name}_result_only.png")
            result_image_np = np.clip(result_image, 0, 1)
            result_image_uint8 = (result_image_np * 255).astype(np.uint8)
            result_image_pil = Image.fromarray(result_image_uint8)
            result_image_pil.save(result_only_path)

            print(f"  Saved results to {output_path}")

        except Exception as e:
            print(f"  Error processing {name}: {str(e)}")
            continue

    print(f"\nInference completed! Results saved to {output_dir}")


if __name__ == "__main__":
    inference()

