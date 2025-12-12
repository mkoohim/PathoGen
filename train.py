"""
PathoGen Training Script

Train a diffusion model for histopathology image inpainting.

Usage:
    python train.py

The training configuration is defined in configs/config.yaml
"""
import sys
import os

# Add the current directory to the path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from src.dataset.datamodule import HistoDataModule
from src.models.pathogen import PathoGenModel
import torch


class AttenSaverCallback(pl.Callback):
    """Callback to save attention module checkpoints at the end of each epoch."""
    
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        
    def on_train_epoch_end(self, trainer, pl_module):
        path = self.checkpoint_path.format(epoch=trainer.current_epoch)
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            pl_module.attn_modules.state_dict(),
            path
        )
        print(f"Saved attention checkpoint to {path}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    """Main training function."""
    # Prepare dataset
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
    config = dict(cfg)
    
    # Initialize data module
    data_module = HistoDataModule(
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["dataset"]["batch_size"],
        image_size=tuple(config["dataset"]["image_size"]),
        train_split=config["dataset"].get("train_split", 0.8),
        num_workers=config["dataset"].get("num_workers", None),
        pin_memory=config["dataset"].get("pin_memory", True),
        persistent_workers=config["dataset"].get("persistent_workers", True),
    )

    # Initialize model
    model = PathoGenModel(cfg=config)
    
    # Setup checkpoint callback
    saver_callback = AttenSaverCallback(
        checkpoint_path=config["model"]["save_folder_attn_ckpt"]
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        **config["trainer_params"], 
        callbacks=[saver_callback], 
        gradient_clip_val=1.0
    )

    # Train the model
    trainer.fit(model, data_module)
    
    print("Training completed!")


if __name__ == "__main__":
    train()

