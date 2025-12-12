"""
PyTorch Lightning DataModule for histopathology dataset.
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.dataset.dataset import HistoDataset
import multiprocessing


class HistoDataModule(pl.LightningDataModule):
    """DataModule for histopathology image inpainting dataset."""
    
    def __init__(
        self,
        data_dir,
        batch_size=8,
        image_size=(512, 512),
        train_split=0.8,
        num_workers=None,
        pin_memory=True,
        persistent_workers=True
    ):
        """
        Initialize the data module.
        
        Args:
            data_dir: Directory containing the dataset
            batch_size: Batch size for training/testing
            image_size: Target image size (height, width)
            train_split: Fraction of data for training
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_split = train_split
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        # Optimize num_workers for multi-GPU setup
        if num_workers is None:
            self.num_workers = min(multiprocessing.cpu_count(), 8)
        else:
            self.num_workers = num_workers

    def setup(self, stage=None):
        """Set up train and test datasets."""
        self.train_dataset = HistoDataset(
            root_dir=self.data_dir,
            image_size=self.image_size,
            mode="train",
            train_split=self.train_split,
        )
        self.test_dataset = HistoDataset(
            root_dir=self.data_dir,
            image_size=self.image_size,
            mode="test",
            train_split=self.train_split,
        )
        
    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

