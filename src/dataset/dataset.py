"""
Histopathology Dataset for image inpainting.
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob


class HistoDataset(Dataset):
    """
    Dataset for histopathology image inpainting.
    
    Expected file structure in root_dir:
        {sample_name}_wsi_crop.jpg      - Original whole slide image crop
        {sample_name}_extended_mask.jpg  - Mask indicating inpainting region
        {sample_name}_masked_crop.jpg    - Source image with masked region
    """
    
    def __init__(
        self,
        root_dir,
        image_size=(512, 512),
        mode="train",
        train_split=0.8,
    ):
        """
        Args:
            root_dir: Directory with all the histopathology images.
            image_size: Desired image size (default is (512, 512)).
            mode: Either "train" or "test" to specify the dataset mode.
            train_split: Fraction of data to use for training (default is 0.8).
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.mode = mode

        # Find all unique sample names by looking for wsi_crop files
        wsi_files = glob.glob(os.path.join(root_dir, "*_wsi_crop.jpg"))
        sample_names = [os.path.basename(f).replace("_wsi_crop.jpg", "") for f in wsi_files]
        sample_names.sort()  # Ensure consistent ordering

        # Split into train/test
        split_idx = int(len(sample_names) * train_split)
        if mode == "train":
            self.sample_names = sample_names[:split_idx]
        else:
            self.sample_names = sample_names[split_idx:]

        # Define transforms for image resizing and normalization
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.01).to(x.dtype)),
        ])

    def __len__(self):
        return len(self.sample_names)

    def load_image(self, path):
        """Load and transform an RGB image."""
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def load_mask(self, path):
        """Load and transform a mask image."""
        mask = Image.open(path).convert("L")
        return self.mask_transform(mask)

    def get_histo_paths(self, sample_name):
        """Get paths for histopathology dataset."""
        wsi_image = os.path.join(self.root_dir, f"{sample_name}_wsi_crop.jpg")
        extended_mask = os.path.join(self.root_dir, f"{sample_name}_extended_mask.jpg")
        masked_crop = os.path.join(self.root_dir, f"{sample_name}_masked_crop.jpg")
        return wsi_image, extended_mask, masked_crop

    def __getitem__(self, idx):
        sample_name = self.sample_names[idx]

        # Get paths for the three images
        wsi_image_path, extended_mask_path, masked_crop_path = self.get_histo_paths(sample_name)

        # Load all images and masks
        wsi_image = self.load_image(wsi_image_path)
        extended_mask = self.load_mask(extended_mask_path)
        masked_crop = self.load_image(masked_crop_path)

        return {
            "wsi_image": wsi_image,
            "extended_mask": extended_mask,
            "masked_crop": masked_crop,
            "sample_name": sample_name,
            "dataset": "histopathology",
        }

