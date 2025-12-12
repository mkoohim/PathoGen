"""
Full CATVTON Model for histopathology image inpainting.

This model uses a diffusion-based approach for image inpainting,
adapted for histopathology images.
"""
from typing import Any, Dict, Union
import pytorch_lightning as pl
import inspect
import PIL
import torch
import tqdm
from accelerate import load_checkpoint_in_model
import os
import logging
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor

from src.models.attn_processor import SkipAttnProcessor
from src.tools.utils import (
    compute_dream_and_update_latents_for_inpaint,
    get_trainable_module,
    init_adapter,
    compute_vae_encodings,
    prepare_image,
    prepare_mask_image,
    resize_and_crop,
    resize_and_padding,
)


class PathoGenModel(pl.LightningModule):
    """
    PathoGen Model for histopathology image inpainting.

    Uses a UNet-based diffusion model with VAE encoding/decoding
    and custom attention processors for inpainting.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters("cfg")

        # Override device detection for inference outside PyTorch Lightning trainer
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_model_path = cfg["model"]["base_model_path"]
        self.eta = cfg["model"]["eta"]
        self.num_train_timesteps = cfg["model"]["num_train_timesteps"]
        self.lr = cfg["training"]["learning_rate"]
        self.height = cfg["dataset"]["image_size"][0]
        self.width = cfg["dataset"]["image_size"][1]
        self.num_inference_steps = cfg["model"]["num_inference_steps"]

        self.weight_dtype = torch.float16  # Use float16 for mixed precision

        # Initialize scheduler
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            base_model_path, subfolder="scheduler"
        )
        
        # Initialize VAE
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        
        # Get training config
        self.use_dream_in_training = cfg["training"]["use_dream_in_training"]
        
        # Initialize UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            base_model_path, subfolder="unet"
        )
        
        # Initialize attention adapter with Skip Cross-Attention
        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)
        self.attn_modules = get_trainable_module(self.unet, "attention")

        # Freeze non-attention parameters
        self.set_no_grad()

        # Optional: Compile model for faster training (PyTorch 2.0+)
        self.compile_model = cfg.get("training", {}).get("compile_model", False)
        if self.compile_model:
            try:
                self.unet = torch.compile(self.unet, mode="reduce-overhead")
                print("Model compiled successfully for faster training")
            except Exception as e:
                print(f"Model compilation failed: {e}. Continuing without compilation.")

    @property
    def device(self):
        """Override device property for inference outside trainer."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self._device

    def set_no_grad(self):
        """Freeze all parameters except attention modules."""
        for name, param in self.unet.named_parameters():
            if "attn1" in name:
                continue
            param.requires_grad = False
        for name, param in self.vae.named_parameters():
            param.requires_grad = False

    def auto_attn_ckpt_load(self, attn_ckpt_folder_path):
        """Automatically load attention checkpoint if available."""
        if os.path.exists(attn_ckpt_folder_path):
            load_checkpoint_in_model(self.attn_modules, attn_ckpt_folder_path)
        else:
            logging.warning(f"Could not find attention checkpoint in {attn_ckpt_folder_path}")

    def prepare_extra_step_kwargs(self, generator, eta):
        """Prepare extra kwargs for the scheduler step."""
        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_unet_input_latent(
        self,
        image,
        mask,
        condition_image,
        classifier_free_guidance=False,
        mask_main_image=True,
    ):
        """Prepare latent inputs for the UNet."""
        concat_dim = -2  # Y axis concat
        
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        mask = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)

        masked_image = image * (mask < 0.5) if mask_main_image else image
        
        # VAE encoding
        masked_image_latent = compute_vae_encodings(masked_image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae)
        mask_latent = torch.nn.functional.interpolate(
            mask, size=masked_image_latent.shape[-2:], mode="nearest"
        )

        # Concatenate latents
        masked_image_condition_latent_concat = torch.cat(
            [masked_image_latent, condition_latent], dim=concat_dim
        )
        mask_latent_concat = torch.cat(
            [mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim
        )

        if classifier_free_guidance:
            masked_image_condition_latent_concat = torch.cat([
                torch.cat([masked_image_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                masked_image_condition_latent_concat,
            ])
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        return masked_image_condition_latent_concat, mask_latent_concat

    def training_step(self, batch, batch_idx):
        """Execute a single training step."""
        image = batch["wsi_image"]
        condition_image = batch["masked_crop"]
        person_mask = batch["extended_mask"]

        images_latent_concat, mask_latent_concat = self.prepare_unet_input_latent(
            image, person_mask, condition_image,
            classifier_free_guidance=False,
            mask_main_image=True,
        )
        gt_images_latent_concat, _ = self.prepare_unet_input_latent(
            image, person_mask, condition_image,
            classifier_free_guidance=False,
            mask_main_image=False,
        )

        # Sample noise
        noise = torch.randn_like(gt_images_latent_concat)
        bsz = gt_images_latent_concat.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=gt_images_latent_concat.device,
        )
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(
            gt_images_latent_concat, noise, timesteps
        )

        # Prepare inpainting model input
        latent_model_input = torch.cat([
            noisy_latents, mask_latent_concat, images_latent_concat,
        ], dim=1)

        if self.use_dream_in_training:
            latent_model_input, noise = compute_dream_and_update_latents_for_inpaint(
                self.unet, self.noise_scheduler, timesteps=timesteps,
                noise=noise, noisy_latents=latent_model_input,
                target=gt_images_latent_concat, encoder_hidden_states=None,
            )

        # Predict noise
        noise_pred = self.unet(
            latent_model_input, timesteps,
            encoder_hidden_states=None, return_dict=False,
        )[0]

        # Get target based on prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                gt_images_latent_concat, noise, timesteps
            )
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def check_inputs(self, image, condition_image, mask, width, height):
        """Check and preprocess inputs."""
        if (isinstance(image, torch.Tensor) and
            isinstance(condition_image, torch.Tensor) and
            isinstance(mask, torch.Tensor)):
            return image, condition_image, mask
        assert image.size == mask.size, "Image and mask must have the same size"
        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))
        return image, condition_image, mask

    @torch.no_grad()
    def forward(
        self,
        image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
    ):
        """
        Run inference on a single image.

        Args:
            image: Input image (WSI crop)
            mask: Inpainting mask
            condition_image: Source image for inpainting

        Returns:
            numpy array: Inpainted image
        """
        classifier_free_guidance = False
        generator = None
        concat_dim = -2

        # Prepare inputs
        image, condition_image, mask = self.check_inputs(
            image, condition_image, mask, self.width, self.height
        )

        masked_latent_concat, mask_latent_concat = self.prepare_unet_input_latent(
            image, mask, condition_image,
            classifier_free_guidance=classifier_free_guidance,
            mask_main_image=True,
        )

        # Prepare noise
        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )

        # Prepare timesteps
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma

        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, self.eta)
        num_warmup_steps = len(timesteps) - self.num_inference_steps * self.noise_scheduler.order

        with tqdm.tqdm(total=self.num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                non_inpainting_latent_model_input = latents
                non_inpainting_latent_model_input = self.noise_scheduler.scale_model_input(
                    non_inpainting_latent_model_input, t
                )

                # Prepare inpainting model input
                inpainting_latent_model_input = torch.cat([
                    non_inpainting_latent_model_input,
                    mask_latent_concat,
                    masked_latent_concat,
                ], dim=1)

                # Predict noise
                noise_pred = self.unet(
                    inpainting_latent_model_input,
                    t.to(self.device),
                    encoder_hidden_states=None,
                    return_dict=False,
                )[0]

                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Decode final latents
        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        return image[0, ...].cpu().numpy().transpose([1, 2, 0])

