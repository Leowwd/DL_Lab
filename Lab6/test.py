"""
test.py — Tester for Conditional DDPM

Provides:
  - Image generation from test conditions
  - Evaluation using pre-trained ResNet18 classifier
  - Grid visualization (8 per row, 4 rows)
  - Denoising process visualization
  - Individual image saving for submission
"""

import os
import torch
from torchvision.utils import make_grid, save_image

from dataset import get_test_conditions
from code.model import ConditionalUNet
from ddpm import DDPMScheduler


class Tester:
    """Generate and evaluate synthetic images from a trained DDPM.

    Usage:
        tester = Tester(config)
        tester.run()
    """

    def __init__(self, config):
        """
        Args:
            config: argparse namespace with test parameters
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # Model
        self.model = ConditionalUNet(
            in_channels=3,
            out_channels=3,
            base_channels=config.base_channels,
            channel_mults=config.channel_mults,
            num_classes=config.num_classes,
            time_dim=config.time_dim,
            dropout=0.0,  # no dropout at inference
        ).to(self.device)

        # Load checkpoint
        self._load_checkpoint(config.ckpt)

        # Scheduler
        self.scheduler = DDPMScheduler(
            num_timesteps=config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            schedule=config.schedule,
            device=self.device,
        )

    def _load_checkpoint(self, path):
        """Load model weights from checkpoint."""
        print(f"[Tester] Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        # Prefer EMA model weights
        if 'ema_model' in ckpt:
            self.model.load_state_dict(ckpt['ema_model'])
            print("[Tester] Loaded EMA model weights")
        elif 'model' in ckpt:
            self.model.load_state_dict(ckpt['model'])
            print("[Tester] Loaded model weights")
        else:
            self.model.load_state_dict(ckpt)
            print("[Tester] Loaded raw state dict")
        self.model.eval()

    def _denormalize(self, images):
        """Convert images from [-1, 1] to [0, 1] for visualization."""
        return (images.clamp(-1, 1) + 1) / 2

    def generate_images(self, conditions, cfg_scale=None):
        """Generate images for the given conditions.

        Args:
            conditions: (N, 24) one-hot label tensor
            cfg_scale: override CFG scale (default: config.cfg_scale)

        Returns:
            images: (N, 3, 64, 64) in [-1, 1]
        """
        if cfg_scale is None:
            cfg_scale = self.config.cfg_scale

        B = conditions.shape[0]
        shape = (B, 3, self.config.img_size, self.config.img_size)

        images = self.scheduler.sample(
            self.model, conditions, shape, self.device,
            cfg_scale=cfg_scale,
        )
        return images.clamp(-1, 1)

    def evaluate(self, test_json_path, name="test"):
        """Generate images for a test JSON and compute accuracy.

        Args:
            test_json_path: path to test.json or new_test.json
            name: label for logging

        Returns:
            accuracy: float
            images: (N, 3, 64, 64) generated images in [-1, 1]
        """
        from evaluator import evaluation_model

        conditions = get_test_conditions(
            test_json_path, self.config.objects_json,
        ).to(self.device)

        print(f"[Tester] Generating {conditions.shape[0]} images for {name}...")
        images = self.generate_images(conditions)

        evaluator = evaluation_model()
        acc = evaluator.eval(images, conditions)
        print(f"[Tester] {name} accuracy: {acc:.4f}")

        return acc, images, conditions

    def save_grid(self, images, path, nrow=8):
        """Save images as a grid.

        Args:
            images: (N, 3, H, W) tensor in [-1, 1]
            path: output file path
            nrow: number of images per row
        """
        grid = make_grid(self._denormalize(images), nrow=nrow, padding=2)
        save_image(grid, path)
        print(f"[Tester] Saved grid: {path}")

    def save_individual_images(self, images, output_dir):
        """Save each image individually as PNG.

        Images are named sequentially: 0.png, 1.png, ...

        Args:
            images: (N, 3, H, W) tensor in [-1, 1]
            output_dir: directory to save images
        """
        os.makedirs(output_dir, exist_ok=True)
        for i in range(images.shape[0]):
            img = self._denormalize(images[i])
            save_image(img, os.path.join(output_dir, f'{i}.png'))
        print(f"[Tester] Saved {images.shape[0]} images to {output_dir}")

    def generate_denoising_process(self, condition_labels, objects_json_path=None):
        """Generate and save the denoising process visualization.

        Shows the progression from pure noise to final image.

        Args:
            condition_labels: list of object name strings,
                e.g. ["red sphere", "cyan cylinder", "cyan cube"]
            objects_json_path: path to objects.json

        Returns:
            intermediates: list of image tensors at different timesteps
        """
        import json
        if objects_json_path is None:
            objects_json_path = self.config.objects_json

        with open(objects_json_path, 'r') as f:
            object_to_idx = json.load(f)

        # Build one-hot condition
        num_classes = len(object_to_idx)
        condition = torch.zeros(1, num_classes, device=self.device)
        for label in condition_labels:
            if label in object_to_idx:
                condition[0, object_to_idx[label]] = 1.0

        shape = (1, 3, self.config.img_size, self.config.img_size)

        # Define which timesteps to capture (evenly spaced, at least 8)
        num_intermediates = 9  # + final = 10 frames
        step_size = self.config.num_timesteps // num_intermediates
        intermediate_steps = list(range(0, self.config.num_timesteps, step_size))

        _, intermediates = self.scheduler.sample(
            self.model, condition, shape, self.device,
            cfg_scale=self.config.cfg_scale,
            return_intermediates=True,
            intermediate_steps=intermediate_steps,
        )

        return intermediates

    def run(self):
        """Full test pipeline: evaluate both test sets, save grids and denoising viz."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Evaluate test.json
        acc_test, images_test, _ = self.evaluate(self.config.test_json, name="test.json")
        self.save_grid(images_test, os.path.join(self.config.output_dir, 'test_grid.png'))
        self.save_individual_images(images_test, os.path.join(self.config.output_dir, 'test'))

        # Evaluate new_test.json
        acc_new, images_new, _ = self.evaluate(self.config.new_test_json, name="new_test.json")
        self.save_grid(images_new, os.path.join(self.config.output_dir, 'new_test_grid.png'))
        self.save_individual_images(images_new, os.path.join(self.config.output_dir, 'new_test'))

        # Denoising process visualization
        denoising_labels = ["red sphere", "cyan cylinder", "cyan cube"]
        print(f"[Tester] Generating denoising process for: {denoising_labels}")
        intermediates = self.generate_denoising_process(denoising_labels)

        # Stack intermediates into a single row grid
        denoising_images = torch.cat(intermediates, dim=0)  # (N_steps, 3, H, W)
        self.save_grid(
            denoising_images,
            os.path.join(self.config.output_dir, 'denoising_process.png'),
            nrow=len(intermediates),
        )

        print(f"\n{'='*50}")
        print(f"  test.json accuracy:     {acc_test:.4f}")
        print(f"  new_test.json accuracy:  {acc_new:.4f}")
        print(f"{'='*50}")
