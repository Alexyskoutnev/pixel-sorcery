"""
GAN Trainer for adversarial image-to-image training.

Trains a generator (U-Net) with a discriminator (PatchGAN) for sharper results.

================================================================================
GAN TRAINING OVERVIEW
================================================================================

Standard U-Net training:
    Loss = L1(output, target) + Perceptual(output, target)
    Problem: Network learns to output "safe" blurry average

GAN training adds adversarial loss:
    Loss = L1 + Perceptual + λ_adv * Adversarial
    The discriminator punishes blurry/unrealistic outputs

Training alternates between:
    1. Train Discriminator: Learn to distinguish real from fake
    2. Train Generator: Learn to fool the discriminator

================================================================================
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.base import BaseModel
from src.models.discriminator import PatchDiscriminator, get_discriminator
from src.utils.logging import logger
from .losses import CombinedLoss


@dataclass
class GANTrainerConfig:
    """Configuration for GAN training."""

    # Learning rates
    lr_generator: float = 1e-4
    lr_discriminator: float = 1e-4
    weight_decay: float = 1e-5

    # Training
    num_epochs: int = 100

    # Loss weights
    l1_weight: float = 100.0  # High weight keeps output close to target
    perceptual_weight: float = 10.0
    ssim_weight: float = 1.0
    adversarial_weight: float = 1.0  # Weight for GAN loss

    # Discriminator settings
    discriminator_type: str = "patch"  # "patch", "patch_small", "multiscale"
    n_discriminator_steps: int = 1  # D steps per G step

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10

    # Device
    device: str = "auto"


class GANTrainer:
    """
    GAN Trainer for image-to-image translation.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        GAN TRAINING LOOP                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  For each batch:                                                        │
    │                                                                         │
    │  1. TRAIN DISCRIMINATOR                                                 │
    │     ─────────────────────                                               │
    │     real_pred = D(input, target)     # Should be 1 (real)              │
    │     fake_pred = D(input, G(input))   # Should be 0 (fake)              │
    │     D_loss = BCE(real_pred, 1) + BCE(fake_pred, 0)                     │
    │     Update D weights                                                    │
    │                                                                         │
    │  2. TRAIN GENERATOR                                                     │
    │     ────────────────                                                    │
    │     output = G(input)                                                   │
    │     fake_pred = D(input, output)     # Want this to be 1 (fool D)      │
    │     G_loss = L1 + Perceptual + λ_adv * BCE(fake_pred, 1)               │
    │     Update G weights                                                    │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    The key insight: Generator is trained to FOOL the discriminator, which
    forces it to produce realistic-looking outputs instead of blurry averages.
    """

    def __init__(
        self,
        generator: BaseModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[GANTrainerConfig] = None,
    ):
        self.config = config or GANTrainerConfig()
        self.device = self._setup_device()
        logger.info(f"Using device: {self.device}")

        # Generator (U-Net)
        self.generator = generator.to(self.device)
        logger.info(f"Generator: {generator.name}")
        logger.info(f"Generator params: {generator.get_num_params():,}")

        # Discriminator (PatchGAN)
        self.discriminator = get_discriminator(self.config.discriminator_type)
        self.discriminator = self.discriminator.to(self.device)
        logger.info(f"Discriminator: {self.config.discriminator_type}")
        logger.info(f"Discriminator params: {self.discriminator.get_num_params():,}")

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Reconstruction loss (L1 + Perceptual + SSIM)
        self.recon_loss = CombinedLoss(
            l1_weight=self.config.l1_weight,
            perceptual_weight=self.config.perceptual_weight,
            ssim_weight=self.config.ssim_weight,
        ).to(self.device)

        # Adversarial loss (BCE)
        self.adversarial_loss = nn.BCEWithLogitsLoss()

        # Optimizers (separate for G and D)
        self.optimizer_G = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.config.lr_generator,
            betas=(0.5, 0.999),  # Standard GAN betas
            weight_decay=self.config.weight_decay,
        )
        self.optimizer_D = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config.lr_discriminator,
            betas=(0.5, 0.999),
            weight_decay=self.config.weight_decay,
        )

        # Schedulers
        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_G, T_max=self.config.num_epochs, eta_min=1e-7
        )
        self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_D, T_max=self.config.num_epochs, eta_min=1e-7
        )

        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.history = {
            "g_loss": [], "d_loss": [], "d_real": [], "d_fake": [],
            "val_loss": [], "lr_g": [], "lr_d": []
        }

        # Checkpointing
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _setup_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.config.device)

    def _get_labels(self, size: tuple, real: bool) -> torch.Tensor:
        """Create labels for real (1) or fake (0) patches."""
        # Label smoothing: use 0.9 for real, 0.1 for fake (helps training)
        value = 0.9 if real else 0.1
        return torch.full(size, value, device=self.device)

    def train(self) -> dict:
        logger.info(f"Starting GAN training for {self.config.num_epochs} epochs...")
        logger.info(f"Adversarial weight: {self.config.adversarial_weight}")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            g_loss, d_loss, d_real, d_fake = self._train_epoch()

            self.history["g_loss"].append(g_loss)
            self.history["d_loss"].append(d_loss)
            self.history["d_real"].append(d_real)
            self.history["d_fake"].append(d_fake)

            # Validation
            val_loss = None
            if self.val_loader:
                val_loss = self._validate()
                self.history["val_loss"].append(val_loss)

            # Learning rates
            lr_g = self.optimizer_G.param_groups[0]["lr"]
            lr_d = self.optimizer_D.param_groups[0]["lr"]
            self.history["lr_g"].append(lr_g)
            self.history["lr_d"].append(lr_d)

            self.scheduler_G.step()
            self.scheduler_D.step()

            # Logging
            msg = f"Epoch {epoch + 1}: G={g_loss:.4f}, D={d_loss:.4f}, "
            msg += f"D(real)={d_real:.2f}, D(fake)={d_fake:.2f}"
            if val_loss:
                msg += f", val={val_loss:.4f}"
            logger.info(msg)

            # Checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pt")

            if val_loss and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model.pt")
                logger.info(f"New best model! (val_loss: {val_loss:.4f})")

        self._save_checkpoint("final_model.pt")
        logger.info("GAN training complete!")
        return self.history

    def _train_epoch(self) -> tuple[float, float, float, float]:
        self.generator.train()
        self.discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0
        total_d_real = 0.0
        total_d_fake = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in pbar:
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)

            # ═══════════════════════════════════════════════════════════════
            # TRAIN DISCRIMINATOR
            # ═══════════════════════════════════════════════════════════════
            for _ in range(self.config.n_discriminator_steps):
                self.optimizer_D.zero_grad()

                # Real pairs (input, target) - should predict "real"
                pred_real = self.discriminator(inputs, targets)
                labels_real = self._get_labels(pred_real.shape, real=True)
                loss_D_real = self.adversarial_loss(pred_real, labels_real)

                # Fake pairs (input, generated) - should predict "fake"
                with torch.no_grad():
                    fake_outputs = self.generator(inputs)
                pred_fake = self.discriminator(inputs, fake_outputs.detach())
                labels_fake = self._get_labels(pred_fake.shape, real=False)
                loss_D_fake = self.adversarial_loss(pred_fake, labels_fake)

                # Combined D loss
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                self.optimizer_D.step()

            # ═══════════════════════════════════════════════════════════════
            # TRAIN GENERATOR
            # ═══════════════════════════════════════════════════════════════
            self.optimizer_G.zero_grad()

            # Generate outputs
            outputs = self.generator(inputs)

            # Reconstruction loss (L1 + Perceptual + SSIM)
            recon_loss, _ = self.recon_loss(outputs, targets)

            # Adversarial loss (fool D into saying "real")
            pred_fake_for_G = self.discriminator(inputs, outputs)
            labels_real_for_G = self._get_labels(pred_fake_for_G.shape, real=True)
            loss_G_adv = self.adversarial_loss(pred_fake_for_G, labels_real_for_G)

            # Total generator loss
            loss_G = recon_loss + self.config.adversarial_weight * loss_G_adv
            loss_G.backward()
            self.optimizer_G.step()

            # Track metrics
            total_g_loss += loss_G.item()
            total_d_loss += loss_D.item()
            total_d_real += torch.sigmoid(pred_real).mean().item()
            total_d_fake += torch.sigmoid(pred_fake).mean().item()

            pbar.set_postfix({
                "G": f"{loss_G.item():.3f}",
                "D": f"{loss_D.item():.3f}",
                "D(r)": f"{torch.sigmoid(pred_real).mean().item():.2f}",
                "D(f)": f"{torch.sigmoid(pred_fake).mean().item():.2f}",
            })

        n = len(self.train_loader)
        return total_g_loss / n, total_d_loss / n, total_d_real / n, total_d_fake / n

    def _validate(self) -> float:
        """Validate generator only (reconstruction loss)."""
        self.generator.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
                outputs = self.generator(inputs)
                loss, _ = self.recon_loss(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _save_checkpoint(self, filename: str):
        """Save generator, discriminator, and training state."""
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.current_epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": self.config,
            "model_config": self.generator.get_config(),
        }, path)
        logger.info(f"Saved checkpoint: {path}")

        # Save eval samples
        if self.val_loader:
            eval_path = str(path).replace(".pt", "_eval.png")
            self._visualize_samples(eval_path)

    def _visualize_samples(self, save_path: str, num_samples: int = 4):
        """Generate and save sample comparisons."""
        import matplotlib.pyplot as plt

        self.generator.eval()
        batch = next(iter(self.val_loader))
        inputs = batch["input"][:num_samples].to(self.device)
        targets = batch["target"][:num_samples]

        with torch.no_grad():
            outputs = self.generator(inputs).cpu()
        inputs = inputs.cpu()

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

        for i in range(num_samples):
            for j, (img, title) in enumerate(zip(
                [inputs[i], outputs[i], targets[i]],
                ["Input", "Output (GAN)", "Target"]
            )):
                ax = axes[i, j] if num_samples > 1 else axes[j]
                ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
                ax.axis("off")
                if i == 0:
                    ax.set_title(title, fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved eval: {save_path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]
        logger.info(f"Loaded checkpoint from {path}")
