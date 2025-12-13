"""
Trainer class for image-to-image models.

Provides a clean, modular training loop that works with any model
inheriting from BaseModel.

Features:
- Automatic device handling (GPU/CPU)
- Learning rate scheduling
- Checkpointing (best model and periodic saves)
- Logging (console)
- Validation during training
- Easy to extend

Usage:
    from src.training import Trainer
    from src.models import UNet

    model = UNet()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
    )
    trainer.train(num_epochs=100)
"""

from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from src.models.base import BaseModel
from src.utils.logging import logger
from .losses import CombinedLoss


@dataclass
class TrainerConfig:

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100

    # Scheduler
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau", or "none"
    scheduler_patience: int = 10  # For plateau scheduler
    scheduler_step_size: int = 30  # For step scheduler

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10  # Save checkpoint every N epochs
    save_best: bool = True  # Save best model based on validation loss

    # Logging
    log_every: int = 10  # Log every N batches

    # Device
    device: str = "auto"  # "auto", "cuda", "mps", or "cpu"

    # Mixed precision
    use_amp: bool = True  # Automatic mixed precision


class Trainer:
    """
    Trainer for image-to-image translation models.

    This trainer is designed to be:
    - Model-agnostic: Works with any model inheriting from BaseModel
    - Loss-agnostic: Works with any loss function
    - Easy to use: Sensible defaults, minimal configuration needed
    - Extensible: Override methods to customize behavior
    """

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        config: Optional[TrainerConfig] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train (must inherit from BaseModel)
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            loss_fn: Loss function (default: L1Loss)
            optimizer: Optimizer (default: AdamW)
            scheduler: LR scheduler (default: CosineAnnealingLR)
            config: Trainer configuration
        """
        self.config = config or TrainerConfig()

        self.device = self._setup_device()
        logger.info(f"Using device: {self.device}")

        self.model = model.to(self.device)
        logger.info(f"Model: {model.name}")
        logger.info(f"Parameters: {model.get_num_params():,}")
        logger.debug(f"Memory: {model.get_memory_usage_mb():.2f} MB")

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_fn = loss_fn or CombinedLoss(l1_weight=1.0)
        if hasattr(self.loss_fn, 'to'):
            self.loss_fn = self.loss_fn.to(self.device)

        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        self.scheduler = scheduler or self._create_scheduler()

        # Mixed Precision Training (AMP - Automatic Mixed Precision)
        # ============================================================
        #
        # WHAT IS MIXED PRECISION?
        # ------------------------
        # By default, PyTorch uses float32 (32-bit) for all computations.
        # Mixed precision uses float16 (16-bit) for most operations, which:
        #   - Uses ~50% less GPU memory (can use larger batches!)
        #   - Runs ~2-3x faster on modern GPUs (Tensor Cores)
        #   - Maintains accuracy by keeping critical ops in float32
        #
        # WHY DO WE NEED A "SCALER"?
        # --------------------------
        # float16 has a smaller range than float32:
        #   - float32 range: ~1e-38 to ~1e+38
        #   - float16 range: ~6e-5 to ~65504
        #
        # Problem: Gradients during backprop can be VERY small (e.g., 1e-8).
        # In float16, these tiny gradients become ZERO (underflow), and
        # the model stops learning!
        #
        # Solution: GradScaler does "loss scaling":
        #   1. SCALE UP the loss by a large factor (e.g., 65536) before backward()
        #   2. This makes gradients 65536x larger → they don't underflow
        #   3. UNSCALE gradients before optimizer.step() (divide by 65536)
        #   4. If gradients overflow (become inf/nan), skip the update and reduce scale
        #
        # THE WORKFLOW (see _train_epoch for usage):
        # ------------------------------------------
        #   with torch.amp.autocast("cuda"):   # Auto-converts ops to float16
        #       output = model(input)
        #       loss = loss_fn(output, target)
        #
        #   scaler.scale(loss).backward()  # Scale loss up, compute gradients
        #   scaler.step(optimizer)          # Unscale gradients, update weights
        #   scaler.update()                 # Adjust scale factor for next iteration
        #
        # NOTE: Only works on CUDA GPUs. MPS (Apple Silicon) doesn't support this yet.
        #
        self.scaler = torch.amp.GradScaler("cuda") if (
            self.config.use_amp and self.device.type == "cuda"
        ) else None

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _setup_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """
        Create a learning rate scheduler based on config.

        WHY USE A SCHEDULER?
        --------------------
        Learning rate (LR) controls how big of a step we take when updating weights.
        - Too high: Training is unstable, loss jumps around, may diverge
        - Too low: Training is slow, may get stuck in local minima

        Solution: Start with a higher LR (learn fast), then gradually decrease it
        (fine-tune). This is called "learning rate scheduling" or "LR decay".

        AVAILABLE SCHEDULERS:
        ---------------------

        1. COSINE ANNEALING (recommended, default)
           -----------------------------------------
           LR follows a cosine curve from initial LR down to eta_min.

           LR
           |‾‾‾‾‾‾‾‾‾\
           |          \
           |           \______
           |                  ‾‾‾‾‾
           └──────────────────────── Epochs
             start              end

           - Smooth decay, no sudden jumps
           - Reaches minimum LR at the end of training
           - Good default for most tasks
           - T_max: Number of epochs for one cycle (we set to total epochs)
           - eta_min: Minimum LR at the end (1e-7 = very small)

        2. STEP LR
           --------
           LR drops by a factor (gamma) every N epochs (step_size).

           LR
           |‾‾‾‾‾|
           |     |_____
           |           |_____
           |                 |_____
           └────────────────────── Epochs
                30    60    90

           - Simple and predictable
           - Can cause sudden changes in training dynamics
           - step_size: How many epochs between drops (default: 30)
           - gamma: Multiply LR by this each step (0.5 = halve it)

        3. REDUCE ON PLATEAU
           ------------------
           LR drops only when validation loss stops improving.

           LR
           |‾‾‾‾‾‾‾‾‾‾|
           |          |_____ (loss stopped improving)
           |                |____________
           └─────────────────────────── Epochs

           - Adaptive: only reduces when needed
           - Good when you don't know how many epochs you'll need
           - patience: Wait this many epochs before reducing
           - factor: Multiply LR by this when reducing (0.5 = halve it)
           - mode="min": We want to minimize loss

        Returns:
            Configured scheduler, or None if scheduler_type is "none"
        """
        if self.config.scheduler_type == "cosine":
            # Cosine annealing: smoothly decrease LR following a cosine curve
            # Best for: Most training scenarios, smooth convergence
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,  # Complete one cosine cycle over all epochs
                eta_min=1e-7,  # Don't let LR go to absolute zero
            )

        elif self.config.scheduler_type == "step":
            # Step decay: drop LR by gamma every step_size epochs
            # Best for: When you want predictable, discrete LR drops
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,  # Drop every N epochs
                gamma=0.5,  # Multiply LR by 0.5 (halve it) at each step
            )

        elif self.config.scheduler_type == "plateau":
            # Reduce on plateau: drop LR when loss stops improving
            # Best for: When you're not sure how long to train
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",  # "min" = reduce when metric stops decreasing
                patience=self.config.scheduler_patience,  # Wait N epochs before reducing
                factor=0.5,  # Multiply LR by 0.5 when reducing
            )

        # No scheduler - LR stays constant throughout training
        return None

    def train(self, 
              num_epochs: Optional[int] = None) -> dict[str, list]:
        num_epochs = num_epochs or self.config.num_epochs

        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Training batches: {len(self.train_loader)}")
        if self.val_loader:
            logger.info(f"Validation batches: {len(self.val_loader)}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_loss = self._train_epoch()
            self.history["train_loss"].append(train_loss)

            val_loss = None
            if self.val_loader:
                val_loss = self._validate()
                self.history["val_loss"].append(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss else train_loss)
                else:
                    self.scheduler.step()

            self._log_epoch(epoch, train_loss, val_loss, current_lr)

            if self.config.save_every and (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pt")

            if self.config.save_best and val_loss and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model.pt")
                logger.info(f"New best model saved! (val_loss: {val_loss:.6f})")

        logger.info("Training complete!")
        self._save_checkpoint("final_model.pt")

        return self.history

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in pbar:
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                # Mixed precision path (CUDA only, ~2x faster, ~50% less memory)
                with torch.amp.autocast("cuda"):
                    # autocast automatically converts operations to float16 where safe
                    outputs = self.model(inputs)
                    loss, _ = self._compute_loss(outputs, targets)

                # Scale loss → backward (gradients are scaled)
                self.scaler.scale(loss).backward()
                # Unscale gradients → optimizer step (skip if inf/nan)
                self.scaler.step(self.optimizer)
                # Adjust scale factor for next iteration
                self.scaler.update()
            else:
                # Standard float32 path (CPU, MPS, or AMP disabled)
                outputs = self.model(inputs)
                loss, _ = self._compute_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / num_batches

    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs = self.model(inputs)
                loss, _ = self._compute_loss(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss, handling different loss function signatures."""
        result = self.loss_fn(outputs, targets)

        # Handle loss functions that return (loss, dict) or just loss
        if isinstance(result, tuple):
            return result
        else:
            return result, {"loss": result.item()}

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        lr: float,
    ):
        """Log epoch results."""
        msg = f"Epoch {epoch + 1}: train_loss={train_loss:.6f}"
        if val_loss:
            msg += f", val_loss={val_loss:.6f}"
        msg += f", lr={lr:.2e}"
        logger.info(msg)

    def _save_checkpoint(self, filename: str):
        """Save a checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "model_config": self.model.get_config(),
        }, path)

    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]

        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch + 1})")

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run inference on inputs.

        Args:
            inputs: Input tensor (B, C, H, W) or single image (C, H, W)

        Returns:
            Output tensor
        """
        self.model.eval()

        # Handle single image
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)

        return outputs.cpu()

    def visualize_samples(
        self,
        num_samples: int = 4,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Create a matplotlib dashboard showing input → output → target comparisons.

        Args:
            num_samples: Number of samples to visualize (default: 4)
            save_path: Optional path to save the figure (e.g., "eval_samples.png")
            show: Whether to display the figure (default: True)

        Example:
            trainer.visualize_samples()  # Show 4 samples
            trainer.visualize_samples(num_samples=8, save_path="results.png")
        """
        import matplotlib.pyplot as plt

        if self.val_loader is None:
            logger.warning("No validation loader available for visualization")
            return

        self.model.eval()

        # Get a batch from validation set
        batch = next(iter(self.val_loader))
        inputs = batch["input"][:num_samples].to(self.device)
        targets = batch["target"][:num_samples]

        # Run inference
        with torch.no_grad():
            outputs = self.model(inputs).cpu()

        inputs = inputs.cpu()

        # Create figure: rows = num_samples, cols = 3 (input, output, target)
        fig, axes = plt.subplots(
            num_samples, 3,
            figsize=(12, 4 * num_samples),
            squeeze=False,
        )

        # Column titles
        col_titles = ["Input (Source)", "Output (Model)", "Target (Ground Truth)"]

        for row in range(num_samples):
            images = [inputs[row], outputs[row], targets[row]]

            for col, (img, title) in enumerate(zip(images, col_titles)):
                ax = axes[row, col]

                # Convert tensor (C, H, W) to numpy (H, W, C) for matplotlib
                img_np = img.permute(1, 2, 0).clamp(0, 1).numpy()
                ax.imshow(img_np)
                ax.axis("off")

                # Add column title only on first row
                if row == 0:
                    ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved evaluation samples to: {save_path}")

        if show:
            plt.show()

        plt.close(fig)
