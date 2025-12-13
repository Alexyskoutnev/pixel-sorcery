#!/usr/bin/env python3
"""
Main training script for image enhancement models.

This script provides a simple interface to train models on paired image data.

Usage:
    # Basic training with defaults
    python train.py --data_dir path/to/images

    # With custom settings
    python train.py \
        --data_dir path/to/images \
        --model unet \
        --image_size 512 \
        --batch_size 8 \
        --epochs 100 \
        --lr 1e-4

    # Use residual learning (recommended for enhancement tasks)
    python train.py --data_dir path/to/images --residual

Example with the hackathon data:
    python train.py \
        --data_dir autohdr-real-estate-577/images \
        --model unet \
        --image_size 512 \
        --batch_size 8 \
        --epochs 100
"""

import argparse
from datetime import datetime
from pathlib import Path

from src.models import get_model, MODEL_REGISTRY, ResidualWrapper
from src.datasets import get_dataloaders
from src.training import Trainer, TrainerConfig, CombinedLoss
from src.utils import logger


def generate_run_name(args) -> str:
    """Generate a unique run name based on config and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Format: model_imgsize_batchsize_timestamp
    return f"{args.model}_{args.image_size}px_bs{args.batch_size}_{timestamp}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train image enhancement models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing paired images (*_src.* and *_tar.*)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Training image size (images will be resized)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="unet_small",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to use",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=64,
        help="Base channel count for U-Net (doubles each level)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Number of encoder/decoder levels",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Use residual learning (output = input + model(input))",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit dataset size (for quick testing)",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="combined",
        choices=["l1", "l2", "combined"],
        help="Loss function type",
    )
    parser.add_argument(
        "--l1_weight",
        type=float,
        default=1.0,
        help="Weight for L1 loss (if using combined)",
    )
    parser.add_argument(
        "--perceptual_weight",
        type=float,
        default=0.1,
        help="Weight for perceptual loss (if using combined)",
    )
    parser.add_argument(
        "--ssim_weight",
        type=float,
        default=0.1,
        help="Weight for SSIM loss (if using combined)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )

    return parser.parse_args()


def main():
    args = parse_args()


    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    # Create data loaders
    logger.info(f"Loading data from: {data_dir}")
    train_loader, val_loader = get_dataloaders(
        root_dir=str(data_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )

    logger.info(f"Creating model: {args.model}")
    model_kwargs = {}
    if args.model == "unet":
        model_kwargs = {
            "base_channels": args.base_channels,
            "depth": args.depth,
        }

    model = get_model(args.model, **model_kwargs)

    # Wrap with residual if requested
    if args.residual:
        logger.info("Using residual learning (output = input + model(input))")
        model = ResidualWrapper(model)

    # Create loss function
    if args.loss == "l1":
        loss_fn = CombinedLoss(l1_weight=1.0)
    elif args.loss == "l2":
        loss_fn = CombinedLoss(l2_weight=1.0)
    else:  # combined
        loss_fn = CombinedLoss(
            l1_weight=args.l1_weight,
            perceptual_weight=args.perceptual_weight,
            ssim_weight=args.ssim_weight,
        )

    logger.info(f"Loss: {args.loss}")
    if args.loss == "combined":
        logger.debug(f"  L1 weight: {args.l1_weight}")
        logger.debug(f"  Perceptual weight: {args.perceptual_weight}")
        logger.debug(f"  SSIM weight: {args.ssim_weight}")

    # Generate unique run directory
    run_name = generate_run_name(args)
    checkpoint_dir = Path(args.checkpoint_dir) / run_name
    logger.info(f"Run: {run_name}")

    # Create trainer config
    config = TrainerConfig(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        checkpoint_dir=str(checkpoint_dir),
        save_every=args.save_every,
        device=args.device,
        use_amp=not args.no_amp,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train!
    history = trainer.train()

    logger.info("=" * 50)
    logger.info("Training Complete!")
    logger.info("=" * 50)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}/")
    logger.info(f"Best model: {checkpoint_dir}/best_model.pt")
    logger.info(f"Eval samples saved alongside each checkpoint (*_eval.png)")


if __name__ == "__main__":
    main()
