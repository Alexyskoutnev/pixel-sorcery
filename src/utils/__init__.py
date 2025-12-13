"""
Utility functions for debugging, visualization, and logging.

Quick usage:
    from src.utils import show, save, compare, logger

    logger.info("Starting training")
    show(tensor)           # Display any image
    save(tensor)           # Save to debug.jpg
    compare(inp, out)      # Side-by-side comparison
"""

from .viz import (
    show,
    save,
    compare,
    grid,
    show_batch,
    tensor_to_image,
    image_to_tensor,
)
from .logging import logger, set_log_level, add_file_logging

__all__ = [
    # Logging
    "logger",
    "set_log_level",
    "add_file_logging",
    # Visualization
    "show",
    "save",
    "compare",
    "grid",
    "show_batch",
    "tensor_to_image",
    "image_to_tensor",
]
