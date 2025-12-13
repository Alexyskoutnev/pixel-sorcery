"""
Quick visualization utilities for debugging.

Usage:
    from src.utils.viz import show, save, compare, tensor_to_image

    # Show a tensor or image
    show(tensor)
    show(pil_image)
    show("path/to/image.jpg")

    # Save quickly
    save(tensor, "debug.jpg")

    # Compare input vs output side by side
    compare(input_tensor, output_tensor)
    compare(input_tensor, output_tensor, target_tensor)  # 3-way comparison
"""

from pathlib import Path
from typing import Union, Optional, List
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from src.utils.logging import logger


ImageLike = Union[torch.Tensor, Image.Image, str, Path]


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor to PIL Image.

    Args:
        tensor: (C, H, W) or (B, C, H, W) tensor in [0, 1] range

    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image from batch

    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")

    # Ensure on CPU and clamp to valid range
    tensor = tensor.detach().cpu().clamp(0, 1)

    return TF.to_pil_image(tensor)


def image_to_tensor(image: ImageLike) -> torch.Tensor:
    """
    Convert various image types to tensor.

    Args:
        image: Tensor, PIL Image, or path string

    Returns:
        Tensor (C, H, W) in [0, 1] range
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        return image.detach().cpu().clamp(0, 1)

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    return TF.to_tensor(image)


def _to_pil(image: ImageLike) -> Image.Image:
    """Convert anything to PIL Image."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, torch.Tensor):
        return tensor_to_image(image)
    raise TypeError(f"Cannot convert {type(image)} to PIL Image")


def show(
    image: ImageLike,
    title: Optional[str] = None,
    size: Optional[tuple[int, int]] = None,
) -> None:
    """
    Display an image. Works with tensors, PIL images, or file paths.

    Args:
        image: Tensor (C,H,W) or (B,C,H,W), PIL Image, or path string
        title: Optional window title (printed to console)
        size: Optional (width, height) to resize for display

    Example:
        show(model_output)
        show("path/to/image.jpg")
        show(batch[0], title="First sample")
    """
    pil_img = _to_pil(image)

    if size:
        pil_img = pil_img.resize(size, Image.BILINEAR)

    if title:
        logger.debug(f"Showing: {title} ({pil_img.size[0]}x{pil_img.size[1]})")

    pil_img.show()


def save(
    image: ImageLike,
    path: str = "debug.jpg",
    quality: int = 95,
) -> str:
    """
    Save an image quickly.

    Args:
        image: Tensor, PIL Image, or path to copy from
        path: Output path (default: debug.jpg)
        quality: JPEG quality (default: 95)

    Returns:
        Path where image was saved

    Example:
        save(model_output)  # Saves to debug.jpg
        save(tensor, "output.png")
    """
    pil_img = _to_pil(image)

    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Save with appropriate format
    if path.lower().endswith(".png"):
        pil_img.save(path)
    else:
        pil_img.save(path, quality=quality)

    logger.debug(f"Saved: {path} ({pil_img.size[0]}x{pil_img.size[1]})")
    return path


def compare(
    *images: ImageLike,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    gap: int = 10,
    show_result: bool = True,
) -> Image.Image:
    """
    Compare multiple images side by side.

    Args:
        *images: 2 or more images (tensors, PIL, or paths)
        labels: Optional labels for each image
        save_path: Optional path to save the comparison
        gap: Pixel gap between images
        show_result: Whether to display the result

    Returns:
        Combined PIL Image

    Example:
        compare(input_img, output_img)
        compare(input, output, target, labels=["Input", "Output", "Target"])
        compare(img1, img2, save_path="comparison.jpg", show_result=False)
    """
    if len(images) < 2:
        raise ValueError("Need at least 2 images to compare")

    # Convert all to PIL
    pil_images = [_to_pil(img) for img in images]

    # Resize all to match the first image's height
    target_height = pil_images[0].size[1]
    resized = []
    for img in pil_images:
        if img.size[1] != target_height:
            ratio = target_height / img.size[1]
            new_width = int(img.size[0] * ratio)
            img = img.resize((new_width, target_height), Image.BILINEAR)
        resized.append(img)

    # Calculate total size
    total_width = sum(img.size[0] for img in resized) + gap * (len(resized) - 1)
    total_height = target_height

    # Add space for labels if provided
    label_height = 30 if labels else 0
    total_height += label_height

    # Create combined image
    combined = Image.new("RGB", (total_width, total_height), color=(40, 40, 40))

    # Paste images
    x_offset = 0
    for i, img in enumerate(resized):
        combined.paste(img, (x_offset, label_height))
        x_offset += img.size[0] + gap

    # Add labels if provided
    if labels:
        try:
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except (OSError, IOError):
                font = ImageFont.load_default()

            x_offset = 0
            for i, (img, label) in enumerate(zip(resized, labels)):
                # Center label above each image
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = x_offset + (img.size[0] - text_width) // 2
                draw.text((text_x, 5), label, fill=(255, 255, 255), font=font)
                x_offset += img.size[0] + gap
        except ImportError:
            pass  # Skip labels if PIL draw not available

    # Save if path provided
    if save_path:
        save(combined, save_path)

    # Show if requested
    if show_result:
        combined.show()

    return combined


def grid(
    images: List[ImageLike],
    cols: int = 4,
    size: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_result: bool = True,
    gap: int = 5,
) -> Image.Image:
    """
    Display images in a grid.

    Args:
        images: List of images
        cols: Number of columns
        size: Size for each image (width, height)
        save_path: Optional path to save
        show_result: Whether to display
        gap: Gap between images

    Returns:
        Combined PIL Image

    Example:
        grid([img1, img2, img3, img4], cols=2)
        grid(batch_of_tensors, cols=4, size=(256, 256))
    """
    pil_images = [_to_pil(img) for img in images]

    # Resize if size specified
    if size:
        pil_images = [img.resize(size, Image.BILINEAR) for img in pil_images]

    # Use first image size as reference
    img_w, img_h = pil_images[0].size

    # Calculate grid dimensions
    rows = (len(pil_images) + cols - 1) // cols
    total_w = cols * img_w + (cols - 1) * gap
    total_h = rows * img_h + (rows - 1) * gap

    # Create grid
    grid_img = Image.new("RGB", (total_w, total_h), color=(40, 40, 40))

    for i, img in enumerate(pil_images):
        row = i // cols
        col = i % cols
        x = col * (img_w + gap)
        y = row * (img_h + gap)

        # Resize to match if needed
        if img.size != (img_w, img_h):
            img = img.resize((img_w, img_h), Image.BILINEAR)

        grid_img.paste(img, (x, y))

    if save_path:
        save(grid_img, save_path)

    if show_result:
        grid_img.show()

    return grid_img


def show_batch(
    batch: dict,
    num_samples: int = 4,
    save_path: Optional[str] = None,
) -> Image.Image:
    """
    Visualize a batch from the dataloader.

    Args:
        batch: Batch dict with 'input' and 'target' keys
        num_samples: Number of samples to show
        save_path: Optional path to save

    Returns:
        Combined PIL Image

    Example:
        batch = next(iter(train_loader))
        show_batch(batch)
    """
    inputs = batch["input"][:num_samples]
    targets = batch["target"][:num_samples]

    # Interleave inputs and targets
    images = []
    labels = []
    for i in range(min(num_samples, len(inputs))):
        images.extend([inputs[i], targets[i]])
        labels.extend([f"Input {i+1}", f"Target {i+1}"])

    return grid(images, cols=2 * min(2, num_samples), save_path=save_path, show_result=True)
