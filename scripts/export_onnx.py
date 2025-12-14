import argparse
import sys
from pathlib import Path

import torch
import onnx

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import get_model


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})
    model_name = model_config.get("name", "unet")

    name_mapping = {"UNetSmall": "unet_small", "UNet": "unet", "UNetLarge": "unet_large"}
    model_name = name_mapping.get(model_name, model_name)

    # Extract model architecture parameters from checkpoint
    model_kwargs = {}
    if "base_channels" in model_config:
        model_kwargs["base_channels"] = model_config["base_channels"]
        print(f"Using base_channels={model_config['base_channels']} from checkpoint")
    if "depth" in model_config:
        model_kwargs["depth"] = model_config["depth"]
        print(f"Using depth={model_config['depth']} from checkpoint")

    model = get_model(model_name, **model_kwargs)

    if "generator_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["generator_state_dict"])
        print("Loaded GAN generator weights")
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model weights")
    else:
        raise ValueError("No model weights found in checkpoint")

    model.to(device)
    model.eval()
    return model


def export_onnx(model: torch.nn.Module,
                output_path: str,
                image_size: int = 512):
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--size", "-s", type=int, default=512)
    args = parser.parse_args()

    if args.output is None:
        bin_dir = Path(__file__).parent.parent / "bin"
        bin_dir.mkdir(exist_ok=True)
        args.output = str(bin_dir / Path(args.checkpoint).with_suffix(".onnx").name)

    model = load_model(args.checkpoint, torch.device("cpu"))
    export_onnx(model, args.output, args.size)
    print(f"Exported: {args.output}")


if __name__ == "__main__":
    main()
