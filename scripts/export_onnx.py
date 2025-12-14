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

    model = get_model(model_name)

    if "generator_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["generator_state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("No model weights found in checkpoint")

    model.to(device)
    model.eval()
    return model


def export_onnx(model: torch.nn.Module, 
                output_path: str, 
                image_size: int = 512):
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
