#!/usr/bin/env python3
"""
Display ONNX model information.

Usage:
    python scripts/info_onnx.py --model bin/model_gan_512.onnx
    python scripts/info_onnx.py --model bin/model_gan_512.onnx --detailed
"""

import argparse
from pathlib import Path

import onnx
import onnxruntime as ort


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def print_model_info(model_path: str, detailed: bool = False):
    """
    Print ONNX model information.

    Args:
        model_path: Path to ONNX model
        detailed: Show detailed layer information
    """
    # Check file exists
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"{'='*70}")
    print(f"ONNX MODEL INFORMATION")
    print(f"{'='*70}")

    # File info
    file_size = model_file.stat().st_size
    print(f"\nFile: {model_path}")
    print(f"Size: {format_bytes(file_size)}")

    # Load model
    print("\nLoading model...")
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Basic info
    print(f"\n{'='*70}")
    print("MODEL METADATA")
    print(f"{'='*70}")

    # IR version and opset
    print(f"IR Version: {model.ir_version}")
    if model.opset_import:
        for opset in model.opset_import:
            domain = opset.domain if opset.domain else "ai.onnx"
            print(f"Opset: {domain} v{opset.version}")

    # Producer info
    if model.producer_name:
        print(f"Producer: {model.producer_name}")
    if model.producer_version:
        print(f"Producer Version: {model.producer_version}")

    # Model metadata
    if model.metadata_props:
        print(f"\nMetadata Properties:")
        for prop in model.metadata_props:
            print(f"  {prop.key}: {prop.value}")

    # Graph info
    graph = model.graph
    print(f"\n{'='*70}")
    print("GRAPH INFORMATION")
    print(f"{'='*70}")
    print(f"Graph Name: {graph.name if graph.name else 'N/A'}")
    print(f"Nodes: {len(graph.node)}")
    print(f"Initializers: {len(graph.initializer)}")

    # Inputs
    print(f"\n{'='*70}")
    print("INPUTS")
    print(f"{'='*70}")
    for i, input_tensor in enumerate(graph.input):
        name = input_tensor.name
        type_proto = input_tensor.type
        if type_proto.HasField("tensor_type"):
            tensor_type = type_proto.tensor_type
            dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
            shape = []
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(str(dim.dim_value))
                elif dim.HasField("dim_param"):
                    shape.append(dim.dim_param)
                else:
                    shape.append("?")
            shape_str = "[" + ", ".join(shape) + "]"
            print(f"{i+1}. {name}")
            print(f"   Shape: {shape_str}")
            print(f"   Type: {dtype}")

    # Outputs
    print(f"\n{'='*70}")
    print("OUTPUTS")
    print(f"{'='*70}")
    for i, output_tensor in enumerate(graph.output):
        name = output_tensor.name
        type_proto = output_tensor.type
        if type_proto.HasField("tensor_type"):
            tensor_type = type_proto.tensor_type
            dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
            shape = []
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(str(dim.dim_value))
                elif dim.HasField("dim_param"):
                    shape.append(dim.dim_param)
                else:
                    shape.append("?")
            shape_str = "[" + ", ".join(shape) + "]"
            print(f"{i+1}. {name}")
            print(f"   Shape: {shape_str}")
            print(f"   Type: {dtype}")

    # Runtime info
    print(f"\n{'='*70}")
    print("RUNTIME INFORMATION")
    print(f"{'='*70}")

    try:
        sess = ort.InferenceSession(model_path)
        providers = sess.get_providers()
        print(f"Available Execution Providers:")
        for provider in providers:
            print(f"  - {provider}")
    except Exception as e:
        print(f"Could not load runtime info: {e}")

    # Detailed layer info
    if detailed:
        print(f"\n{'='*70}")
        print("LAYER DETAILS")
        print(f"{'='*70}")

        # Count operations by type
        op_count = {}
        for node in graph.node:
            op_type = node.op_type
            op_count[op_type] = op_count.get(op_type, 0) + 1

        print(f"\nOperation Counts:")
        for op_type, count in sorted(op_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  {op_type}: {count}")

        print(f"\nAll Nodes:")
        for i, node in enumerate(graph.node):
            inputs = ", ".join(node.input[:3])  # Show first 3 inputs
            if len(node.input) > 3:
                inputs += f", ... (+{len(node.input)-3} more)"
            outputs = ", ".join(node.output)

            print(f"\n{i+1}. {node.name if node.name else f'node_{i}'}")
            print(f"   Op: {node.op_type}")
            print(f"   Inputs: {inputs}")
            print(f"   Outputs: {outputs}")

            # Show attributes if any
            if node.attribute:
                print(f"   Attributes:")
                for attr in node.attribute:
                    print(f"     - {attr.name}: {attr.type}")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Display ONNX model information")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed layer information"
    )

    args = parser.parse_args()

    print_model_info(args.model, args.detailed)


if __name__ == "__main__":
    main()
