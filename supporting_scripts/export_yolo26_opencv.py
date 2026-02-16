#!/usr/bin/env python3
"""
YOLO26 Model Export Script for OpenCV DNN Compatibility

This script exports YOLO26 models in a format compatible with OpenCV's DNN module.

Background:
-----------
YOLO26 has two detection heads:
1. One-to-One Head (end2end=True, default):
   - Outputs: [batch, 300, 6] where 6 = [x1, y1, x2, y2, confidence, class_id]
   - NMS-free, 43% faster inference
   - Works with ONNXRuntime but NOT fully supported by OpenCV DNN

2. One-to-Many Head (end2end=False):
   - Outputs: [batch, 84, 8400] where 84 = [4 bbox coords + 80 classes]
   - Requires NMS post-processing
   - Fully compatible with OpenCV DNN module

This script exports with end2end=False for OpenCV DNN compatibility.

Usage:
------
    python3 export_yolo26_opencv.py --model yolo26n.pt --output yolo26n_opencv.onnx

    # Export different model sizes:
    python3 export_yolo26_opencv.py --model yolo26s.pt
    python3 export_yolo26_opencv.py --model yolo26m.pt
    python3 export_yolo26_opencv.py --model yolo26l.pt
    python3 export_yolo26_opencv.py --model yolo26x.pt

Requirements:
-------------
    pip install ultralytics onnx onnxslim

Date: 2026-02-16
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
    import onnx
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Please install: pip install ultralytics onnx onnxslim")
    sys.exit(1)


def export_yolo26_for_opencv(
    model_path: str,
    output_path: str = None,
    imgsz: int = 640,
    simplify: bool = True,
    opset: int = 19,
    verbose: bool = True
) -> Path:
    """
    Export YOLO26 model in OpenCV DNN compatible format.

    Args:
        model_path (str): Path to YOLO26 .pt model file
        output_path (str, optional): Output ONNX file path. If None, uses default naming.
        imgsz (int): Input image size (default: 640)
        simplify (bool): Simplify ONNX model (default: True)
        opset (int): ONNX opset version (default: 19)
        verbose (bool): Print detailed information (default: True)

    Returns:
        Path: Path to exported ONNX model

    Raises:
        FileNotFoundError: If model_path doesn't exist
        Exception: If export fails
    """

    # Validate input model path
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if verbose:
        print("=" * 80)
        print("YOLO26 to OpenCV DNN Export Tool")
        print("=" * 80)
        print(f"\nInput Model:  {model_path}")
        print(f"Image Size:   {imgsz}x{imgsz}")
        print(f"Simplify:     {simplify}")
        print(f"ONNX Opset:   {opset}")
        print("\nExport Mode:  One-to-Many Head (end2end=False)")
        print("Output Format: [batch, 84, 8400] - Compatible with OpenCV DNN")
        print("-" * 80)

    try:
        # Load the YOLO26 model
        if verbose:
            print(f"\n[1/3] Loading model from '{model_path}'...")
        model = YOLO(str(model_path))

        if verbose:
            print(f"      Model loaded successfully!")
            print(f"      Task: {model.task}")
            print(f"      Model type: {model.model.__class__.__name__}")

        # Determine output path
        if output_path is None:
            output_path = model_path.stem + "_opencv.onnx"
        output_path = Path(output_path)

        if verbose:
            print(f"\n[2/3] Exporting to ONNX format...")
            print(f"      Output: {output_path}")

        # Export with one-to-many head for OpenCV DNN compatibility
        # The end2end=False parameter forces the model to use the traditional
        # one-to-many detection head instead of the NMS-free one-to-one head
        export_path = model.export(
            format='onnx',
            imgsz=imgsz,
            simplify=simplify,
            opset=opset,
            end2end=False  # Critical: Use one-to-many head for OpenCV DNN
        )

        if verbose:
            print(f"      Export successful!")

        # Verify the exported model
        if verbose:
            print(f"\n[3/3] Verifying exported model...")

        try:
            onnx_model = onnx.load(str(export_path))
            onnx.checker.check_model(onnx_model)

            # Get output shape information
            output_info = onnx_model.graph.output[0]
            output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]

            if verbose:
                print(f"      ✓ ONNX model is valid")
                print(f"      ✓ Output shape: {output_shape}")

                # Verify it's the correct format for OpenCV
                if len(output_shape) == 3 and output_shape[1] == 84:
                    print(f"      ✓ Format confirmed: One-to-Many Head (OpenCV compatible)")
                elif len(output_shape) == 3 and output_shape[1] == 300:
                    print(f"      ⚠ Warning: Output shows One-to-One format [1, 300, 6]")
                    print(f"        This may not work correctly with OpenCV DNN!")
                else:
                    print(f"      ⚠ Warning: Unexpected output shape")

        except Exception as e:
            if verbose:
                print(f"      ⚠ Warning: Could not verify model - {e}")

        # Rename to desired output path if different
        export_path = Path(export_path)
        if export_path != output_path:
            export_path.rename(output_path)
            final_path = output_path
        else:
            final_path = export_path

        if verbose:
            print("\n" + "=" * 80)
            print("✓ Export completed successfully!")
            print("=" * 80)
            print(f"\nExported model: {final_path.absolute()}")
            print(f"Model size:     {final_path.stat().st_size / (1024*1024):.2f} MB")
            print("\nUsage with OpenCV DNN:")
            print(f"  net = cv::dnn::readNetFromONNX(\"{final_path.name}\");")
            print("\nPost-processing required:")
            print("  - Use YOLOv8 post-processing (not YOLO26 one-to-one)")
            print("  - Apply NMS with threshold ~0.45")
            print("  - Output format: [1, 84, 8400]")
            print("  - Decode: 4 bbox coords + 80 class scores per anchor")
            print("=" * 80)

        return final_path

    except Exception as e:
        print(f"\n✗ Export failed: {e}", file=sys.stderr)
        raise


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Export YOLO26 models for OpenCV DNN compatibility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model yolo26n.pt
  %(prog)s --model yolo26n.pt --output custom_name.onnx
  %(prog)s --model yolo26s.pt --imgsz 640 --simplify
  %(prog)s --model yolo26l.pt --no-simplify --opset 17

Model Sizes:
  yolo26n - Nano:   2.6M params,  6.1 GFLOPs (fastest, smallest)
  yolo26s - Small: 10.0M params, 22.8 GFLOPs
  yolo26m - Medium: 21.9M params, 75.4 GFLOPs
  yolo26l - Large: 26.3M params, 93.8 GFLOPs
  yolo26x - XLarge: 59.0M params, 209.5 GFLOPs (slowest, most accurate)
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default='yolo26n.pt',
        help='Path to YOLO26 .pt model file (default: yolo26n.pt)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output ONNX file path (default: <model_name>_opencv.onnx)'
    )

    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )

    parser.add_argument(
        '--simplify',
        dest='simplify',
        action='store_true',
        default=True,
        help='Simplify ONNX model (default: True)'
    )

    parser.add_argument(
        '--no-simplify',
        dest='simplify',
        action='store_false',
        help='Do not simplify ONNX model'
    )

    parser.add_argument(
        '--opset',
        type=int,
        default=19,
        help='ONNX opset version (default: 19)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )

    args = parser.parse_args()

    try:
        export_yolo26_for_opencv(
            model_path=args.model,
            output_path=args.output,
            imgsz=args.imgsz,
            simplify=args.simplify,
            opset=args.opset,
            verbose=not args.quiet
        )
        return 0

    except KeyboardInterrupt:
        print("\n\nExport cancelled by user.")
        return 130

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
