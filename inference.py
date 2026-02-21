"""
Inference script for Watermark Injection Model

Usage:
    python inference.py --model full_model.pth --input image.jpg --output watermarked.jpg
    python inference.py --model full_model.pth --input image.jpg  # auto-generates output name
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

# Import model classes (required for torch.load to unpickle the model)
# These imports ARE used - pickle needs them to reconstruct the model
from save_full_model import WatermarkInjectionNetwork, AlphaEncoder, AlphaDecoder  # noqa: F401


# ============================================================================
# Corner Configuration
# ============================================================================

CORNERS = ['top-left', 'top-right', 'bottom-left', 'bottom-right']


# ============================================================================
# Inference Functions
# ============================================================================

def load_model(model_path, device):
    """Load full model from .pth file."""
    print(f"Loading model: {model_path}")

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model


def add_watermark(image_path, model, device, corner):
    """
    Add watermark to a single image.

    Args:
        image_path: Path to input image
        model: Loaded WatermarkInjectionNetwork model
        device: torch device
        corner: Which corner to place watermark

    Returns:
        watermarked_image: Watermarked image (BGR, uint8)
    """
    # Read image
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_h, original_w = img.shape[:2]

    # Get model's expected image size from template buffer
    # Templates are stored as template_bottom_right, etc.
    template = model.template_bottom_right
    image_size = template.shape[1]  # Template shape is [3, H, W]

    # Preprocess
    img_resized = cv2.resize(img, (image_size, image_size))
    img_normalized = img_resized.astype(np.float32) / 255.0

    # BGR to RGB, HWC to CHW
    img_rgb = img_normalized[:, :, ::-1].copy()
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).float().to(device)

    # Inference
    with torch.no_grad():
        output, _ = model(img_tensor, [corner])

    # Post-process
    watermarked = output.squeeze(0).cpu().numpy()
    watermarked = np.transpose(watermarked, (1, 2, 0))  # CHW to HWC
    watermarked = watermarked[:, :, ::-1]  # RGB to BGR
    watermarked = np.clip(watermarked * 255, 0, 255).astype(np.uint8)

    # Resize back to original dimensions
    watermarked_original_size = cv2.resize(watermarked, (original_w, original_h))

    return watermarked_original_size


def main():
    parser = argparse.ArgumentParser(description='Add watermark to image using trained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to full model (.pth file)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path (optional, auto-generated if not provided)')
    parser.add_argument('--corner', type=str, default=None, choices=CORNERS,
                        help='Corner for watermark (default: random)')

    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load model
    model = load_model(args.model, device)

    # Select corner (random if not specified)
    if args.corner:
        corner = args.corner
    else:
        corner = random.choice(CORNERS)
    print(f"Using corner: {corner}")

    # Process image
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Generate output path if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        corner_suffix = corner.replace('-', '_')
        output_path = input_path.parent / f"{input_path.stem}_watermarked_{corner_suffix}{input_path.suffix}"

    print(f"\nProcessing: {input_path}")

    # Add watermark
    watermarked = add_watermark(input_path, model, device, corner)

    # Save result
    cv2.imwrite(str(output_path), watermarked)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
