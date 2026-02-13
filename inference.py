"""
Inference script for Watermark Addition Model

Usage:
    python inference.py --model checkpoints/watermark_addition_best.pth --input image.jpg --output watermarked.jpg
    python inference.py --model watermark_addition_model.pth --input ./images/ --output ./output/
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Model Definition (must match training)
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class WatermarkAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ConvBlock(64, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = ConvBlock(32, 16)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Decoder
        self.dec1 = ConvBlock(16, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(64, 32)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(32, 16)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Output
        self.output = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.pool1(self.enc1(x))
        e2 = self.pool2(self.enc2(e1))
        e3 = self.pool3(self.enc3(e2))

        d1 = self.up1(self.dec1(e3))
        d2 = self.up2(self.dec2(d1))
        d3 = self.up3(self.dec3(d2))

        return self.sigmoid(self.output(d3))


# ============================================================================
# Inference Functions
# ============================================================================

def load_model(model_path, device):
    """Load trained model from checkpoint."""
    model = WatermarkAutoencoder().to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle both full checkpoint and state_dict only
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Model loaded from: {model_path}")
    return model


def add_watermark(image_path, model, device, width=512, height=512):
    """
    Add watermark to a single image.

    Args:
        image_path: Path to input image
        model: Loaded model
        device: torch device
        width, height: Model input size

    Returns:
        watermarked_original_size: Watermarked image at original dimensions (BGR)
    """
    # Read image
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_h, original_w = img.shape[:2]

    # Preprocess
    img_resized = cv2.resize(img, (width, height))
    img_normalized = img_resized.astype(np.float32) / 255.0

    # BGR to RGB, HWC to CHW
    img_rgb = img_normalized[:, :, ::-1]
    img_chw = np.ascontiguousarray(np.transpose(img_rgb, (2, 0, 1)))
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)

    # Post-process
    watermarked = output.squeeze(0).cpu().numpy()
    watermarked = np.transpose(watermarked, (1, 2, 0))  # CHW to HWC
    watermarked = watermarked[:, :, ::-1]  # RGB to BGR
    watermarked = (watermarked * 255).astype(np.uint8)

    # Resize back to original dimensions
    watermarked_original_size = cv2.resize(watermarked, (original_w, original_h))

    return watermarked_original_size


def process_single_image(input_path, output_path, model, device):
    """Process a single image."""
    print(f"Processing: {input_path}")

    watermarked = add_watermark(input_path, model, device)
    cv2.imwrite(str(output_path), watermarked)

    print(f"Saved: {output_path}")


def process_directory(input_dir, output_dir, model, device):
    """Process all images in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

    print(f"Found {len(image_files)} images in {input_dir}")

    for img_path in image_files:
        output_path = output_dir / f"watermarked_{img_path.name}"
        try:
            process_single_image(img_path, output_path, model, device)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"\nDone! Processed {len(image_files)} images.")


def main():
    parser = argparse.ArgumentParser(description='Add watermark to images using trained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pth file)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image path or directory')
    parser.add_argument('--width', type=int, default=512,
                        help='Model input width (default: 512)')
    parser.add_argument('--height', type=int, default=512,
                        help='Model input height (default: 512)')

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

    # Process input
    input_path = Path(args.input)

    if input_path.is_file():
        # Single image
        process_single_image(input_path, args.output, model, device)
    elif input_path.is_dir():
        # Directory of images
        process_directory(input_path, args.output, model, device)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == '__main__':
    main()
