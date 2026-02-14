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


class DoubleConvBlock(nn.Module):
    """Two consecutive conv blocks for better feature extraction."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class WatermarkAutoencoder(nn.Module):
    """
    Deep U-Net with RESIDUAL LEARNING for watermark addition.
    Model predicts watermark overlay, adds to input.
    """

    def __init__(self, residual_scale=0.5):
        super().__init__()

        self.residual_scale = residual_scale  # Must match training

        # Encoder path
        self.enc1 = DoubleConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = DoubleConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = DoubleConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = DoubleConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = DoubleConvBlock(512, 512)

        # Decoder path with skip connections
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConvBlock(512 + 512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConvBlock(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConvBlock(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConvBlock(64 + 64, 64)

        # Output layer - predicts residual
        self.output = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # RESIDUAL: predict watermark overlay and add to input
        residual = self.tanh(self.output(d1)) * self.residual_scale
        out = torch.clamp(x + residual, 0, 1)

        return out


# ============================================================================
# Inference Functions
# ============================================================================

def load_model(model_path, device):
    """Load trained model from checkpoint or full model file."""
    # Load checkpoint/model
    loaded = torch.load(model_path, map_location=device, weights_only=False)

    # Check if it's a full model or just weights/checkpoint
    if isinstance(loaded, nn.Module):
        # Full model saved with torch.save(model, path)
        model = loaded.to(device)
        print(f"Loaded full model from: {model_path}")
    elif isinstance(loaded, dict):
        # Checkpoint or state_dict
        model = WatermarkAutoencoder().to(device)
        if 'model_state_dict' in loaded:
            model.load_state_dict(loaded['model_state_dict'])
            print(f"Loaded checkpoint from epoch {loaded.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(loaded)
            print(f"Loaded state_dict from: {model_path}")
    else:
        raise ValueError(f"Unknown model format in {model_path}")

    model.eval()
    print(f"Model architecture:\n{model}\n")
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


def process_single_image(input_path, output_path, model, device, width=512, height=512):
    """Process a single image."""
    print(f"Processing: {input_path}")

    watermarked = add_watermark(input_path, model, device, width, height)
    cv2.imwrite(str(output_path), watermarked)

    print(f"Saved: {output_path}")


def process_directory(input_dir, output_dir, model, device, width=512, height=512):
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
            process_single_image(img_path, output_path, model, device, width, height)
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
                        help='Model input width (default: 512, use same as training)')
    parser.add_argument('--height', type=int, default=512,
                        help='Model input height (default: 512, use same as training)')

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
        process_single_image(input_path, args.output, model, device, args.width, args.height)
    elif input_path.is_dir():
        # Directory of images
        process_directory(input_path, args.output, model, device, args.width, args.height)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == '__main__':
    main()
