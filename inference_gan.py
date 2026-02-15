#!/usr/bin/env python3
"""
Inference Script for Watermark GAN

Usage:
    # Single image
    python inference_gan.py --model ./watermark_model/model_best.pth --input image.jpg --output watermarked.jpg
    
    # Batch processing
    python inference_gan.py --model ./watermark_model/model_best.pth --input_dir ./images --output_dir ./results
    
    # Keep original size
    python inference_gan.py --model ./model.pth --input image.jpg --output out.jpg --keep_size
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm


# ============================================================================
# Model Architecture (must match training)
# ============================================================================

class PretrainedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=None)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
    def forward(self, x):
        features = []
        
        x = self.relu(self.bn1(self.conv1(x)))
        features.append(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        
        return features


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class WatermarkGenerator(nn.Module):
    def __init__(self, residual_scale=1.0):
        super().__init__()
        self.residual_scale = residual_scale
        
        self.encoder = PretrainedEncoder()
        
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        
        self.final_up = nn.ConvTranspose2d(64, 64, 2, stride=2)
        
        self.output = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        
        d4 = self.dec4(features[4], features[3])
        d3 = self.dec3(d4, features[2])
        d2 = self.dec2(d3, features[1])
        d1 = self.dec1(d2, features[0])
        
        d0 = self.final_up(d1)
        
        if d0.shape[2:] != x.shape[2:]:
            d0 = F.interpolate(d0, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        residual = self.output(d0) * self.residual_scale
        watermarked = torch.clamp(x + residual, 0, 1)
        
        return watermarked, residual


# ============================================================================
# Inference Functions
# ============================================================================

def load_model(checkpoint_path: str, device: torch.device) -> WatermarkGenerator:
    """Load trained model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get residual scale from checkpoint
    args = checkpoint.get('args', {})
    residual_scale = args.get('residual_scale', 1.0)
    
    model = WatermarkGenerator(residual_scale=residual_scale)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def process_image(model: WatermarkGenerator, 
                  image_path: str, 
                  device: torch.device,
                  size: int = 256,
                  keep_original_size: bool = True) -> np.ndarray:
    """Add watermark to a single image."""
    
    # Load
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load: {image_path}")
    
    original_size = (img.shape[1], img.shape[0])  # (width, height)
    
    # Preprocess
    img_resized = cv2.resize(img, (size, size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_rgb = img_normalized[:, :, ::-1].copy()
    
    # To tensor
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    img_tensor = img_tensor.to(device)
    
    # Generate
    with torch.no_grad():
        watermarked, _ = model(img_tensor)
    
    # Post-process
    result = watermarked.squeeze(0).cpu().numpy()
    result = np.clip(result.transpose(1, 2, 0), 0, 1)
    result = (result[:, :, ::-1] * 255).astype(np.uint8)  # RGB to BGR
    
    # Resize back if needed
    if keep_original_size:
        result = cv2.resize(result, original_size)
    
    return result


def process_batch(model: WatermarkGenerator,
                  input_dir: str,
                  output_dir: str,
                  device: torch.device,
                  size: int = 256,
                  keep_original_size: bool = True):
    """Process all images in a directory."""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images")
    
    success = 0
    failed = 0
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            result = process_image(model, str(img_path), device, size, keep_original_size)
            
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), result)
            success += 1
            
        except Exception as e:
            print(f"\nFailed: {img_path.name} - {e}")
            failed += 1
    
    print(f"\n✓ Completed: {success} success, {failed} failed")
    print(f"✓ Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Watermark GAN Inference')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str,
                        help='Input image path')
    parser.add_argument('--output', type=str,
                        help='Output image path')
    parser.add_argument('--input_dir', type=str,
                        help='Input directory for batch processing')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for batch processing')
    parser.add_argument('--size', type=int, default=256,
                        help='Processing size (should match training)')
    parser.add_argument('--keep_size', action='store_true',
                        help='Keep original image size')
    
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = load_model(args.model, device)
    print("✓ Model loaded")
    
    # Process
    if args.input and args.output:
        # Single image
        print(f"\nProcessing: {args.input}")
        result = process_image(model, args.input, device, args.size, args.keep_size)
        cv2.imwrite(args.output, result)
        print(f"✓ Saved to: {args.output}")
        
    elif args.input_dir and args.output_dir:
        # Batch
        print(f"\nBatch processing: {args.input_dir}")
        process_batch(model, args.input_dir, args.output_dir, device, 
                     args.size, args.keep_size)
        
    else:
        print("Error: Provide --input/--output OR --input_dir/--output_dir")
        return


if __name__ == '__main__':
    main()
