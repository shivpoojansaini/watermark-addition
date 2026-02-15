"""
Synthetic Watermark Addition Training

This approach creates visible watermarks programmatically, guaranteeing
the model learns to add obvious, visible watermarks.

Usage:
    python train_synthetic_watermark.py --image_dir ./data/clean_images --epochs 50
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================================
# Watermark Generation Functions
# ============================================================================

def create_text_watermark(image, text="SAMPLE", opacity=0.3, position='bottom-right',
                          font_scale=1.5, color=(255, 255, 255)):
    """Add a text watermark to an image."""
    h, w = image.shape[:2]
    overlay = image.copy()

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate position
    padding = 20
    if position == 'bottom-right':
        x = w - text_w - padding
        y = h - padding
    elif position == 'bottom-left':
        x = padding
        y = h - padding
    elif position == 'top-right':
        x = w - text_w - padding
        y = text_h + padding
    elif position == 'top-left':
        x = padding
        y = text_h + padding
    elif position == 'center':
        x = (w - text_w) // 2
        y = (h + text_h) // 2
    else:
        x = w - text_w - padding
        y = h - padding

    # Add shadow for visibility
    cv2.putText(overlay, text, (x+2, y+2), font, font_scale, (0, 0, 0), thickness+1)
    # Add main text
    cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness)

    # Blend with original
    watermarked = cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)
    # Make watermark more visible by using max blend for text area
    watermarked = cv2.addWeighted(image, 1-opacity, overlay, opacity, 0)

    return watermarked


def create_diagonal_text_watermark(image, text="WATERMARK", opacity=0.15,
                                    font_scale=2.0, color=(200, 200, 200)):
    """Add diagonal repeating text watermark across the entire image."""
    h, w = image.shape[:2]
    overlay = np.zeros_like(image)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 3

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Create diagonal pattern
    spacing_x = text_w + 100
    spacing_y = text_h + 80

    for y in range(-h, h*2, spacing_y):
        for x in range(-w, w*2, spacing_x):
            # Offset every other row
            offset = (y // spacing_y) * (spacing_x // 2)
            cv2.putText(overlay, text, (x + offset, y), font, font_scale, color, thickness)

    # Rotate the overlay
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -30, 1.0)
    overlay = cv2.warpAffine(overlay, rotation_matrix, (w, h))

    # Blend
    watermarked = cv2.addWeighted(image, 1.0, overlay, opacity, 0)

    return watermarked


def create_logo_watermark(image, logo_size=100, opacity=0.5, position='bottom-right'):
    """Add a circular logo watermark with UOA/AI branding."""
    h, w = image.shape[:2]
    overlay = image.copy()

    # Create a simple circular logo
    logo = np.zeros((logo_size, logo_size, 3), dtype=np.uint8)

    # Draw concentric circles
    center = (logo_size // 2, logo_size // 2)
    cv2.circle(logo, center, logo_size // 2 - 5, (255, 255, 255), -1)
    cv2.circle(logo, center, logo_size // 2 - 15, (70, 130, 180), -1)  # Steel blue
    cv2.circle(logo, center, logo_size // 3, (255, 255, 255), -1)

    # Add "AI" text in center
    text = random.choice(["AI", "UOA"])
    font_scale = 0.8 if text == "UOA" else 1.2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    cv2.putText(logo, text, (logo_size//2 - tw//2, logo_size//2 + th//2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 50, 50), 2)

    # Calculate position
    padding = 20
    if position == 'bottom-right':
        x = w - logo_size - padding
        y = h - logo_size - padding
    elif position == 'bottom-left':
        x = padding
        y = h - logo_size - padding
    elif position == 'top-right':
        x = w - logo_size - padding
        y = padding
    elif position == 'center':
        x = (w - logo_size) // 2
        y = (h - logo_size) // 2
    else:
        x = w - logo_size - padding
        y = h - logo_size - padding

    # Ensure valid coordinates
    x = max(0, min(x, w - logo_size))
    y = max(0, min(y, h - logo_size))

    # Create mask for blending
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y:y+logo_size, x:x+logo_size] = opacity

    # Blend logo
    for c in range(3):
        overlay[y:y+logo_size, x:x+logo_size, c] = (
            overlay[y:y+logo_size, x:x+logo_size, c] * (1 - opacity) +
            logo[:, :, c] * opacity
        ).astype(np.uint8)

    return overlay


def create_border_watermark(image, border_width=10, color=(255, 200, 100), opacity=0.7):
    """Add a colored border as watermark."""
    h, w = image.shape[:2]
    overlay = image.copy()

    # Draw border
    cv2.rectangle(overlay, (0, 0), (w, h), color, border_width)

    # Also add corner markers
    corner_size = 30
    # Top-left
    cv2.line(overlay, (0, corner_size), (corner_size, 0), color, 3)
    # Top-right
    cv2.line(overlay, (w-corner_size, 0), (w, corner_size), color, 3)
    # Bottom-left
    cv2.line(overlay, (0, h-corner_size), (corner_size, h), color, 3)
    # Bottom-right
    cv2.line(overlay, (w-corner_size, h), (w, h-corner_size), color, 3)

    watermarked = cv2.addWeighted(image, 1-opacity, overlay, opacity, 0)
    return watermarked


def apply_random_watermark(image, watermark_type=None):
    """Apply a random watermark type to the image."""
    if watermark_type is None:
        watermark_type = random.choice(['text', 'diagonal', 'logo', 'border', 'combined'])

    # Custom watermark texts for UOA AI project
    AI_TEXTS = ['AI Generated Image', 'AI Image', 'UOA']
    AI_TEXTS_SHORT = ['AI Image', 'UOA']

    if watermark_type == 'text':
        positions = ['bottom-right', 'bottom-left', 'top-right', 'center']
        return create_text_watermark(
            image,
            text=random.choice(AI_TEXTS),
            opacity=random.uniform(0.3, 0.6),
            position=random.choice(positions),
            font_scale=random.uniform(1.0, 2.0)
        )

    elif watermark_type == 'diagonal':
        return create_diagonal_text_watermark(
            image,
            text=random.choice(AI_TEXTS_SHORT),
            opacity=random.uniform(0.1, 0.25)
        )

    elif watermark_type == 'logo':
        positions = ['bottom-right', 'bottom-left', 'top-right', 'center']
        return create_logo_watermark(
            image,
            logo_size=random.randint(60, 120),
            opacity=random.uniform(0.4, 0.7),
            position=random.choice(positions)
        )

    elif watermark_type == 'border':
        colors = [(255, 200, 100), (100, 200, 255), (200, 100, 255), (255, 100, 100)]
        return create_border_watermark(
            image,
            border_width=random.randint(5, 15),
            color=random.choice(colors),
            opacity=random.uniform(0.5, 0.8)
        )

    elif watermark_type == 'combined':
        # Apply multiple watermarks
        img = image.copy()
        if random.random() > 0.5:
            img = create_diagonal_text_watermark(img, text=random.choice(AI_TEXTS_SHORT), opacity=random.uniform(0.08, 0.15))
        if random.random() > 0.3:
            img = create_logo_watermark(img, opacity=random.uniform(0.3, 0.5))
        if random.random() > 0.5:
            img = create_text_watermark(img, text=random.choice(AI_TEXTS), opacity=random.uniform(0.3, 0.5))
        return img

    return image


# ============================================================================
# Model Definition
# ============================================================================

class DoubleConvBlock(nn.Module):
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
    """U-Net with residual learning for watermark addition."""

    def __init__(self, residual_scale=0.5):
        super().__init__()
        self.residual_scale = residual_scale

        # Encoder
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

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConvBlock(128, 64)

        # Output
        self.output = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Residual output
        residual = self.tanh(self.output(d1)) * self.residual_scale
        out = torch.clamp(x + residual, 0, 1)

        return out


# ============================================================================
# Dataset
# ============================================================================

class SyntheticWatermarkDataset(Dataset):
    """Dataset that generates watermarked images on-the-fly."""

    def __init__(self, image_dir, width=512, height=512, watermark_type=None):
        self.image_dir = Path(image_dir)
        self.width = width
        self.height = height
        self.watermark_type = watermark_type

        # Find all images
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(self.image_dir.glob(f'*{ext}'))
            self.image_files.extend(self.image_dir.glob(f'*{ext.upper()}'))

        print(f"Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))

        if img is None:
            # Return a random valid image if this one fails
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Resize
        img = cv2.resize(img, (self.width, self.height))

        # Create watermarked version
        watermarked = apply_random_watermark(img, self.watermark_type)

        # Normalize and convert
        clean = img.astype(np.float32) / 255.0
        watermarked = watermarked.astype(np.float32) / 255.0

        # BGR to RGB, HWC to CHW
        clean = np.ascontiguousarray(clean[:, :, ::-1].transpose(2, 0, 1))
        watermarked = np.ascontiguousarray(watermarked[:, :, ::-1].transpose(2, 0, 1))

        return (
            torch.from_numpy(clean),
            torch.from_numpy(watermarked)
        )


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, val_loader, device, args):
    """Train the model."""
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')

        for clean, watermarked in pbar:
            clean = clean.to(device)
            watermarked = watermarked.to(device)

            optimizer.zero_grad()
            output = model(clean)

            # Combined loss
            mse_loss = criterion(output, watermarked)
            l1_loss = l1_criterion(output, watermarked)
            loss = mse_loss + 0.5 * l1_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for clean, watermarked in val_loader:
                clean = clean.to(device)
                watermarked = watermarked.to(device)
                output = model(clean)
                loss = criterion(output, watermarked) + 0.5 * l1_criterion(output, watermarked)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, args.output_model)
            print(f'  Saved best model (loss: {val_loss:.6f})')

        scheduler.step(val_loss)

    return history


def plot_results(model, val_loader, device, save_path='synthetic_results.png'):
    """Plot sample results."""
    model.eval()

    # Get a batch
    clean, watermarked = next(iter(val_loader))
    clean = clean.to(device)

    with torch.no_grad():
        output = model(clean)

    # Convert to numpy
    clean = clean.cpu().numpy()
    watermarked = watermarked.numpy()
    output = output.cpu().numpy()

    # Plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for i in range(4):
        # Clean
        img = clean[i].transpose(1, 2, 0)
        axes[0, i].imshow(img)
        axes[0, i].set_title('Clean Input')
        axes[0, i].axis('off')

        # Target watermarked
        img = watermarked[i].transpose(1, 2, 0)
        axes[1, i].imshow(img)
        axes[1, i].set_title('Target (Synthetic WM)')
        axes[1, i].axis('off')

        # Model output
        img = output[i].transpose(1, 2, 0)
        axes[2, i].imshow(img)
        axes[2, i].set_title('Model Output')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'Saved results to: {save_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train watermark addition with synthetic watermarks')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing clean images')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--output_model', type=str, default='synthetic_watermark_model.pth')
    parser.add_argument('--watermark_type', type=str, default=None,
                        choices=['text', 'diagonal', 'logo', 'border', 'combined', None],
                        help='Type of watermark (default: random)')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Dataset - use the non-watermarked images as clean source
    dataset = SyntheticWatermarkDataset(
        args.image_dir,
        args.width,
        args.height,
        args.watermark_type
    )

    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')

    # Model
    model = WatermarkAutoencoder(residual_scale=0.5).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'\nModel architecture:\n{model}\n')

    # Train
    history = train_model(model, train_loader, val_loader, device, args)

    # Plot results
    plot_results(model, val_loader, device)

    print(f'\nTraining complete! Model saved to: {args.output_model}')
    print(f'Use with inference.py:')
    print(f'  python inference.py --model {args.output_model} --input image.jpg --output watermarked.jpg')


if __name__ == '__main__':
    main()
