"""
Create Synthetic Watermark Dataset

This script takes clean images and creates watermarked versions with visible
AI/UOA watermarks. Run this BEFORE training to create the dataset.

Usage:
    python create_watermark_dataset.py --data_root ./data/wm-nowm

This will:
1. Read clean images from train/no-watermark and valid/no-watermark
2. Create watermarked versions with AI/UOA text
3. Save to train/watermark and valid/watermark (replacing old watermarks)
"""

import argparse
import os
import random
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np


# ============================================================================
# Watermark Generation Functions
# ============================================================================

def create_text_watermark(image, text="AI Generated Image", opacity=0.4, position='bottom-right',
                          font_scale=1.5, color=(255, 255, 255)):
    """Add a text watermark to an image."""
    h, w = image.shape[:2]
    overlay = image.copy()

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

    # Ensure coordinates are valid
    x = max(0, min(x, w - text_w))
    y = max(text_h, min(y, h))

    # Add shadow for visibility
    cv2.putText(overlay, text, (x+2, y+2), font, font_scale, (0, 0, 0), thickness+2)
    # Add main text
    cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness)

    # Blend with original
    watermarked = cv2.addWeighted(image, 1-opacity, overlay, opacity, 0)

    return watermarked


def create_diagonal_text_watermark(image, text="AI Image", opacity=0.15,
                                    font_scale=1.5, color=(200, 200, 200)):
    """Add diagonal repeating text watermark across the entire image."""
    h, w = image.shape[:2]
    overlay = np.zeros_like(image)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Create diagonal pattern
    spacing_x = text_w + 80
    spacing_y = text_h + 60

    for y_pos in range(-h, h*2, spacing_y):
        for x_pos in range(-w, w*2, spacing_x):
            offset = (y_pos // spacing_y) * (spacing_x // 2)
            cv2.putText(overlay, text, (x_pos + offset, y_pos), font, font_scale, color, thickness)

    # Rotate the overlay
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -30, 1.0)
    overlay = cv2.warpAffine(overlay, rotation_matrix, (w, h))

    # Blend
    watermarked = cv2.addWeighted(image, 1.0, overlay, opacity, 0)

    return watermarked


def create_logo_watermark(image, logo_size=80, opacity=0.5, position='bottom-right'):
    """Add a circular logo watermark with UOA/AI branding."""
    h, w = image.shape[:2]
    overlay = image.copy()

    # Create a simple circular logo
    logo = np.zeros((logo_size, logo_size, 3), dtype=np.uint8)

    # Draw concentric circles
    center = (logo_size // 2, logo_size // 2)
    cv2.circle(logo, center, logo_size // 2 - 3, (255, 255, 255), -1)
    cv2.circle(logo, center, logo_size // 2 - 10, (70, 130, 180), -1)
    cv2.circle(logo, center, logo_size // 3, (255, 255, 255), -1)

    # Add text in center
    text = random.choice(["AI", "UOA"])
    font_scale = 0.6 if text == "UOA" else 0.9
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    cv2.putText(logo, text, (logo_size//2 - tw//2, logo_size//2 + th//2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 50, 50), 2)

    # Calculate position
    padding = 15
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

    # Blend logo
    for c in range(3):
        overlay[y:y+logo_size, x:x+logo_size, c] = (
            overlay[y:y+logo_size, x:x+logo_size, c] * (1 - opacity) +
            logo[:, :, c] * opacity
        ).astype(np.uint8)

    return overlay


def create_corner_watermark(image, text="AI", opacity=0.6):
    """Add small corner watermark."""
    h, w = image.shape[:2]
    overlay = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Bottom right corner
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    x = w - text_w - 10
    y = h - 10

    # Draw background rectangle
    cv2.rectangle(overlay, (x-5, y-text_h-5), (x+text_w+5, y+5), (0, 0, 0), -1)
    cv2.putText(overlay, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    watermarked = cv2.addWeighted(image, 1-opacity, overlay, opacity, 0)
    return watermarked


def apply_watermark(image, watermark_style=None):
    """Apply watermark to image with specified or random style."""

    # Watermark texts for UOA AI project
    AI_TEXTS = ['AI Generated Image', 'AI Image', 'UOA']
    AI_TEXTS_SHORT = ['AI Image', 'UOA', 'AI']

    if watermark_style is None:
        watermark_style = random.choice(['text', 'diagonal', 'logo', 'combined', 'corner'])

    if watermark_style == 'text':
        positions = ['bottom-right', 'bottom-left', 'top-right', 'center']
        return create_text_watermark(
            image,
            text=random.choice(AI_TEXTS),
            opacity=random.uniform(0.35, 0.55),
            position=random.choice(positions),
            font_scale=random.uniform(1.0, 1.8)
        )

    elif watermark_style == 'diagonal':
        return create_diagonal_text_watermark(
            image,
            text=random.choice(AI_TEXTS_SHORT),
            opacity=random.uniform(0.12, 0.22)
        )

    elif watermark_style == 'logo':
        positions = ['bottom-right', 'bottom-left', 'top-right']
        return create_logo_watermark(
            image,
            logo_size=random.randint(60, 100),
            opacity=random.uniform(0.4, 0.6),
            position=random.choice(positions)
        )

    elif watermark_style == 'corner':
        return create_corner_watermark(
            image,
            text=random.choice(AI_TEXTS_SHORT),
            opacity=random.uniform(0.5, 0.7)
        )

    elif watermark_style == 'combined':
        img = image.copy()
        # Add diagonal background
        if random.random() > 0.4:
            img = create_diagonal_text_watermark(img, text=random.choice(AI_TEXTS_SHORT), opacity=random.uniform(0.08, 0.15))
        # Add corner text or logo
        if random.random() > 0.5:
            img = create_logo_watermark(img, opacity=random.uniform(0.35, 0.5))
        else:
            img = create_text_watermark(img, text=random.choice(AI_TEXTS), opacity=random.uniform(0.35, 0.5))
        return img

    return image


# ============================================================================
# Dataset Creation
# ============================================================================

def process_folder(input_dir, output_dir, watermark_style=None):
    """Process all images in a folder and create watermarked versions."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))

    print(f"\nProcessing {len(image_files)} images from {input_dir}")
    print(f"Output directory: {output_dir}")

    success_count = 0
    error_count = 0

    for img_path in tqdm(image_files, desc="Creating watermarks"):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                error_count += 1
                continue

            # Apply watermark
            watermarked = apply_watermark(img, watermark_style)

            # Save with same filename to output directory
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), watermarked)

            success_count += 1

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            error_count += 1

    print(f"Completed: {success_count} success, {error_count} errors")
    return success_count, error_count


def main():
    parser = argparse.ArgumentParser(description='Create synthetic watermark dataset')
    parser.add_argument('--data_root', type=str, default='./data/wm-nowm',
                        help='Root directory containing train/valid folders')
    parser.add_argument('--watermark_style', type=str, default=None,
                        choices=['text', 'diagonal', 'logo', 'corner', 'combined', None],
                        help='Watermark style (default: random mix)')
    parser.add_argument('--backup', action='store_true',
                        help='Backup existing watermark folder before replacing')

    args = parser.parse_args()

    data_root = Path(args.data_root)

    print("="*60)
    print("SYNTHETIC WATERMARK DATASET CREATOR")
    print("="*60)
    print(f"\nData root: {data_root}")
    print(f"Watermark style: {args.watermark_style or 'random mix'}")
    print(f"\nWatermark texts: 'AI Generated Image', 'AI Image', 'UOA'")

    # Process training data
    train_input = data_root / 'train' / 'no-watermark'
    train_output = data_root / 'train' / 'watermark'

    if train_input.exists():
        if args.backup and train_output.exists():
            import shutil
            backup_path = data_root / 'train' / 'watermark_backup'
            print(f"\nBacking up existing watermarks to: {backup_path}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.move(str(train_output), str(backup_path))

        print("\n" + "-"*60)
        print("PROCESSING TRAINING DATA")
        print("-"*60)
        process_folder(train_input, train_output, args.watermark_style)
    else:
        print(f"\nWarning: Training input not found: {train_input}")

    # Process validation data
    valid_input = data_root / 'valid' / 'no-watermark'
    valid_output = data_root / 'valid' / 'watermark'

    if valid_input.exists():
        if args.backup and valid_output.exists():
            import shutil
            backup_path = data_root / 'valid' / 'watermark_backup'
            print(f"\nBacking up existing watermarks to: {backup_path}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.move(str(valid_output), str(backup_path))

        print("\n" + "-"*60)
        print("PROCESSING VALIDATION DATA")
        print("-"*60)
        process_folder(valid_input, valid_output, args.watermark_style)
    else:
        print(f"\nWarning: Validation input not found: {valid_input}")

    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE!")
    print("="*60)
    print("\nNow you can train with the original training script:")
    print(f"""
    python train_watermark_addition.py \\
        --data_root {args.data_root} \\
        --epochs 100 \\
        --batch_size 8 \\
        --checkpoint_dir ./checkpoints \\
        --width 512 \\
        --height 512
    """)


if __name__ == '__main__':
    main()
