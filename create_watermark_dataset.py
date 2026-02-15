"""
Create Synthetic Watermark Dataset

This script takes clean images and creates watermarked versions with "UOA" text
in one of 4 corners (random position per image).

Usage:
    python create_watermark_dataset.py --data_root ./data/wm-nowm

This will:
1. Read clean images from train/no-watermark and valid/no-watermark
2. Create watermarked versions with "UOA" in random corner
3. Save to train/watermark and valid/watermark
"""

import argparse
import random
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np


def add_uoa_watermark(image, position=None, opacity=0.5):
    """
    Add semi-transparent "UOA" watermark to one of 4 corners.
    Only the watermark text is blended, rest of image unchanged.

    Args:
        image: Input image (BGR)
        position: 'top-left', 'top-right', 'bottom-left', 'bottom-right' or None (random)
        opacity: Watermark opacity (0.0 = invisible, 1.0 = solid)

    Returns:
        Watermarked image
    """
    h, w = image.shape[:2]
    result = image.copy()

    # Create transparent overlay (black background)
    overlay = np.zeros_like(image, dtype=np.uint8)

    # Watermark settings
    text = "UOA"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0  # Larger text
    thickness = 4
    color = (255, 255, 255)  # White
    shadow_color = (80, 80, 80)  # Gray shadow
    padding = 25

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Random corner if not specified
    if position is None:
        position = random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])

    # Calculate position
    if position == 'top-left':
        x = padding
        y = text_h + padding
    elif position == 'top-right':
        x = w - text_w - padding
        y = text_h + padding
    elif position == 'bottom-left':
        x = padding
        y = h - padding
    elif position == 'bottom-right':
        x = w - text_w - padding
        y = h - padding
    else:
        x = w - text_w - padding
        y = h - padding

    # Ensure valid coordinates
    x = max(0, min(x, w - text_w))
    y = max(text_h, min(y, h))

    # Draw shadow on overlay (offset by 2 pixels)
    cv2.putText(overlay, text, (x + 2, y + 2), font, font_scale, shadow_color, thickness + 2)

    # Draw main text on overlay
    cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness)

    # Create mask where overlay has content (non-zero pixels)
    gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Blend only where mask is active (watermark area)
    # watermark_region = original * (1 - opacity) + overlay * opacity
    for c in range(3):
        result[:, :, c] = np.where(
            mask > 0,
            (image[:, :, c] * (1 - opacity) + overlay[:, :, c] * opacity).astype(np.uint8),
            image[:, :, c]
        )

    return result


def process_folder(input_dir, output_dir):
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
    print("Watermark: 'UOA' in random corner (top-left, top-right, bottom-left, bottom-right)")

    success_count = 0
    error_count = 0
    position_counts = {'top-left': 0, 'top-right': 0, 'bottom-left': 0, 'bottom-right': 0}

    for img_path in tqdm(image_files, desc="Creating watermarks"):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                error_count += 1
                continue

            # Random corner
            position = random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])
            position_counts[position] += 1

            # Apply watermark
            watermarked = add_uoa_watermark(img, position)

            # Save with same filename to output directory
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), watermarked)

            success_count += 1

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            error_count += 1

    print(f"\nCompleted: {success_count} success, {error_count} errors")
    print(f"Position distribution:")
    for pos, count in position_counts.items():
        print(f"  {pos}: {count} images")

    return success_count, error_count


def main():
    parser = argparse.ArgumentParser(description='Create UOA watermark dataset')
    parser.add_argument('--data_root', type=str, default='./data/wm-nowm',
                        help='Root directory containing train/valid folders')
    parser.add_argument('--backup', action='store_true',
                        help='Backup existing watermark folder before replacing')

    args = parser.parse_args()

    data_root = Path(args.data_root)

    print("=" * 60)
    print("UOA WATERMARK DATASET CREATOR")
    print("=" * 60)
    print(f"\nData root: {data_root}")
    print("Watermark: 'UOA' text in random corner")

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

        print("\n" + "-" * 60)
        print("PROCESSING TRAINING DATA")
        print("-" * 60)
        process_folder(train_input, train_output)
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

        print("\n" + "-" * 60)
        print("PROCESSING VALIDATION DATA")
        print("-" * 60)
        process_folder(valid_input, valid_output)
    else:
        print(f"\nWarning: Validation input not found: {valid_input}")

    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETE!")
    print("=" * 60)
    print("\nNow train with:")
    print(f"""
    python train_watermark_addition.py \\
        --data_root {args.data_root} \\
        --epochs 100 \\
        --batch_size 8 \\
        --max_images 5000
    """)


if __name__ == '__main__':
    main()
