#!/usr/bin/env python3
"""
Quick Start Script for Watermark GAN Training

Just point it at your data and go!

Usage:
    python quick_train.py --data_root ./data/wm-nowm
    
    # Or with custom paths:
    python quick_train.py --clean ./clean_images --watermarked ./watermarked_images
    
    # Fast test (5 epochs):
    python quick_train.py --data_root ./data/wm-nowm --fast
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Quick Train Watermark GAN')
    
    # Data
    parser.add_argument('--data_root', type=str, 
                        help='Root dir with train/watermark and train/no-watermark')
    parser.add_argument('--clean', type=str, help='Clean images directory')
    parser.add_argument('--watermarked', type=str, help='Watermarked images directory')
    
    # Quick options
    parser.add_argument('--fast', action='store_true', help='Fast test (5 epochs)')
    parser.add_argument('--medium', action='store_true', help='Medium training (30 epochs)')
    parser.add_argument('--epochs', type=int, default=None, help='Custom epochs')
    parser.add_argument('--max_pairs', type=int, default=None, 
                        help='Limit number of image pairs (e.g., 1000)')
    
    # Output
    parser.add_argument('--output', type=str, default='./watermark_model',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Validate paths
    if args.data_root:
        clean_dir = os.path.join(args.data_root, 'train', 'no-watermark')
        wm_dir = os.path.join(args.data_root, 'train', 'watermark')
    elif args.clean and args.watermarked:
        clean_dir = args.clean
        wm_dir = args.watermarked
    else:
        print("Error: Provide --data_root OR both --clean and --watermarked")
        sys.exit(1)
    
    if not os.path.exists(clean_dir):
        print(f"Error: Clean directory not found: {clean_dir}")
        sys.exit(1)
    if not os.path.exists(wm_dir):
        print(f"Error: Watermarked directory not found: {wm_dir}")
        sys.exit(1)
    
    # Determine epochs
    if args.epochs:
        epochs = args.epochs
    elif args.fast:
        epochs = 5
    elif args.medium:
        epochs = 30
    else:
        epochs = 100
    
    # Build command
    cmd = f"""python train_watermark_gan.py \\
    --clean_dir "{clean_dir}" \\
    --watermarked_dir "{wm_dir}" \\
    --epochs {epochs} \\
    --batch_size 8 \\
    --image_size 256 \\
    --output_dir "{args.output}" \\
    --checkpoint_path "{args.output}/model.pth" \\
    --freeze_encoder_epochs {min(10, epochs // 3)} \\
    --vis_every {max(1, epochs // 10)}"""
    
    # Add max_pairs if specified
    if args.max_pairs:
        cmd += f" \\\n    --max_pairs {args.max_pairs}"
    
    print("="*60)
    print("WATERMARK GAN - QUICK TRAIN")
    print("="*60)
    print(f"\nClean images: {clean_dir}")
    print(f"Watermarked images: {wm_dir}")
    print(f"Epochs: {epochs}")
    if args.max_pairs:
        print(f"Max pairs: {args.max_pairs}")
    print(f"Output: {args.output}")
    print("\nRunning command:")
    print(cmd)
    print("\n" + "="*60 + "\n")
    
    os.system(cmd)


if __name__ == '__main__':
    main()
