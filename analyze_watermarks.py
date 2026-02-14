"""
Analyze the watermarks in the training dataset.
This will help understand why the model might not be producing visible watermarks.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_watermark_pair(wm_path, no_wm_path, output_dir):
    """Analyze a single watermark/no-watermark pair."""
    # Load images
    wm_img = cv2.imread(str(wm_path))
    no_wm_img = cv2.imread(str(no_wm_path))

    if wm_img is None or no_wm_img is None:
        print(f"Could not load: {wm_path} or {no_wm_path}")
        return None

    # Resize to same size
    target_size = (512, 512)
    wm_img = cv2.resize(wm_img, target_size)
    no_wm_img = cv2.resize(no_wm_img, target_size)

    # Calculate difference
    diff = cv2.absdiff(wm_img, no_wm_img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Statistics
    stats = {
        'max_diff': diff.max(),
        'mean_diff': diff.mean(),
        'std_diff': diff.std(),
        'pixels_changed_10': (diff_gray > 10).sum() / diff_gray.size * 100,
        'pixels_changed_30': (diff_gray > 30).sum() / diff_gray.size * 100,
        'pixels_changed_50': (diff_gray > 50).sum() / diff_gray.size * 100,
    }

    return wm_img, no_wm_img, diff, stats


def main():
    data_root = Path('./data/wm-nowm/train')
    wm_dir = data_root / 'watermark'
    no_wm_dir = data_root / 'no-watermark'

    output_dir = Path('./watermark_analysis')
    output_dir.mkdir(exist_ok=True)

    # Get matching files
    wm_files = {os.path.splitext(f)[0].lower(): f for f in os.listdir(wm_dir)}
    no_wm_files = {os.path.splitext(f)[0].lower(): f for f in os.listdir(no_wm_dir)}

    matched = []
    for name in wm_files:
        if name in no_wm_files:
            matched.append((wm_dir / wm_files[name], no_wm_dir / no_wm_files[name]))

    print(f"Found {len(matched)} matched pairs")

    if len(matched) == 0:
        print("No matched pairs found!")
        return

    # Analyze first 10 pairs
    all_stats = []
    fig, axes = plt.subplots(min(5, len(matched)), 4, figsize=(16, 4*min(5, len(matched))))

    for i, (wm_path, no_wm_path) in enumerate(matched[:5]):
        result = analyze_watermark_pair(wm_path, no_wm_path, output_dir)
        if result is None:
            continue

        wm_img, no_wm_img, diff, stats = result
        all_stats.append(stats)

        # Convert BGR to RGB for display
        wm_rgb = cv2.cvtColor(wm_img, cv2.COLOR_BGR2RGB)
        no_wm_rgb = cv2.cvtColor(no_wm_img, cv2.COLOR_BGR2RGB)
        diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)

        # Amplify difference for visibility (multiply by 5)
        diff_amplified = np.clip(diff_rgb * 5, 0, 255).astype(np.uint8)

        ax = axes[i] if len(matched) >= 5 else axes
        if len(matched) < 5:
            ax = [axes]

        axes[i, 0].imshow(no_wm_rgb)
        axes[i, 0].set_title(f'Clean (No Watermark)')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(wm_rgb)
        axes[i, 1].set_title(f'Watermarked')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(diff_rgb)
        axes[i, 2].set_title(f'Difference (Raw)\nMax: {stats["max_diff"]:.0f}')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(diff_amplified)
        axes[i, 3].set_title(f'Difference (5x Amplified)\nChanged: {stats["pixels_changed_30"]:.1f}%')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'watermark_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to: {output_dir / 'watermark_comparison.png'}")

    # Print aggregate statistics
    print("\n" + "="*60)
    print("WATERMARK ANALYSIS RESULTS")
    print("="*60)

    if all_stats:
        avg_max = np.mean([s['max_diff'] for s in all_stats])
        avg_mean = np.mean([s['mean_diff'] for s in all_stats])
        avg_changed_30 = np.mean([s['pixels_changed_30'] for s in all_stats])
        avg_changed_50 = np.mean([s['pixels_changed_50'] for s in all_stats])

        print(f"\nAverage Statistics (across {len(all_stats)} pairs):")
        print(f"  Max pixel difference: {avg_max:.1f} (out of 255)")
        print(f"  Mean pixel difference: {avg_mean:.2f}")
        print(f"  Pixels changed >30: {avg_changed_30:.2f}%")
        print(f"  Pixels changed >50: {avg_changed_50:.2f}%")

        print("\n" + "-"*60)
        if avg_mean < 5:
            print("DIAGNOSIS: Watermarks are VERY SUBTLE (nearly invisible)")
            print("  -> The model is learning correctly, but watermarks are too faint")
            print("  -> RECOMMENDATION: Use synthetic watermark approach")
        elif avg_mean < 15:
            print("DIAGNOSIS: Watermarks are SUBTLE")
            print("  -> May need more training or stronger loss weights")
        else:
            print("DIAGNOSIS: Watermarks are VISIBLE")
            print("  -> Model should be able to learn these patterns")

    plt.close()

    # Also analyze what regions have the watermarks
    print("\n" + "="*60)
    print("WATERMARK LOCATION ANALYSIS")
    print("="*60)

    # Check one pair in detail
    if matched:
        wm_path, no_wm_path = matched[0]
        result = analyze_watermark_pair(wm_path, no_wm_path, output_dir)
        if result:
            wm_img, no_wm_img, diff, stats = result
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # Find where the watermark is located
            h, w = diff_gray.shape
            quadrants = {
                'top-left': diff_gray[:h//2, :w//2].mean(),
                'top-right': diff_gray[:h//2, w//2:].mean(),
                'bottom-left': diff_gray[h//2:, :w//2].mean(),
                'bottom-right': diff_gray[h//2:, w//2:].mean(),
                'center': diff_gray[h//4:3*h//4, w//4:3*w//4].mean(),
            }

            print("\nWatermark intensity by region:")
            for region, intensity in sorted(quadrants.items(), key=lambda x: -x[1]):
                bar = '█' * int(intensity / 2)
                print(f"  {region:15s}: {intensity:5.2f} {bar}")


if __name__ == '__main__':
    main()
