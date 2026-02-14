"""
Generate architecture diagrams for the Watermark Addition Model report.
Creates publication-quality figures for the interim report.

Usage:
    python generate_diagrams.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_unet_architecture_diagram():
    """Create a U-Net architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Watermark Addition U-Net Architecture', fontsize=16, fontweight='bold', pad=20)

    # Colors
    encoder_color = '#3498db'  # Blue
    decoder_color = '#e74c3c'  # Red
    bottleneck_color = '#9b59b6'  # Purple
    skip_color = '#2ecc71'  # Green
    output_color = '#f39c12'  # Orange

    # Encoder blocks (left side, going down)
    encoder_blocks = [
        {'name': 'Input\n3×512×512', 'x': 1, 'y': 8.5, 'w': 1.5, 'h': 0.8},
        {'name': 'Enc1\n64×512×512', 'x': 1, 'y': 7.2, 'w': 1.5, 'h': 0.8},
        {'name': 'Enc2\n128×256×256', 'x': 1, 'y': 5.9, 'w': 1.5, 'h': 0.8},
        {'name': 'Enc3\n256×128×128', 'x': 1, 'y': 4.6, 'w': 1.5, 'h': 0.8},
        {'name': 'Enc4\n512×64×64', 'x': 1, 'y': 3.3, 'w': 1.5, 'h': 0.8},
    ]

    # Bottleneck
    bottleneck = {'name': 'Bottleneck\n512×32×32', 'x': 7.25, 'y': 2, 'w': 1.5, 'h': 0.8}

    # Decoder blocks (right side, going up)
    decoder_blocks = [
        {'name': 'Dec4\n512×64×64', 'x': 13.5, 'y': 3.3, 'w': 1.5, 'h': 0.8},
        {'name': 'Dec3\n256×128×128', 'x': 13.5, 'y': 4.6, 'w': 1.5, 'h': 0.8},
        {'name': 'Dec2\n128×256×256', 'x': 13.5, 'y': 5.9, 'w': 1.5, 'h': 0.8},
        {'name': 'Dec1\n64×512×512', 'x': 13.5, 'y': 7.2, 'w': 1.5, 'h': 0.8},
        {'name': 'Output\n3×512×512', 'x': 13.5, 'y': 8.5, 'w': 1.5, 'h': 0.8},
    ]

    # Draw encoder blocks
    for i, block in enumerate(encoder_blocks):
        color = '#95a5a6' if i == 0 else encoder_color
        rect = FancyBboxPatch((block['x'], block['y']), block['w'], block['h'],
                               boxstyle="round,pad=0.05", facecolor=color,
                               edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(block['x'] + block['w']/2, block['y'] + block['h']/2, block['name'],
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Draw bottleneck
    rect = FancyBboxPatch((bottleneck['x'], bottleneck['y']), bottleneck['w'], bottleneck['h'],
                           boxstyle="round,pad=0.05", facecolor=bottleneck_color,
                           edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(bottleneck['x'] + bottleneck['w']/2, bottleneck['y'] + bottleneck['h']/2,
            bottleneck['name'], ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Draw decoder blocks
    for i, block in enumerate(decoder_blocks):
        color = output_color if i == 4 else decoder_color
        rect = FancyBboxPatch((block['x'], block['y']), block['w'], block['h'],
                               boxstyle="round,pad=0.05", facecolor=color,
                               edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(block['x'] + block['w']/2, block['y'] + block['h']/2, block['name'],
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Draw encoder arrows (down)
    for i in range(len(encoder_blocks) - 1):
        ax.annotate('', xy=(1.75, encoder_blocks[i+1]['y'] + encoder_blocks[i+1]['h']),
                    xytext=(1.75, encoder_blocks[i]['y']),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.text(2.7, (encoder_blocks[i]['y'] + encoder_blocks[i+1]['y'] + encoder_blocks[i+1]['h'])/2,
                'MaxPool\n2×2', fontsize=7, ha='center', va='center')

    # Encoder to bottleneck
    ax.annotate('', xy=(bottleneck['x'], bottleneck['y'] + bottleneck['h']/2),
                xytext=(encoder_blocks[-1]['x'] + encoder_blocks[-1]['w'], encoder_blocks[-1]['y'] + encoder_blocks[-1]['h']/2),
                arrowprops=dict(arrowstyle='->', color='black', lw=2, connectionstyle="arc3,rad=-0.3"))

    # Bottleneck to decoder
    ax.annotate('', xy=(decoder_blocks[0]['x'], decoder_blocks[0]['y'] + decoder_blocks[0]['h']/2),
                xytext=(bottleneck['x'] + bottleneck['w'], bottleneck['y'] + bottleneck['h']/2),
                arrowprops=dict(arrowstyle='->', color='black', lw=2, connectionstyle="arc3,rad=0.3"))

    # Draw decoder arrows (up)
    for i in range(len(decoder_blocks) - 1):
        ax.annotate('', xy=(14.25, decoder_blocks[i+1]['y']),
                    xytext=(14.25, decoder_blocks[i]['y'] + decoder_blocks[i]['h']),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.text(12.3, (decoder_blocks[i]['y'] + decoder_blocks[i]['h'] + decoder_blocks[i+1]['y'])/2,
                'ConvT\n2×2', fontsize=7, ha='center', va='center')

    # Skip connections
    skip_ys = [7.6, 6.3, 5.0, 3.7]
    for i, y in enumerate(skip_ys):
        ax.annotate('', xy=(13.5, y),
                    xytext=(2.5, y),
                    arrowprops=dict(arrowstyle='->', color=skip_color, lw=2,
                                   linestyle='--', connectionstyle="arc3,rad=0"))
        ax.text(8, y + 0.15, f'Skip Connection (e{i+1})', fontsize=8, ha='center',
                color=skip_color, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=encoder_color, edgecolor='black', label='Encoder (DoubleConvBlock)'),
        mpatches.Patch(facecolor=bottleneck_color, edgecolor='black', label='Bottleneck'),
        mpatches.Patch(facecolor=decoder_color, edgecolor='black', label='Decoder (DoubleConvBlock)'),
        mpatches.Patch(facecolor=skip_color, edgecolor='black', label='Skip Connections'),
        mpatches.Patch(facecolor=output_color, edgecolor='black', label='Output Layer'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig('diagram_unet_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('diagram_unet_architecture.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: diagram_unet_architecture.png/pdf")
    plt.close()


def create_residual_learning_diagram():
    """Create a diagram showing residual learning concept."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Residual Learning for Watermark Addition', fontsize=16, fontweight='bold', pad=20)

    # Colors
    input_color = '#3498db'
    model_color = '#9b59b6'
    residual_color = '#e74c3c'
    output_color = '#2ecc71'
    add_color = '#f39c12'

    # Input block
    rect = FancyBboxPatch((0.5, 2.5), 2, 1.2, boxstyle="round,pad=0.1",
                           facecolor=input_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.5, 3.1, 'Clean Image\n(Input)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # U-Net model block
    rect = FancyBboxPatch((4, 2.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                           facecolor=model_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5.25, 3.1, 'U-Net\nEncoder-Decoder', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Residual output
    rect = FancyBboxPatch((7.5, 2.5), 2, 1.2, boxstyle="round,pad=0.1",
                           facecolor=residual_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(8.5, 3.1, 'Residual\n(Watermark)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Scale factor
    rect = FancyBboxPatch((7.5, 0.8), 2, 0.8, boxstyle="round,pad=0.1",
                           facecolor='#95a5a6', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(8.5, 1.2, '× 0.5 (scale)', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    # Add operation
    circle = plt.Circle((10.5, 3.1), 0.4, facecolor=add_color, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(10.5, 3.1, '+', ha='center', va='center', fontsize=20, fontweight='bold', color='white')

    # Output
    rect = FancyBboxPatch((11.5, 2.5), 2, 1.2, boxstyle="round,pad=0.1",
                           facecolor=output_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(12.5, 3.1, 'Watermarked\n(Output)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Arrows
    # Input to Model
    ax.annotate('', xy=(4, 3.1), xytext=(2.5, 3.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Model to Residual
    ax.annotate('', xy=(7.5, 3.1), xytext=(6.5, 3.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Residual to Scale
    ax.annotate('', xy=(8.5, 2.5), xytext=(8.5, 1.6),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Scale to Add (curved up)
    ax.annotate('', xy=(10.1, 3.1), xytext=(9.5, 1.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=2,
                               connectionstyle="arc3,rad=-0.3"))

    # Input to Add (skip connection)
    ax.annotate('', xy=(10.5, 3.5), xytext=(1.5, 4.5),
                arrowprops=dict(arrowstyle='->', color=input_color, lw=2,
                               linestyle='--', connectionstyle="arc3,rad=-0.2"))
    ax.text(6, 4.8, 'Skip (Identity)', fontsize=10, ha='center',
            color=input_color, fontweight='bold')

    # Add to Output
    ax.annotate('', xy=(11.5, 3.1), xytext=(10.9, 3.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Formula
    ax.text(7, 0.3, r'$\mathbf{Output} = \mathrm{clamp}(\mathbf{Input} + \mathrm{tanh}(\mathbf{Residual}) \times 0.5,\ 0,\ 1)$',
            fontsize=12, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('diagram_residual_learning.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('diagram_residual_learning.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: diagram_residual_learning.png/pdf")
    plt.close()


def create_data_pipeline_diagram():
    """Create a data pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Data Pipeline for Watermark Addition', fontsize=16, fontweight='bold', pad=20)

    # Colors
    data_color = '#3498db'
    process_color = '#9b59b6'
    split_color = '#e74c3c'
    output_color = '#2ecc71'

    # Data folders
    rect = FancyBboxPatch((1, 6.5), 2, 0.8, boxstyle="round,pad=0.1",
                           facecolor=data_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(2, 6.9, 'Watermarked\n(12,510)', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    rect = FancyBboxPatch((9, 6.5), 2, 0.8, boxstyle="round,pad=0.1",
                           facecolor=data_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(10, 6.9, 'Non-Watermarked\n(12,477)', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    # Matching
    rect = FancyBboxPatch((4.5, 5), 3, 0.8, boxstyle="round,pad=0.1",
                           facecolor=process_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 5.4, 'Filename Matching\n(1,744 pairs)', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    # Split
    rect = FancyBboxPatch((4.5, 3.5), 3, 0.8, boxstyle="round,pad=0.1",
                           facecolor=split_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 3.9, 'Train/Val Split\n(80%/20%)', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    # Train/Val boxes
    rect = FancyBboxPatch((2, 2), 2.5, 0.8, boxstyle="round,pad=0.1",
                           facecolor=output_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(3.25, 2.4, 'Training Set\n(1,395 pairs)', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    rect = FancyBboxPatch((7.5, 2), 2.5, 0.8, boxstyle="round,pad=0.1",
                           facecolor=output_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(8.75, 2.4, 'Validation Set\n(349 pairs)', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    # Preprocessing
    rect = FancyBboxPatch((2, 0.5), 8, 1, boxstyle="round,pad=0.1",
                           facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 1, 'Preprocessing: Resize(512×512) → Normalize[0,1] → BGR→RGB → HWC→CHW',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Arrows
    ax.annotate('', xy=(4.5, 5.4), xytext=(3, 6.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(7.5, 5.4), xytext=(9, 6.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(6, 5), xytext=(6, 4.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(3.25, 3.5), xytext=(4.5, 3.9),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(8.75, 3.5), xytext=(7.5, 3.9),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(3.25, 2), xytext=(3.25, 1.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(8.75, 2), xytext=(8.75, 1.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    plt.tight_layout()
    plt.savefig('diagram_data_pipeline.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('diagram_data_pipeline.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: diagram_data_pipeline.png/pdf")
    plt.close()


def create_training_loop_diagram():
    """Create a training loop flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('Training Loop Flowchart', fontsize=16, fontweight='bold', pad=20)

    # Colors
    start_color = '#2ecc71'
    process_color = '#3498db'
    decision_color = '#f39c12'
    end_color = '#e74c3c'

    boxes = [
        {'name': 'Start\nEpoch = 0', 'x': 4, 'y': 11, 'w': 2, 'h': 0.6, 'color': start_color},
        {'name': 'Load Training Batch\n(clean, watermarked)', 'x': 3.5, 'y': 9.8, 'w': 3, 'h': 0.7, 'color': process_color},
        {'name': 'Forward Pass\noutput = model(clean)', 'x': 3.5, 'y': 8.6, 'w': 3, 'h': 0.7, 'color': process_color},
        {'name': 'Compute Loss\nloss(output, watermarked)', 'x': 3.5, 'y': 7.4, 'w': 3, 'h': 0.7, 'color': process_color},
        {'name': 'Backward Pass\nloss.backward()', 'x': 3.5, 'y': 6.2, 'w': 3, 'h': 0.7, 'color': process_color},
        {'name': 'Update Weights\noptimizer.step()', 'x': 3.5, 'y': 5, 'w': 3, 'h': 0.7, 'color': process_color},
        {'name': 'More\nBatches?', 'x': 4, 'y': 3.8, 'w': 2, 'h': 0.7, 'color': decision_color, 'diamond': True},
        {'name': 'Validation\nPhase', 'x': 3.5, 'y': 2.6, 'w': 3, 'h': 0.7, 'color': process_color},
        {'name': 'Save\nCheckpoint?', 'x': 4, 'y': 1.5, 'w': 2, 'h': 0.7, 'color': decision_color, 'diamond': True},
        {'name': 'Early\nStop?', 'x': 4, 'y': 0.4, 'w': 2, 'h': 0.7, 'color': decision_color, 'diamond': True},
    ]

    for box in boxes:
        if box.get('diamond'):
            # Diamond shape for decisions
            diamond = plt.Polygon([(box['x'] + box['w']/2, box['y'] + box['h']),
                                   (box['x'] + box['w'], box['y'] + box['h']/2),
                                   (box['x'] + box['w']/2, box['y']),
                                   (box['x'], box['y'] + box['h']/2)],
                                  facecolor=box['color'], edgecolor='black', linewidth=2)
            ax.add_patch(diamond)
        else:
            rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                                   boxstyle="round,pad=0.05", facecolor=box['color'],
                                   edgecolor='black', linewidth=2)
            ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, box['name'],
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Arrows
    arrow_pairs = [
        ((5, 11), (5, 10.5)),
        ((5, 9.8), (5, 9.3)),
        ((5, 8.6), (5, 8.1)),
        ((5, 7.4), (5, 6.9)),
        ((5, 6.2), (5, 5.7)),
        ((5, 5), (5, 4.5)),
        ((5, 3.8), (5, 3.3)),
        ((5, 2.6), (5, 2.15)),
        ((5, 1.5), (5, 1.1)),
    ]

    for start, end in arrow_pairs:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Yes/No labels and loops
    ax.text(6.3, 4.1, 'Yes', fontsize=9, fontweight='bold')
    ax.annotate('', xy=(8, 9.8), xytext=(6, 4.15),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                               connectionstyle="arc3,rad=-0.3"))

    ax.text(3.5, 4.1, 'No', fontsize=9, fontweight='bold')

    ax.text(6.3, 1.8, 'Yes', fontsize=9, fontweight='bold')
    ax.annotate('', xy=(8.5, 1.85), xytext=(6, 1.85),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(8.5, 1.85, 'Save\nModel', fontsize=8, ha='left')

    ax.text(3.5, 1.8, 'No', fontsize=9, fontweight='bold')

    ax.text(6.3, 0.75, 'Yes', fontsize=9, fontweight='bold')
    ax.annotate('', xy=(8.5, 0.75), xytext=(6, 0.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(8.5, 0.75, 'END', fontsize=10, ha='left', fontweight='bold', color='red')

    ax.text(2.5, 0.75, 'No', fontsize=9, fontweight='bold')
    ax.annotate('', xy=(1, 11.3), xytext=(4, 0.75),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                               connectionstyle="arc3,rad=0.5"))
    ax.text(0.5, 6, 'Next\nEpoch', fontsize=9, fontweight='bold', rotation=90)

    plt.tight_layout()
    plt.savefig('diagram_training_loop.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('diagram_training_loop.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: diagram_training_loop.png/pdf")
    plt.close()


def create_loss_function_diagram():
    """Create a loss function components diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Watermark Loss Function Components', fontsize=16, fontweight='bold', pad=20)

    # Colors
    mse_color = '#3498db'
    l1_color = '#2ecc71'
    diff_color = '#e74c3c'
    total_color = '#9b59b6'

    # Loss components
    components = [
        {'name': 'MSE Loss\nλ=1.0', 'desc': 'Pixel-wise\nreconstruction', 'x': 1, 'y': 3.5, 'color': mse_color},
        {'name': 'L1 Loss\nλ=0.5', 'desc': 'Sharp edges\npreservation', 'x': 4, 'y': 3.5, 'color': l1_color},
        {'name': 'Diff Penalty\nλ=1.0', 'desc': 'Encourage\ndifferences', 'x': 7, 'y': 3.5, 'color': diff_color},
        {'name': 'Min Diff\nλ=1.0', 'desc': 'Force ≥2%\nchange', 'x': 10, 'y': 3.5, 'color': diff_color},
    ]

    for comp in components:
        rect = FancyBboxPatch((comp['x'], comp['y']), 1.8, 1.2,
                               boxstyle="round,pad=0.1", facecolor=comp['color'],
                               edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(comp['x'] + 0.9, comp['y'] + 0.6, comp['name'],
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        ax.text(comp['x'] + 0.9, comp['y'] - 0.5, comp['desc'],
                ha='center', va='center', fontsize=9, color='gray')

    # Total loss
    rect = FancyBboxPatch((4.5, 1), 3, 1, boxstyle="round,pad=0.1",
                           facecolor=total_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 1.5, 'Total Loss\n(Weighted Sum)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Plus signs and arrows
    for i, comp in enumerate(components):
        ax.annotate('', xy=(6, 2), xytext=(comp['x'] + 0.9, comp['y']),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        if i < len(components) - 1:
            ax.text(comp['x'] + 2.4, 4.1, '+', fontsize=16, fontweight='bold')

    # Formula
    ax.text(6, 0.3, r'$\mathcal{L} = \lambda_{mse} \cdot MSE + \lambda_{L1} \cdot L1 + \lambda_{diff} \cdot (DiffPenalty + MinDiffPenalty)$',
            fontsize=11, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('diagram_loss_function.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('diagram_loss_function.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: diagram_loss_function.png/pdf")
    plt.close()


if __name__ == '__main__':
    print("Generating architecture diagrams...")
    print("=" * 50)

    create_unet_architecture_diagram()
    create_residual_learning_diagram()
    create_data_pipeline_diagram()
    create_training_loop_diagram()
    create_loss_function_diagram()

    print("=" * 50)
    print("\nAll diagrams generated successfully!")
    print("\nFiles created:")
    print("  - diagram_unet_architecture.png/pdf")
    print("  - diagram_residual_learning.png/pdf")
    print("  - diagram_data_pipeline.png/pdf")
    print("  - diagram_training_loop.png/pdf")
    print("  - diagram_loss_function.png/pdf")
    print("\nUse these in your interim report!")
