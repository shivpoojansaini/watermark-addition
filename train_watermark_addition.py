"""
Watermark Addition using Convolutional Autoencoder (PyTorch)

This script trains a model to ADD watermarks to clean images.
Input: Clean images -> Output: Watermarked images

Usage:
    python train_watermark_addition.py --data_root ./data/wm-nowm --epochs 100 --batch_size 20

Download dataset from: https://www.kaggle.com/datasets/felicepollano/watermarked-not-watermarked-images
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


def train_test_split_simple(X, y, train_size=0.8, random_state=42):
    """Simple train/test split without sklearn dependency."""
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * train_size)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.0005
DEFAULT_CHECKPOINT_DIR = './checkpoints'


# ============================================================================
# GPU Setup
# ============================================================================

def setup_device():
    """Setup and return the best available device (CUDA/MPS/CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        print("Using CPU (No GPU available)")

    return device


# ============================================================================
# Dataset Class
# ============================================================================

class WatermarkDataset(Dataset):
    """Custom Dataset for watermark addition training."""

    def __init__(self, clean_images, watermarked_images, transform=None, augment=False):
        """
        Args:
            clean_images: numpy array of clean images (input)
            watermarked_images: numpy array of watermarked images (target)
            transform: optional transforms to apply
            augment: whether to apply data augmentation
        """
        self.clean_images = clean_images
        self.watermarked_images = watermarked_images
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean = self.clean_images[idx].astype(np.float32) / 255.0
        watermarked = self.watermarked_images[idx].astype(np.float32) / 255.0

        # Apply augmentation
        if self.augment and random.random() > 0.5:
            clean, watermarked = self._augment(clean, watermarked)

        # Convert BGR to RGB and HWC to CHW (use ascontiguousarray to fix negative strides)
        clean = np.ascontiguousarray(np.transpose(clean[:, :, ::-1], (2, 0, 1)))
        watermarked = np.ascontiguousarray(np.transpose(watermarked[:, :, ::-1], (2, 0, 1)))

        return torch.from_numpy(clean), torch.from_numpy(watermarked)

    def _augment(self, clean, watermarked):
        """Apply random augmentation to both images."""
        # Random brightness
        brightness = random.uniform(0.8, 1.2)
        clean = np.clip(clean * brightness, 0, 1)
        watermarked = np.clip(watermarked * brightness, 0, 1)

        # Random contrast
        contrast = random.uniform(0.8, 1.2)
        clean = np.clip((clean - 0.5) * contrast + 0.5, 0, 1)
        watermarked = np.clip((watermarked - 0.5) * contrast + 0.5, 0, 1)

        # Random horizontal flip
        if random.random() > 0.5:
            clean = np.fliplr(clean).copy()
            watermarked = np.fliplr(watermarked).copy()

        return clean, watermarked


# ============================================================================
# Model Definition
# ============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class WatermarkAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for watermark addition.

    Architecture:
        Encoder: Conv -> MaxPool -> BN (x3)
        Decoder: Conv -> Upsample (x3)
    """

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
        # Encoder
        e1 = self.pool1(self.enc1(x))
        e2 = self.pool2(self.enc2(e1))
        e3 = self.pool3(self.enc3(e2))

        # Decoder
        d1 = self.up1(self.dec1(e3))
        d2 = self.up2(self.dec2(d1))
        d3 = self.up3(self.dec3(d2))

        # Output
        out = self.sigmoid(self.output(d3))

        return out


# ============================================================================
# Data Processing Functions
# ============================================================================

def take_filename(filedir):
    """Extract filename from file path."""
    return os.path.basename(filedir)


def match_filenames(watermarkedarr, nonwatermarkedarr, dname_wm, dname_nwm):
    """Match watermarked and non-watermarked images by filename."""
    sortedwmarr = []
    sortednwmarr = []

    wmarr = list(watermarkedarr)
    nwmarr = list(nonwatermarkedarr)

    length = max(len(watermarkedarr), len(nonwatermarkedarr))

    for pos in range(length):
        try:
            if length == len(watermarkedarr):
                exist_nwm = nwmarr.index(wmarr[pos])
                sortedwmarr.append(dname_wm + watermarkedarr[pos])
                sortednwmarr.append(dname_nwm + nonwatermarkedarr[exist_nwm])
            elif length == len(nonwatermarkedarr):
                exist_wm = wmarr.index(nwmarr[pos])
                sortedwmarr.append(dname_wm + watermarkedarr[exist_wm])
                sortednwmarr.append(dname_nwm + nonwatermarkedarr[pos])
        except ValueError:
            continue

    return np.array(sortedwmarr), np.array(sortednwmarr)


def create_pixel_array(files, width, height):
    """Load images and convert to pixel arrays."""
    data = []
    for image in tqdm(files, desc="Loading images"):
        try:
            img_arr = cv2.imread(image, cv2.IMREAD_COLOR)
            if img_arr is not None:
                resized_arr = cv2.resize(img_arr, (width, height))
                data.append(resized_arr)
        except Exception as e:
            print(f"Error loading {image}: {e}")
    return np.array(data)


def load_dataset(data_root, width, height):
    """Load and prepare the dataset."""
    print(f"\nLoading dataset from: {data_root}")

    # Training paths
    train_path_wm = f'{data_root}/train/watermark/'
    train_path_nwm = f'{data_root}/train/no-watermark/'

    # Collect filenames
    tp_watermarked = []
    tp_nonwatermarked = []

    for root, dirs, files in os.walk(train_path_wm, topdown=True):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                tp_watermarked.append(take_filename(file))

    for root, dirs, files in os.walk(train_path_nwm, topdown=True):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                tp_nonwatermarked.append(take_filename(file))

    print(f"Found {len(tp_watermarked)} watermarked images")
    print(f"Found {len(tp_nonwatermarked)} non-watermarked images")

    # Match filenames
    tp_wm_sorted, tp_nwm_sorted = match_filenames(
        np.array(tp_watermarked), np.array(tp_nonwatermarked),
        train_path_wm, train_path_nwm
    )

    print(f"Matched {len(tp_wm_sorted)} training pairs")

    # Load pixel values
    print("\nLoading training images...")
    train_wms = create_pixel_array(tp_wm_sorted, width, height)
    train_nwms = create_pixel_array(tp_nwm_sorted, width, height)

    return train_nwms, train_wms  # X=clean, y=watermarked


# ============================================================================
# Checkpoint Functions
# ============================================================================

class CheckpointManager:
    """Manages model checkpoints."""

    def __init__(self, checkpoint_dir, model_name='watermark_addition'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.best_loss = float('inf')

    def save_checkpoint(self, model, optimizer, epoch, loss, is_best=False):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / f'{self.model_name}_latest.pth'
        torch.save(checkpoint, latest_path)

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            periodic_path = self.checkpoint_dir / f'{self.model_name}_epoch_{epoch+1}.pth'
            torch.save(checkpoint, periodic_path)
            print(f"Saved periodic checkpoint: {periodic_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f'{self.model_name}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path} (loss: {loss:.6f})")
            self.best_loss = loss

    def load_checkpoint(self, model, optimizer=None, checkpoint_path=None):
        """Load a checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / f'{self.model_name}_latest.pth'

        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}")
            return 0, float('inf')

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        self.best_loss = loss

        print(f"Resumed from epoch {epoch + 1} with loss {loss:.6f}")
        return epoch + 1, loss


# ============================================================================
# Training Functions
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for clean, watermarked in pbar:
        clean = clean.to(device, non_blocking=True)
        watermarked = watermarked.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        output = model(clean)
        loss = criterion(output, watermarked)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Valid]")
        for clean, watermarked in pbar:
            clean = clean.to(device, non_blocking=True)
            watermarked = watermarked.to(device, non_blocking=True)

            output = model(clean)
            loss = criterion(output, watermarked)

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return running_loss / len(dataloader)


def train_model(model, train_loader, val_loader, device, args, checkpoint_manager, start_epoch=0):
    """Full training loop."""
    print("\n" + "="*60)
    print("Training Watermark Addition Model (PyTorch)")
    print("="*60)
    print(f"Device: {device}")
    print(f"Image size: {args.width}x{args.height}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*60 + "\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=args.patience)

    # Load optimizer state if resuming
    if start_epoch > 0:
        checkpoint_manager.load_checkpoint(model, optimizer)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(start_epoch, args.epochs):
        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        history['train_loss'].append(train_loss)

        # Validation
        val_loss = validate(model, val_loader, criterion, device, epoch)
        history['val_loss'].append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")

        # Save checkpoint
        is_best = val_loss < checkpoint_manager.best_loss
        checkpoint_manager.save_checkpoint(model, optimizer, epoch, val_loss, is_best)

        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

        # GPU memory info
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    return history


# ============================================================================
# Inference Function
# ============================================================================

def add_watermark_to_image(image_path, model, device, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """
    Add watermark to a single image.

    Args:
        image_path: Path to the input clean image
        model: Trained watermark addition model
        device: torch device
        width: Target width for resizing
        height: Target height for resizing

    Returns:
        watermarked: Watermarked image (numpy array, BGR)
    """
    model.eval()

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_shape = img.shape[:2]

    img_resized = cv2.resize(img, (width, height))
    img_normalized = img_resized.astype(np.float32) / 255.0

    # BGR to RGB, HWC to CHW
    img_tensor = torch.tensor(np.transpose(img_normalized[:, :, ::-1], (2, 0, 1)))
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    # Convert back
    watermarked = output.squeeze(0).cpu().numpy()
    watermarked = np.transpose(watermarked, (1, 2, 0))[:, :, ::-1]  # CHW to HWC, RGB to BGR
    watermarked = (watermarked * 255).astype(np.uint8)

    watermarked_original_size = cv2.resize(watermarked, (original_shape[1], original_shape[0]))

    return watermarked, watermarked_original_size


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_results(model, dataloader, device, num_images=5, save_path=None):
    """Plot comparison of input and output images."""
    model.eval()

    # Get a batch
    clean, watermarked = next(iter(dataloader))
    clean = clean[:num_images].to(device)

    with torch.no_grad():
        output = model(clean)

    # Convert to numpy
    clean = clean.cpu().numpy()
    output = output.cpu().numpy()
    watermarked = watermarked[:num_images].numpy()

    plt.figure(figsize=(15, 9))

    for i in range(num_images):
        # Input (clean)
        plt.subplot(3, num_images, i + 1)
        img = np.transpose(clean[i], (1, 2, 0))
        plt.imshow(img)
        plt.title('Clean Input')
        plt.axis('off')

        # Output (predicted watermarked)
        plt.subplot(3, num_images, num_images + i + 1)
        img = np.transpose(output[i], (1, 2, 0))
        plt.imshow(img)
        plt.title('Predicted')
        plt.axis('off')

        # Ground truth (actual watermarked)
        plt.subplot(3, num_images, 2 * num_images + i + 1)
        img = np.transpose(watermarked[i], (1, 2, 0))
        plt.imshow(img)
        plt.title('Ground Truth')
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Results saved to: {save_path}")

    plt.show()


def plot_training_history(history, save_path=None):
    """Plot training loss history."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training history saved to: {save_path}")

    plt.show()


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Watermark Addition Model (PyTorch)')
    parser.add_argument('--data_root', type=str, default='./data/wm-nowm',
                        help='Path to dataset root directory')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH,
                        help='Image width')
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT,
                        help='Image height')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help='Directory for saving checkpoints')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from specific checkpoint file')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--no_plots', action='store_true',
                        help='Disable plotting')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output_model', type=str, default='watermark_addition_model.pth',
                        help='Output model filename')

    args = parser.parse_args()

    # Setup device
    device = setup_device()

    # Check if dataset exists
    if not os.path.exists(args.data_root):
        print(f"Error: Dataset not found at '{args.data_root}'")
        print("Please download from: https://www.kaggle.com/datasets/felicepollano/watermarked-not-watermarked-images")
        return

    # Load dataset
    X, y = load_dataset(args.data_root, args.width, args.height)

    # Split into train/val
    X_train, X_val, y_train, y_val = train_test_split_simple(X, y, train_size=0.8, random_state=42)
    print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}")

    # Create datasets
    train_dataset = WatermarkDataset(X_train, y_train, augment=not args.no_augmentation)
    val_dataset = WatermarkDataset(X_val, y_val, augment=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # Create model
    model = WatermarkAutoencoder().to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume or args.resume_from:
        checkpoint_path = args.resume_from if args.resume_from else None
        start_epoch, _ = checkpoint_manager.load_checkpoint(model, checkpoint_path=checkpoint_path)

    # Train
    history = train_model(model, train_loader, val_loader, device, args, checkpoint_manager, start_epoch)

    # Load best model for final evaluation
    checkpoint_manager.load_checkpoint(model, checkpoint_path=checkpoint_manager.checkpoint_dir / 'watermark_addition_best.pth')

    # Save final model (just weights)
    torch.save(model.state_dict(), args.output_model)
    print(f"\nFinal model saved to: {args.output_model}")

    # Plot results
    if not args.no_plots:
        plot_training_history(history, 'training_history.png')
        plot_results(model, val_loader, device, num_images=5, save_path='results.png')

    print("\nTraining complete!")
    print(f"Best model checkpoint: {checkpoint_manager.checkpoint_dir / 'watermark_addition_best.pth'}")
    print(f"Latest checkpoint: {checkpoint_manager.checkpoint_dir / 'watermark_addition_latest.pth'}")


if __name__ == '__main__':
    main()
