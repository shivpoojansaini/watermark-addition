"""
GAN-based Visible Watermark Addition with Pretrained Backbone

Designed for fine-tuning on small datasets (1000+ pairs).
Uses pretrained VGG/ResNet encoder + GAN for fast, effective training.

Usage:
    python train_watermark_gan.py \
        --data_root ./data/wm-nowm \
        --epochs 50 \
        --batch_size 8

    # Or with custom paths:
    python train_watermark_gan.py \
        --clean_dir ./data/clean \
        --watermarked_dir ./data/watermarked \
        --epochs 100
"""

import argparse
import os
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm


# ============================================================================
# Device Setup
# ============================================================================

def setup_device():
    """Setup and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("⚠ Using CPU (training will be slow)")
    return device


# ============================================================================
# Pretrained Feature Extractor (for Perceptual Loss)
# ============================================================================

class VGGFeatureExtractor(nn.Module):
    """
    Pretrained VGG19 for perceptual loss.
    Extracts features at multiple scales.
    """
    
    def __init__(self, device: torch.device):
        super().__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Extract features at different layers
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18]) # relu3_4
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27]) # relu4_4
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        x = (x - self.mean) / self.std
        
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        
        return [h1, h2, h3, h4]


# ============================================================================
# Generator with Pretrained Encoder
# ============================================================================

class PretrainedEncoder(nn.Module):
    """
    Use pretrained ResNet34 as encoder backbone.
    Already knows how to extract meaningful image features!
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet34
        if pretrained:
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet34(weights=None)
        
        # Extract encoder layers
        self.conv1 = resnet.conv1      # 64, stride 2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # stride 2
        
        self.layer1 = resnet.layer1    # 64
        self.layer2 = resnet.layer2    # 128, stride 2
        self.layer3 = resnet.layer3    # 256, stride 2
        self.layer4 = resnet.layer4    # 512, stride 2
        
        # Output channels at each level
        self.channels = [64, 64, 128, 256, 512]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns features at multiple scales."""
        features = []
        
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # 64 channels, 1/2 resolution
        
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        features.append(x)  # 64 channels, 1/4 resolution
        
        x = self.layer2(x)
        features.append(x)  # 128 channels, 1/8 resolution
        
        x = self.layer3(x)
        features.append(x)  # 256 channels, 1/16 resolution
        
        x = self.layer4(x)
        features.append(x)  # 512 channels, 1/32 resolution
        
        return features


class DecoderBlock(nn.Module):
    """Decoder block with skip connection."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
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
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class WatermarkGenerator(nn.Module):
    """
    Generator with pretrained ResNet encoder.
    
    Architecture:
    - Pretrained ResNet34 encoder (frozen initially, then fine-tuned)
    - Custom decoder with skip connections
    - Residual learning: output = input + learned_watermark
    """
    
    def __init__(self, pretrained: bool = True, residual_scale: float = 1.0):
        super().__init__()
        
        self.residual_scale = residual_scale
        
        # Pretrained encoder
        self.encoder = PretrainedEncoder(pretrained=pretrained)
        
        # Decoder
        # Channels: 512 -> 256 -> 128 -> 64 -> 64 -> 3
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        
        # Final upsampling to original resolution
        self.final_up = nn.ConvTranspose2d(64, 64, 2, stride=2)
        
        # Output head - predicts residual (watermark to add)
        self.output = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Clean image [B, 3, H, W]
        
        Returns:
            watermarked: Image with watermark added
            residual: The watermark that was added
        """
        # Encode
        features = self.encoder(x)
        # features: [f0(64), f1(64), f2(128), f3(256), f4(512)]
        
        # Decode with skip connections
        d4 = self.dec4(features[4], features[3])  # 512 + 256 -> 256
        d3 = self.dec3(d4, features[2])           # 256 + 128 -> 128
        d2 = self.dec2(d3, features[1])           # 128 + 64 -> 64
        d1 = self.dec1(d2, features[0])           # 64 + 64 -> 64
        
        # Final upsample
        d0 = self.final_up(d1)
        
        # Handle size mismatch with input
        if d0.shape[2:] != x.shape[2:]:
            d0 = F.interpolate(d0, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Generate residual (watermark)
        residual = self.output(d0) * self.residual_scale
        
        # Add to input
        watermarked = torch.clamp(x + residual, 0, 1)
        
        return watermarked, residual
    
    def freeze_encoder(self):
        """Freeze encoder for initial training (train decoder only)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("  Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("  Encoder unfrozen for fine-tuning")


# ============================================================================
# Discriminator (PatchGAN style)
# ============================================================================

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator - classifies if 70x70 patches are real/fake.
    More effective than full-image discriminator for image-to-image tasks.
    """
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        def discriminator_block(in_ch, out_ch, stride=2, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            # Input: 3 x 256 x 256 (or similar)
            discriminator_block(in_channels, 64, normalize=False),  # 64 x 128 x 128
            discriminator_block(64, 128),                            # 128 x 64 x 64
            discriminator_block(128, 256),                           # 256 x 32 x 32
            discriminator_block(256, 512, stride=1),                 # 512 x 31 x 31
            nn.Conv2d(512, 1, 4, padding=1)                          # 1 x 30 x 30
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns patch-level predictions."""
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for better gradient flow.
    Uses 2 discriminators at different scales.
    """
    
    def __init__(self):
        super().__init__()
        
        self.disc1 = PatchDiscriminator()  # Full scale
        self.disc2 = PatchDiscriminator()  # Half scale
        
        self.downsample = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        
        outputs.append(self.disc1(x))
        outputs.append(self.disc2(self.downsample(x)))
        
        return outputs


# ============================================================================
# Loss Functions
# ============================================================================

class GANLoss(nn.Module):
    """GAN loss with label smoothing."""
    
    def __init__(self, use_lsgan: bool = True):
        super().__init__()
        self.use_lsgan = use_lsgan
        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if target_is_real:
            # Label smoothing: use 0.9 instead of 1.0
            target = torch.ones_like(pred) * 0.9
        else:
            target = torch.zeros_like(pred)
        
        return self.loss(pred, target)


class WatermarkAwareLoss(nn.Module):
    """
    Loss that weights watermark regions more heavily.
    """
    
    def __init__(self, watermark_weight: float = 10.0):
        super().__init__()
        self.watermark_weight = watermark_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # L1 loss
        l1_loss = torch.abs(pred - target)
        
        if mask is not None:
            # Weight watermark regions more
            weight = 1.0 + (self.watermark_weight - 1.0) * mask
            l1_loss = l1_loss * weight
        
        return l1_loss.mean()


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self, vgg_extractor: VGGFeatureExtractor):
        super().__init__()
        self.vgg = vgg_extractor
        self.weights = [1.0, 1.0, 1.0, 1.0]  # Weight for each layer
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        
        loss = 0.0
        for w, pf, tf in zip(self.weights, pred_features, target_features):
            loss += w * F.l1_loss(pf, tf)
        
        return loss


# ============================================================================
# Dataset
# ============================================================================

class WatermarkPairDataset(Dataset):
    """
    Dataset for paired watermark training.
    Expects matched filenames in clean and watermarked directories.
    """
    
    def __init__(self, clean_dir: str, watermarked_dir: str, 
                 size: Tuple[int, int] = (256, 256),
                 augment: bool = True):
        
        self.clean_dir = Path(clean_dir)
        self.watermarked_dir = Path(watermarked_dir)
        self.size = size
        self.augment = augment
        
        # Find matching pairs
        self.pairs = self._find_pairs()
        
        print(f"Found {len(self.pairs)} image pairs")
    
    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching clean/watermarked image pairs."""
        pairs = []
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        # Get all clean images
        clean_files = {}
        for ext in extensions:
            for f in self.clean_dir.glob(f'*{ext}'):
                clean_files[f.stem.lower()] = f
            for f in self.clean_dir.glob(f'*{ext.upper()}'):
                clean_files[f.stem.lower()] = f
        
        # Find matching watermarked images
        for ext in extensions:
            for wm_path in self.watermarked_dir.glob(f'*{ext}'):
                stem = wm_path.stem.lower()
                if stem in clean_files:
                    pairs.append((clean_files[stem], wm_path))
            for wm_path in self.watermarked_dir.glob(f'*{ext.upper()}'):
                stem = wm_path.stem.lower()
                if stem in clean_files:
                    pairs.append((clean_files[stem], wm_path))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clean_path, wm_path = self.pairs[idx]
        
        # Load images
        clean = cv2.imread(str(clean_path))
        watermarked = cv2.imread(str(wm_path))
        
        if clean is None or watermarked is None:
            # Return random valid pair if load fails
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Resize
        clean = cv2.resize(clean, self.size)
        watermarked = cv2.resize(watermarked, self.size)
        
        # Augmentation
        if self.augment and random.random() > 0.5:
            # Horizontal flip
            clean = cv2.flip(clean, 1)
            watermarked = cv2.flip(watermarked, 1)
        
        if self.augment and random.random() > 0.5:
            # Random brightness/contrast
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.uniform(-20, 20)    # brightness
            clean = np.clip(alpha * clean + beta, 0, 255).astype(np.uint8)
            watermarked = np.clip(alpha * watermarked + beta, 0, 255).astype(np.uint8)
        
        # Normalize to [0, 1]
        clean = clean.astype(np.float32) / 255.0
        watermarked = watermarked.astype(np.float32) / 255.0
        
        # BGR to RGB
        clean = clean[:, :, ::-1].copy()
        watermarked = watermarked[:, :, ::-1].copy()
        
        # Compute difference mask (where watermark is)
        diff = np.abs(watermarked - clean).mean(axis=2, keepdims=True)
        mask = (diff > 0.05).astype(np.float32)  # Threshold for watermark detection
        
        # To tensor [C, H, W]
        clean = torch.from_numpy(clean.transpose(2, 0, 1)).float()
        watermarked = torch.from_numpy(watermarked.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        
        return {
            'clean': clean,
            'watermarked': watermarked,
            'mask': mask
        }


# ============================================================================
# Training
# ============================================================================

class WatermarkGANTrainer:
    """
    Complete training pipeline for watermark GAN.
    """
    
    def __init__(self, args, device: torch.device):
        self.args = args
        self.device = device
        
        # Models
        print("\n" + "="*60)
        print("INITIALIZING MODELS")
        print("="*60)
        
        self.generator = WatermarkGenerator(
            pretrained=True,
            residual_scale=args.residual_scale
        ).to(device)
        
        self.discriminator = MultiScaleDiscriminator().to(device)
        
        print(f"Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        # Feature extractor for perceptual loss
        self.vgg = VGGFeatureExtractor(device)
        
        # Losses
        self.gan_loss = GANLoss(use_lsgan=True)
        self.l1_loss = WatermarkAwareLoss(watermark_weight=args.watermark_weight)
        self.perceptual_loss = PerceptualLoss(self.vgg)
        
        # Optimizers
        self.opt_g = optim.Adam(
            self.generator.parameters(),
            lr=args.lr_g,
            betas=(0.5, 0.999)
        )
        self.opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=args.lr_d,
            betas=(0.5, 0.999)
        )
        
        # Schedulers
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_g, T_max=args.epochs, eta_min=1e-6
        )
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_d, T_max=args.epochs, eta_min=1e-6
        )
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {'g_loss': [], 'd_loss': [], 'val_loss': []}
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch in pbar:
            clean = batch['clean'].to(self.device)
            real_wm = batch['watermarked'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            self.opt_d.zero_grad()
            
            # Real samples
            d_real = self.discriminator(real_wm)
            loss_d_real = sum(self.gan_loss(d, True) for d in d_real) / len(d_real)
            
            # Fake samples
            fake_wm, _ = self.generator(clean)
            d_fake = self.discriminator(fake_wm.detach())
            loss_d_fake = sum(self.gan_loss(d, False) for d in d_fake) / len(d_fake)
            
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            self.opt_d.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            self.opt_g.zero_grad()
            
            fake_wm, residual = self.generator(clean)
            
            # Adversarial loss
            d_fake = self.discriminator(fake_wm)
            loss_g_adv = sum(self.gan_loss(d, True) for d in d_fake) / len(d_fake)
            
            # Reconstruction loss (L1 with watermark weighting)
            loss_g_l1 = self.l1_loss(fake_wm, real_wm, mask)
            
            # Perceptual loss
            loss_g_perc = self.perceptual_loss(fake_wm, real_wm)
            
            # Total generator loss
            loss_g = (
                self.args.lambda_adv * loss_g_adv +
                self.args.lambda_l1 * loss_g_l1 +
                self.args.lambda_perc * loss_g_perc
            )
            
            loss_g.backward()
            self.opt_g.step()
            
            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()
            
            pbar.set_postfix({
                'G': f'{loss_g.item():.4f}',
                'D': f'{loss_d.item():.4f}'
            })
        
        return total_g_loss / len(dataloader), total_d_loss / len(dataloader)
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model."""
        self.generator.eval()
        
        total_loss = 0
        
        for batch in dataloader:
            clean = batch['clean'].to(self.device)
            real_wm = batch['watermarked'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            fake_wm, _ = self.generator(clean)
            
            loss = self.l1_loss(fake_wm, real_wm, mask)
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'opt_g_state_dict': self.opt_g.state_dict(),
            'opt_d_state_dict': self.opt_d.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'args': vars(self.args)
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
        self.opt_d.load_state_dict(checkpoint['opt_d_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop."""
        
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        print(f"Epochs: {self.args.epochs}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Learning rate (G): {self.args.lr_g}")
        print(f"Learning rate (D): {self.args.lr_d}")
        print(f"Lambda adversarial: {self.args.lambda_adv}")
        print(f"Lambda L1: {self.args.lambda_l1}")
        print(f"Lambda perceptual: {self.args.lambda_perc}")
        print(f"Watermark weight: {self.args.watermark_weight}")
        print(f"Residual scale: {self.args.residual_scale}")
        
        # Phase 1: Train decoder only (encoder frozen)
        if self.args.freeze_encoder_epochs > 0:
            print("\n" + "-"*60)
            print(f"PHASE 1: Training decoder only ({self.args.freeze_encoder_epochs} epochs)")
            print("-"*60)
            self.generator.freeze_encoder()
        
        for epoch in range(self.args.epochs):
            self.current_epoch = epoch
            
            # Unfreeze encoder after initial phase
            if epoch == self.args.freeze_encoder_epochs and self.args.freeze_encoder_epochs > 0:
                print("\n" + "-"*60)
                print("PHASE 2: Fine-tuning full model")
                print("-"*60)
                self.generator.unfreeze_encoder()
                
                # Reduce learning rate for fine-tuning
                for param_group in self.opt_g.param_groups:
                    param_group['lr'] = self.args.lr_g * 0.1
            
            # Train
            g_loss, d_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update history
            self.history['g_loss'].append(g_loss)
            self.history['d_loss'].append(d_loss)
            self.history['val_loss'].append(val_loss)
            
            # Update schedulers
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            # Print progress
            print(f"Epoch {epoch + 1}/{self.args.epochs} | "
                  f"G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                print(f"  ✓ New best model!")
            
            self.save_checkpoint(self.args.checkpoint_path, is_best=is_best)
            
            # Visualize every N epochs
            if (epoch + 1) % self.args.vis_every == 0 or epoch == 0:
                self.visualize(val_loader, epoch + 1)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
    
    @torch.no_grad()
    def visualize(self, dataloader: DataLoader, epoch: int, num_samples: int = 4):
        """Visualize model outputs."""
        self.generator.eval()
        
        batch = next(iter(dataloader))
        clean = batch['clean'][:num_samples].to(self.device)
        real_wm = batch['watermarked'][:num_samples]
        
        fake_wm, residual = self.generator(clean)
        
        # Convert to numpy
        clean = clean.cpu().numpy()
        real_wm = real_wm.numpy()
        fake_wm = fake_wm.cpu().numpy()
        residual = residual.cpu().numpy()
        
        # Plot
        fig, axes = plt.subplots(4, num_samples, figsize=(4 * num_samples, 16))
        
        for i in range(num_samples):
            # Clean input
            axes[0, i].imshow(clean[i].transpose(1, 2, 0))
            axes[0, i].set_title('Clean Input')
            axes[0, i].axis('off')
            
            # Real watermarked
            axes[1, i].imshow(real_wm[i].transpose(1, 2, 0))
            axes[1, i].set_title('Target (Real WM)')
            axes[1, i].axis('off')
            
            # Generated watermarked
            axes[2, i].imshow(np.clip(fake_wm[i].transpose(1, 2, 0), 0, 1))
            axes[2, i].set_title('Generated')
            axes[2, i].axis('off')
            
            # Residual (amplified)
            res = residual[i].transpose(1, 2, 0)
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            axes[3, i].imshow(res)
            axes[3, i].set_title('Residual (amplified)')
            axes[3, i].axis('off')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.args.output_dir, f'results_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved visualization: {save_path}")


# ============================================================================
# Inference
# ============================================================================

def load_trained_model(checkpoint_path: str, device: torch.device) -> WatermarkGenerator:
    """Load trained model for inference."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get args from checkpoint
    args = checkpoint.get('args', {})
    residual_scale = args.get('residual_scale', 1.0)
    
    model = WatermarkGenerator(pretrained=False, residual_scale=residual_scale)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def add_watermark(model: WatermarkGenerator, image_path: str, 
                  device: torch.device, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Add watermark to a single image."""
    # Load image
    img = cv2.imread(image_path)
    original_size = img.shape[:2]
    
    # Preprocess
    img_resized = cv2.resize(img, size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_rgb = img_normalized[:, :, ::-1].copy()
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        watermarked, _ = model(img_tensor)
    
    # Post-process
    watermarked = watermarked.squeeze(0).cpu().numpy()
    watermarked = np.clip(watermarked.transpose(1, 2, 0), 0, 1)
    watermarked = (watermarked[:, :, ::-1] * 255).astype(np.uint8)
    
    # Resize back to original
    watermarked = cv2.resize(watermarked, (original_size[1], original_size[0]))
    
    return watermarked


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Watermark GAN')
    
    # Data paths
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory (expects train/watermark and train/no-watermark)')
    parser.add_argument('--clean_dir', type=str, default=None,
                        help='Directory with clean images')
    parser.add_argument('--watermarked_dir', type=str, default=None,
                        help='Directory with watermarked images')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr_g', type=float, default=2e-4, help='Generator LR')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='Discriminator LR')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Loss weights
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='Adversarial loss weight')
    parser.add_argument('--lambda_l1', type=float, default=100.0, help='L1 loss weight')
    parser.add_argument('--lambda_perc', type=float, default=10.0, help='Perceptual loss weight')
    parser.add_argument('--watermark_weight', type=float, default=10.0,
                        help='Extra weight for watermark regions')
    
    # Model
    parser.add_argument('--residual_scale', type=float, default=1.0)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--freeze_encoder_epochs', type=int, default=10,
                        help='Epochs to train with frozen encoder')
    
    # Dataset limiting
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Maximum number of image pairs to use (default: use all)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./output_gan')
    parser.add_argument('--checkpoint_path', type=str, default='watermark_gan.pth')
    parser.add_argument('--vis_every', type=int, default=5, help='Visualize every N epochs')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Setup directories
    if args.data_root:
        args.clean_dir = os.path.join(args.data_root, 'train', 'no-watermark')
        args.watermarked_dir = os.path.join(args.data_root, 'train', 'watermark')
    
    if not args.clean_dir or not args.watermarked_dir:
        print("Error: Provide either --data_root or both --clean_dir and --watermarked_dir")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = setup_device()
    
    # Dataset
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    full_dataset = WatermarkPairDataset(
        args.clean_dir,
        args.watermarked_dir,
        size=(args.image_size, args.image_size),
        augment=True
    )
    
    # Limit number of pairs if specified
    if args.max_pairs and args.max_pairs < len(full_dataset):
        print(f"Limiting dataset from {len(full_dataset)} to {args.max_pairs} pairs")
        indices = torch.randperm(len(full_dataset))[:args.max_pairs].tolist()
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
    
    # Split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Disable augmentation for validation
    val_dataset.dataset.augment = False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Trainer
    trainer = WatermarkGANTrainer(args, device)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Final visualization
    print("\n" + "="*60)
    print("GENERATING FINAL RESULTS")
    print("="*60)
    trainer.visualize(val_loader, epoch=args.epochs, num_samples=6)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(trainer.history['g_loss'], label='Generator')
    plt.plot(trainer.history['d_loss'], label='Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(trainer.history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'), dpi=150)
    plt.close()
    
    print(f"\n✓ Model saved to: {args.checkpoint_path}")
    print(f"✓ Best model saved to: {args.checkpoint_path.replace('.pth', '_best.pth')}")
    print(f"✓ Results saved to: {args.output_dir}/")
    
    print("\n" + "-"*60)
    print("USAGE FOR INFERENCE:")
    print("-"*60)
    print(f"""
from train_watermark_gan import load_trained_model, add_watermark
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_trained_model('{args.checkpoint_path.replace('.pth', '_best.pth')}', device)

# Add watermark to image
result = add_watermark(model, 'input.jpg', device, size=({args.image_size}, {args.image_size}))
cv2.imwrite('watermarked.jpg', result)
    """)


if __name__ == '__main__':
    main()
