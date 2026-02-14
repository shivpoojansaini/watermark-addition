# Watermark Addition Model - Architecture Report

## 1. Project Overview

This project implements a deep learning model for **automatic watermark addition** to images. Given a clean (non-watermarked) image as input, the model outputs a watermarked version of the same image.

### 1.1 Problem Statement
- **Input**: Clean image without watermark
- **Output**: Image with watermark added
- **Approach**: Residual learning using U-Net architecture

---

## 2. Model Architecture

### 2.1 High-Level Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WATERMARK ADDITION MODEL                              │
│                    (U-Net with Residual Learning)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────┐                                      ┌───────────────┐      │
│   │   INPUT   │                                      │    OUTPUT     │      │
│   │   Clean   │──────────────────┬───────────────────►│  Watermarked  │      │
│   │   Image   │                  │                   │    Image      │      │
│   │ 512x512x3 │                  │ (+)               │  512x512x3    │      │
│   └─────┬─────┘                  │                   └───────────────┘      │
│         │                        │                          ▲               │
│         ▼                        │                          │               │
│   ┌─────────────────────────────────────────────────────────┴─────┐        │
│   │                         U-NET ENCODER-DECODER                  │        │
│   │                                                                │        │
│   │    ENCODER              BOTTLENECK           DECODER           │        │
│   │   ┌───────┐            ┌─────────┐          ┌───────┐         │        │
│   │   │Conv64 │──┐         │ Conv512 │     ┌────│ Conv64│         │        │
│   │   │512x512│  │         │  32x32  │     │    │512x512│         │        │
│   │   └───┬───┘  │         └────┬────┘     │    └───┬───┘         │        │
│   │       ↓      │skip          │          │skip    ↓             │        │
│   │   ┌───────┐  │connection    │      connection┌───────┐        │        │
│   │   │Conv128│──┼──────────────┼──────────┼─────│Conv128│        │        │
│   │   │256x256│  │              │          │     │256x256│        │        │
│   │   └───┬───┘  │              │          │     └───┬───┘        │        │
│   │       ↓      │              │          │         ↓            │        │
│   │   ┌───────┐  │              │          │     ┌───────┐        │        │
│   │   │Conv256│──┼──────────────┼──────────┼─────│Conv256│        │        │
│   │   │128x128│  │              │          │     │128x128│        │        │
│   │   └───┬───┘  │              │          │     └───┬───┘        │        │
│   │       ↓      │              │          │         ↓            │        │
│   │   ┌───────┐  │              ▼          │     ┌───────┐        │        │
│   │   │Conv512│──┴─────────►[ CONCAT ]◄────┴─────│Conv512│        │        │
│   │   │ 64x64 │                                  │ 64x64 │        │        │
│   │   └───────┘                                  └───────┘        │        │
│   │                                                                │        │
│   │   Output: Predicted Residual (Watermark Overlay) × 0.5        │        │
│   └────────────────────────────────────────────────────────────────┘        │
│                                                                             │
│   Final Output = Input + (Predicted Residual × residual_scale)             │
│                = Clean Image + Watermark Overlay                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Detailed Architecture Diagram

```
INPUT IMAGE (512 × 512 × 3)
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ENCODER PATH                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐                               │
│  │   DoubleConv    │      │   MaxPool2d     │                               │
│  │   3 → 64        │─────►│   2×2           │                               │
│  │   512×512×64    │      │   256×256×64    │─────────────────────┐ (e1)    │
│  └─────────────────┘      └────────┬────────┘                     │         │
│                                    │                              │         │
│  ┌─────────────────┐      ┌────────▼────────┐                     │         │
│  │   DoubleConv    │      │   MaxPool2d     │                     │         │
│  │   64 → 128      │◄─────│   2×2           │                     │         │
│  │   256×256×128   │      │   128×128×128   │─────────────────┐   │ (e2)    │
│  └─────────────────┘      └────────┬────────┘                 │   │         │
│                                    │                          │   │         │
│  ┌─────────────────┐      ┌────────▼────────┐                 │   │         │
│  │   DoubleConv    │      │   MaxPool2d     │                 │   │         │
│  │   128 → 256     │◄─────│   2×2           │                 │   │         │
│  │   128×128×256   │      │   64×64×256     │─────────────┐   │   │ (e3)    │
│  └─────────────────┘      └────────┬────────┘             │   │   │         │
│                                    │                      │   │   │         │
│  ┌─────────────────┐      ┌────────▼────────┐             │   │   │         │
│  │   DoubleConv    │      │   MaxPool2d     │             │   │   │         │
│  │   256 → 512     │◄─────│   2×2           │             │   │   │         │
│  │   64×64×512     │      │   32×32×512     │─────────┐   │   │   │ (e4)    │
│  └─────────────────┘      └────────┬────────┘         │   │   │   │         │
│                                    │                  │   │   │   │         │
└────────────────────────────────────┼──────────────────┼───┼───┼───┼─────────┘
                                     │                  │   │   │   │
                                     ▼                  │   │   │   │
┌──────────────────────────────────────────────────────────────────────────────┐
│                              BOTTLENECK                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐                                                         │
│  │   DoubleConv    │                                                         │
│  │   512 → 512     │                                                         │
│  │   32×32×512     │                                                         │
│  └────────┬────────┘                                                         │
└───────────┼──────────────────────────────────────────────────────────────────┘
            │                  │   │   │   │
            ▼                  ▼   │   │   │
┌──────────────────────────────────────────────────────────────────────────────┐
│                              DECODER PATH                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐      ┌─────────────────┐                               │
│  │ ConvTranspose2d │      │     CONCAT      │                               │
│  │   512 → 512     │─────►│  [up4, e4]      │──► DoubleConv(1024→512)       │
│  │   64×64×512     │      │   64×64×1024    │    64×64×512                  │
│  └─────────────────┘      └────────┬────────┘                               │
│                                    │              │                          │
│  ┌─────────────────┐      ┌────────▼────────┐    ▼                          │
│  │ ConvTranspose2d │      │     CONCAT      │                               │
│  │   512 → 256     │─────►│  [up3, e3]      │──► DoubleConv(512→256)        │
│  │   128×128×256   │      │  128×128×512    │    128×128×256                │
│  └─────────────────┘      └────────┬────────┘                               │
│                                    │              │                          │
│  ┌─────────────────┐      ┌────────▼────────┐    ▼                          │
│  │ ConvTranspose2d │      │     CONCAT      │                               │
│  │   256 → 128     │─────►│  [up2, e2]      │──► DoubleConv(256→128)        │
│  │   256×256×128   │      │  256×256×256    │    256×256×128                │
│  └─────────────────┘      └────────┬────────┘                               │
│                                    │              │                          │
│  ┌─────────────────┐      ┌────────▼────────┐    ▼                          │
│  │ ConvTranspose2d │      │     CONCAT      │                               │
│  │   128 → 64      │─────►│  [up1, e1]      │──► DoubleConv(128→64)         │
│  │   512×512×64    │      │  512×512×128    │    512×512×64                 │
│  └─────────────────┘      └────────┬────────┘                               │
│                                    │                                         │
└────────────────────────────────────┼─────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT LAYER                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐      │
│  │   Conv2d 1×1    │─────►│      Tanh       │─────►│   × 0.5 scale   │      │
│  │   64 → 3        │      │   [-1, +1]      │      │   [-0.5, +0.5]  │      │
│  │   512×512×3     │      │                 │      │   (RESIDUAL)    │      │
│  └─────────────────┘      └─────────────────┘      └────────┬────────┘      │
│                                                             │               │
│                              RESIDUAL ADDITION              │               │
│                    ┌────────────────┐                       │               │
│      INPUT ───────►│      ADD       │◄──────────────────────┘               │
│                    │   input + res  │                                       │
│                    └────────┬───────┘                                       │
│                             │                                               │
│                    ┌────────▼───────┐                                       │
│                    │     CLAMP      │                                       │
│                    │   [0, 1]       │                                       │
│                    └────────┬───────┘                                       │
│                             │                                               │
│                             ▼                                               │
│                   OUTPUT: Watermarked Image (512×512×3)                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 DoubleConvBlock Detail

```
┌─────────────────────────────────────────────────────────────────┐
│                     DoubleConvBlock                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (H × W × C_in)                                          │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                           │
│  │   Conv2d 3×3    │  kernel=3, padding=1 (preserves size)     │
│  │   C_in → C_out  │                                           │
│  └────────┬────────┘                                           │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  BatchNorm2d    │  Normalizes activations                   │
│  │    (C_out)      │                                           │
│  └────────┬────────┘                                           │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │      ReLU       │  Non-linearity                            │
│  │   (inplace)     │                                           │
│  └────────┬────────┘                                           │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │   Conv2d 3×3    │  kernel=3, padding=1                      │
│  │  C_out → C_out  │                                           │
│  └────────┬────────┘                                           │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  BatchNorm2d    │                                           │
│  │    (C_out)      │                                           │
│  └────────┬────────┘                                           │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │      ReLU       │                                           │
│  │   (inplace)     │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  Output (H × W × C_out)                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Architecture Summary Table

| Layer | Type | Input Size | Output Size | Parameters |
|-------|------|------------|-------------|------------|
| **ENCODER** |||||
| enc1 | DoubleConvBlock | 512×512×3 | 512×512×64 | 38,720 |
| pool1 | MaxPool2d(2,2) | 512×512×64 | 256×256×64 | 0 |
| enc2 | DoubleConvBlock | 256×256×64 | 256×256×128 | 221,440 |
| pool2 | MaxPool2d(2,2) | 256×256×128 | 128×128×128 | 0 |
| enc3 | DoubleConvBlock | 128×128×128 | 128×128×256 | 885,248 |
| pool3 | MaxPool2d(2,2) | 128×128×256 | 64×64×256 | 0 |
| enc4 | DoubleConvBlock | 64×64×256 | 64×64×512 | 3,539,968 |
| pool4 | MaxPool2d(2,2) | 64×64×512 | 32×32×512 | 0 |
| **BOTTLENECK** |||||
| bottleneck | DoubleConvBlock | 32×32×512 | 32×32×512 | 4,720,640 |
| **DECODER** |||||
| up4 | ConvTranspose2d | 32×32×512 | 64×64×512 | 1,049,088 |
| dec4 | DoubleConvBlock | 64×64×1024 | 64×64×512 | 7,079,424 |
| up3 | ConvTranspose2d | 64×64×512 | 128×128×256 | 524,544 |
| dec3 | DoubleConvBlock | 128×128×512 | 128×128×256 | 1,769,984 |
| up2 | ConvTranspose2d | 128×128×256 | 256×256×128 | 131,200 |
| dec2 | DoubleConvBlock | 256×256×256 | 256×256×128 | 442,624 |
| up1 | ConvTranspose2d | 256×256×128 | 512×512×64 | 32,832 |
| dec1 | DoubleConvBlock | 512×512×128 | 512×512×64 | 110,720 |
| **OUTPUT** |||||
| output | Conv2d(1×1) | 512×512×64 | 512×512×3 | 195 |
| tanh | Tanh | 512×512×3 | 512×512×3 | 0 |
| **TOTAL** |||| **~20.5M** |

---

## 4. Dataset Description

### 4.1 Dataset Source
- **Name**: Watermarked/Not-Watermarked Images Dataset
- **Source**: Kaggle (felicepollano/watermarked-not-watermarked-images)

### 4.2 Dataset Structure

```
data/wm-nowm/
├── train/
│   ├── watermark/          # 12,510 watermarked images
│   │   ├── image001.jpg
│   │   ├── image002.jpeg
│   │   └── ...
│   └── no-watermark/       # 12,477 clean images
│       ├── image001.jpg
│       ├── image002.jpeg
│       └── ...
└── valid/
    ├── watermark/
    └── no-watermark/
```

### 4.3 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Watermarked Images | 12,510 |
| Total Non-Watermarked Images | 12,477 |
| **Matched Pairs** | **1,744** |
| Training Pairs (80%) | 1,395 |
| Validation Pairs (20%) | 349 |
| Image Formats | JPG, JPEG, PNG |
| Training Resolution | 512 × 512 |

### 4.4 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │  Watermark   │         │ No-Watermark │                     │
│  │   Folder     │         │    Folder    │                     │
│  │  (12,510)    │         │   (12,477)   │                     │
│  └──────┬───────┘         └──────┬───────┘                     │
│         │                        │                              │
│         └──────────┬─────────────┘                              │
│                    ▼                                            │
│         ┌──────────────────┐                                    │
│         │ Filename Matching│  Match by name (ignore extension)  │
│         │   (1,744 pairs)  │                                    │
│         └────────┬─────────┘                                    │
│                  │                                              │
│         ┌────────▼─────────┐                                    │
│         │   Train/Val      │  80% / 20% split                   │
│         │     Split        │                                    │
│         └────────┬─────────┘                                    │
│                  │                                              │
│    ┌─────────────┴─────────────┐                               │
│    ▼                           ▼                                │
│  ┌────────────┐          ┌────────────┐                        │
│  │  Training  │          │ Validation │                        │
│  │   1,395    │          │    349     │                        │
│  │   pairs    │          │   pairs    │                        │
│  └─────┬──────┘          └─────┬──────┘                        │
│        │                       │                                │
│        ▼                       ▼                                │
│  ┌────────────────────────────────────────┐                    │
│  │           PREPROCESSING                 │                    │
│  │  1. Resize to 512×512                  │                    │
│  │  2. Normalize to [0, 1]                │                    │
│  │  3. BGR → RGB conversion               │                    │
│  │  4. HWC → CHW (PyTorch format)         │                    │
│  │  5. Data Augmentation (optional)       │                    │
│  └────────────────────────────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Training Process

### 5.1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Batch Size | 8 |
| Epochs | 100 (with early stopping) |
| Image Size | 512 × 512 |
| Device | NVIDIA GPU (CUDA) |

### 5.2 Loss Function

```
┌─────────────────────────────────────────────────────────────────┐
│                    WATERMARK LOSS FUNCTION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Total Loss = λ_mse × MSE_Loss                                  │
│             + λ_l1 × L1_Loss                                    │
│             + λ_diff × Difference_Penalty                       │
│             + λ_diff × Min_Diff_Penalty                         │
│                                                                 │
│  Where:                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ MSE_Loss = MSE(output, target)                          │   │
│  │           Pixel-wise reconstruction accuracy            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ L1_Loss = L1(output, target)                            │   │
│  │           Absolute difference for sharper edges         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Diff_Penalty = ReLU(target_diff - output_diff)          │   │
│  │           Penalize if output too similar to input       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Min_Diff_Penalty = ReLU(min_threshold - output_diff)    │   │
│  │           Force minimum visible change (≥2%)            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Hyperparameters:                                               │
│  • λ_mse = 1.0                                                  │
│  • λ_l1 = 0.5                                                   │
│  • λ_diff = 1.0                                                 │
│  • min_threshold = 0.02 (2% minimum difference)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  for epoch in range(100):                                       │
│      │                                                          │
│      ▼                                                          │
│  ┌───────────────────────────────────────────────┐             │
│  │              TRAINING PHASE                    │             │
│  │  for batch in train_loader:                   │             │
│  │      1. Forward pass: output = model(clean)   │             │
│  │      2. Compute loss(output, watermarked)     │             │
│  │      3. Backward pass: loss.backward()        │             │
│  │      4. Update weights: optimizer.step()      │             │
│  └───────────────────────────────────────────────┘             │
│      │                                                          │
│      ▼                                                          │
│  ┌───────────────────────────────────────────────┐             │
│  │            VALIDATION PHASE                    │             │
│  │  with torch.no_grad():                        │             │
│  │      for batch in val_loader:                 │             │
│  │          1. Forward pass                      │             │
│  │          2. Compute validation loss           │             │
│  └───────────────────────────────────────────────┘             │
│      │                                                          │
│      ▼                                                          │
│  ┌───────────────────────────────────────────────┐             │
│  │            CHECKPOINTING                       │             │
│  │  • Save best model (lowest val_loss)          │             │
│  │  • Save latest model                          │             │
│  │  • Save periodic (every 10 epochs)            │             │
│  │  • Early stopping (patience=20)               │             │
│  └───────────────────────────────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Residual Learning Concept

### 6.1 Why Residual Learning?

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL APPROACH                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Clean Image ──► Model ──► Watermarked Image                   │
│                                                                 │
│   Problem: Model must reconstruct ENTIRE image                  │
│   Risk: Model learns identity mapping (output = input)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    RESIDUAL APPROACH                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      ┌─────────────┐                            │
│                      │   Model     │                            │
│   Clean Image ──┬───►│  Predicts   │──► Watermark Residual      │
│                 │    │  RESIDUAL   │        │                   │
│                 │    └─────────────┘        │                   │
│                 │                           ▼                   │
│                 │                    ┌─────────────┐            │
│                 └───────────────────►│     ADD     │────► Output│
│                                      └─────────────┘            │
│                                                                 │
│   Output = Input + Residual × scale_factor                      │
│                                                                 │
│   Advantage: Model only learns WHAT TO ADD (watermark)          │
│   Forces non-identity mapping                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Mathematical Formulation

```
Traditional:  y = f(x)                    where y ≈ x (identity risk)

Residual:     y = x + g(x) × α           where g(x) = watermark overlay
                                               α = 0.5 (scale factor)

Constraints:  g(x) ∈ [-1, 1]  (tanh activation)
              y ∈ [0, 1]      (clamped output)
```

---

## 7. Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                                              │
│  │ Input Image  │  (any size, e.g., 1920×1080)                 │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────┐                      │
│  │         PREPROCESSING                 │                      │
│  │  1. Read image (OpenCV BGR)          │                      │
│  │  2. Store original dimensions        │                      │
│  │  3. Resize to 512×512                │                      │
│  │  4. Normalize [0, 255] → [0, 1]      │                      │
│  │  5. BGR → RGB                        │                      │
│  │  6. HWC → CHW                        │                      │
│  │  7. Add batch dimension              │                      │
│  │  8. Move to GPU                      │                      │
│  └──────────────┬───────────────────────┘                      │
│                 │                                               │
│                 ▼                                               │
│  ┌──────────────────────────────────────┐                      │
│  │           MODEL INFERENCE             │                      │
│  │  with torch.no_grad():               │                      │
│  │      output = model(input)           │                      │
│  └──────────────┬───────────────────────┘                      │
│                 │                                               │
│                 ▼                                               │
│  ┌──────────────────────────────────────┐                      │
│  │        POST-PROCESSING                │                      │
│  │  1. Remove batch dimension           │                      │
│  │  2. CHW → HWC                        │                      │
│  │  3. RGB → BGR                        │                      │
│  │  4. Denormalize [0, 1] → [0, 255]    │                      │
│  │  5. Convert to uint8                 │                      │
│  │  6. Resize to original dimensions    │                      │
│  └──────────────┬───────────────────────┘                      │
│                 │                                               │
│                 ▼                                               │
│  ┌──────────────┐                                              │
│  │ Output Image │  (original size, watermarked)                │
│  └──────────────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Technology Stack

| Component | Technology |
|-----------|------------|
| Deep Learning Framework | PyTorch 2.x |
| GPU Acceleration | CUDA (NVIDIA) |
| Image Processing | OpenCV (cv2) |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Progress Bars | tqdm |

---

## 9. Key Features

1. **U-Net Architecture**: Encoder-decoder with skip connections for preserving spatial details
2. **Residual Learning**: Model predicts watermark overlay, not full image reconstruction
3. **Custom Loss Function**: Combines MSE, L1, and difference penalties
4. **Comprehensive Checkpointing**: Best, latest, and periodic model saves
5. **Early Stopping**: Prevents overfitting (patience=20 epochs)
6. **GPU Acceleration**: Full CUDA support for fast training
7. **Flexible Inference**: Supports any input image size

---

## 10. Model Output

```
Model: WatermarkAutoencoder
Total Parameters: 20,554,819 (~20.5M)
Trainable Parameters: 20,554,819
Input Shape: (batch, 3, 512, 512)
Output Shape: (batch, 3, 512, 512)
```

---

*Report generated for Capstone Project - Watermark Addition Model*
