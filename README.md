# GAN-Based Forensic Framework for Detecting Forged Digital Documents

> **Research Paper:** *A GAN-Based Forensic Framework for Detecting Forged Digital Medical Certificates*  
> **Authors:** Mrigashi, Dr. Shashi Kumar — Bennett University  
> **Dataset:** [DocTamper (Qu et al., CVPR 2023)](https://github.com/qcf-568/DocTamper) via Kaggle

---

## Overview

This project presents a comparative study of **seven deep learning architectures** for binary classification of document images as authentic or forged. It addresses a gap in existing literature: most forgery detection studies evaluate only a single architecture, making cross-model comparison difficult. We benchmark all seven models under identical experimental conditions on the large-scale DocTamper dataset (170,000 document images).

**Top result:** EfficientNet-B0 and the Hybrid (EfficientNet-B0 + ResNet50) ensemble both achieve **100% accuracy, precision, recall, and F1-score** on the test set.

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| CNN (custom) | 99.98% | 99.96% | 100.0% | 99.98% |
| ResNet50 | 99.98% | 99.96% | 100.0% | 99.98% |
| **EfficientNet-B0** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| Vision Transformer (ViT) | 99.96% | 100.0% | 99.93% | 99.96% |
| GAN Discriminator | 99.96% | 99.93% | 100.0% | 99.96% |
| **Hybrid (EfficientNet + ResNet50)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| VAE-GAN | 99.83% | 99.86% | 99.94% | 99.90% |

---

## Models

### 1. Custom CNN
Three-block convolutional network (32 → 64 → 128 channels), each block with Conv2d + BatchNorm + ReLU + MaxPool. FC layers: 32,768 → 256 → 2 with 0.5 dropout. Trained for 12 epochs.

### 2. ResNet50
Pretrained ResNet50 (ImageNet) fine-tuned end-to-end. Final FC replaced with `Linear(2048 → 2)`. No frozen layers. Trained for 10 epochs.

### 3. EfficientNet-B0
Pretrained EfficientNet-B0 fine-tuned with `Linear(1280 → 2)` head. Compound scaling optimizes depth, width, and resolution simultaneously. Trained for 10 epochs.

### 4. Vision Transformer (ViT)
`vit_tiny_patch16_224` from `timm`. Input images resized to 224×224 and split into 16×16 patches, passed through multi-head self-attention blocks. Batch size reduced to 4 due to memory constraints. Trained for 10 epochs.

### 5. GAN Discriminator (`LightDiscriminator`)
Standalone binary classifier with three strided conv layers (3 → 32 → 64 → 128), BatchNorm on last two layers, and `Linear(32,768 → 1)` with sigmoid. Trained with Binary Cross-Entropy, threshold = 0.5.

### 6. Hybrid (EfficientNet-B0 + ResNet50)
Both pretrained backbones frozen; classifier layers replaced with `Identity()`. Their feature vectors (1,280 + 2,048 = 3,328-d) are concatenated and passed to `Linear(3328 → 512) → ReLU → Dropout(0.3) → Linear(512 → 2)`. Only the classifier is trained. 5 epochs.

### 7. VAE-GAN (One-Class Learning)
Trained exclusively on real documents — no forged examples needed. At inference, anomaly score = `0.7 × pixel MSE + 0.3 × KL divergence`. Threshold optimized via ROC curve. Achieves 99.83% despite never seeing a forgery during training.

---

## Dataset

**DocTamper** (Qu et al., CVPR 2023) — 170,000 bilingual document images (contracts, invoices, receipts, certificates) with authentic and machine-tampered variants.

Due to compute constraints on Google Colab, a balanced subset of **16,000 images** (8,000 real + 8,000 fake) was used.

- Train / Test split: **80 / 20** → 12,800 train, 3,200 test
- Images resized to **128×128** (224×224 for ViT)

---

## Data Augmentation (Train Only)

| Augmentation | Value |
|---|---|
| Random Horizontal Flip | p=0.5 |
| Random Rotation | ±25° |
| Color Jitter | brightness ±0.3, contrast ±0.3 |
| Random Affine | ±15° |
| Gaussian Blur | kernel size 3 |

No augmentation applied at test time — resize and tensor conversion only.

---

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0003 (discriminative models), 0.0002 (VAE-GAN) |
| Batch Size | 128 (CNN, ResNet50, EfficientNet, GAN, Hybrid), 4 (ViT), 64 (VAE-GAN) |
| Environment | Google Colab (GPU, CUDA) |

---

## Project Structure

```
├── medical.ipynb        
├── README.md
```

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision timm efficientnet_pytorch scikit-learn tqdm matplotlib seaborn
```

### Running the Notebook

1. Upload `archive.zip` (DocTamper dataset from Kaggle) to your Google Drive
2. Open `medical.ipynb` in Google Colab
3. Mount Drive and update the zip path in Cell 2
4. Run all cells sequentially — each section is labeled by model name

### Dataset

Download DocTamper from Kaggle: [DocTamper Dataset](https://www.kaggle.com/datasets/...)

---

## Key Findings

- **EfficientNet's compound scaling** makes it particularly well-suited for visual forensic tasks — it captures fine-grained texture and layout artifacts that signal forgery.
- **The Hybrid ensemble** improves diversity by combining EfficientNet's efficiency-optimized features (1,280-d) with ResNet50's depth-optimized features (2,048-d), reaching the same perfect score without training the backbones.
- **VAE-GAN's 99.83% accuracy** — achieved with zero forged training examples — suggests this approach is viable in real-world scenarios where collecting labeled forgeries is difficult.
- **ViT trades recall for precision** (99.93% recall vs 100% precision), meaning it occasionally misses a forgery. This is consistent with known ViT behavior on smaller datasets.

---

## Limitations

- DocTamper contains general documents (contracts, invoices), not domain-specific medical certificates. Performance on real medical certificates may differ, especially for human-crafted forgeries not present in the training distribution.
- All models trained at 128×128 — subtle forgery signals (font inconsistencies, stamp mismatches) may be lost at this resolution.
- Forgeries in DocTamper are machine-generated (mask-based), not hand-crafted by a human forger.

---

## Future Work

- Collect a domain-specific medical certificate dataset with real forgeries
- Train at higher resolutions (512×512+) to capture fine-grained artifacts
- Extend to forgery localization (pixel-level segmentation, not just classification)
- Evaluate adversarial robustness of the best-performing models

---


## Author

**Mrigashi** | B.Tech CSE (AI & ML), Bennett University  
📧 e23cseu0327@bennett.edu.in | [LinkedIn](https://linkedin.com/in/mrigashi) | [GitHub](https://github.com/mrigashi)
