# ArmFormer: Lightweight Transformer Architecture for Real-Time Multi-Class Weapon Segmentation and Classification

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2510.16854-b31b1b.svg)](https://arxiv.org/abs/2510.16854)
[![IEEE](https://img.shields.io/badge/IEEE-Published-00629B.svg)](https://ieeexplore.ieee.org/document/11402089)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MMSegmentation](https://img.shields.io/badge/Framework-MMSegmentation-green.svg)](https://github.com/open-mmlab/mmsegmentation)


[📄 Paper (arXiv)](https://arxiv.org/abs/2510.16854) · [📄 Paper (IEEE)](https://ieeexplore.ieee.org/document/11402089) · [📊 Results](#results) · [🚀 Quick Start](#quick-start)

</div>

---

## Abstract

> The escalating threat of weapon-related violence necessitates automated detection systems capable of pixel-level precision for accurate threat assessment in real-time security applications. Traditional weapon detection approaches rely on object detection frameworks that provide only coarse bounding box localizations, lacking the fine-grained segmentation required for comprehensive threat analysis. This paper presents **ArmFormer**, a lightweight transformer-based semantic segmentation framework that strategically integrates Convolutional Block Attention Module (CBAM) with MixVisionTransformer architecture to achieve superior accuracy while maintaining computational efficiency suitable for resource-constrained edge devices. Comprehensive experiments demonstrate that ArmFormer achieves **80.64% mIoU** and **89.13% mFscore** while maintaining real-time inference at **82.26 FPS** — with only 4.886G FLOPs and 3.66M parameters, outperforming heavyweight models requiring up to **48× more computation**.

---

## Architecture

```
                        Input Image (512 × 512)
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════════╗
║                  CBAM-Enhanced MixVisionTransformer Encoder          ║
║                                                                      ║
║  Stage 1: PatchEmbed(k=7, s=4) ──► Transformer Blocks ──► CBAM      ║
║           └──────────────────────────────────────────────────────►  F₁  (H/4,  32ch)
║                                                                      ║
║  Stage 2: OverlapPatchMerge(k=3, s=2) ──► Transformer Blocks ──► CBAM
║           └──────────────────────────────────────────────────────►  F₂  (H/8,  64ch)
║                                                                      ║
║  Stage 3: OverlapPatchMerge(k=3, s=2) ──► Transformer Blocks ──► CBAM
║           └──────────────────────────────────────────────────────►  F₃  (H/16, 160ch)
║                                                                      ║
║  Stage 4: OverlapPatchMerge(k=3, s=2) ──► Transformer Blocks ──► CBAM
║           └──────────────────────────────────────────────────────►  F₄  (H/32, 256ch)
╚══════════════════════════════════════════════════════════════════════╝
                               │
                    Concat {F₁, F₂, F₃, F₄}
                    Bilinear Upsample → Unified Scale
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════════╗
║              CBAM-Integrated Hamburger Decoder Head                  ║
║                                                                      ║
║   [F₁ ‖ F₂ ‖ F₃ ‖ F₄]                                              ║
║         │                                                            ║
║         ▼                                                            ║
║      CBAM₁  ──► Channel Attention (MLP, r=16)                       ║
║               ──► Spatial Attention (Conv7×7)                        ║
║         │                                                            ║
║         ▼                                                            ║
║    Hamburger Module  (Global context via matrix decomposition)       ║
║         │                                                            ║
║         ▼                                                            ║
║      CBAM₂  ──► Channel Attention + Spatial Attention               ║
║         │                                                            ║
║         ▼                                                            ║
║   Classification Head  (1×1 Conv → N classes)                       ║
╚══════════════════════════════════════════════════════════════════════╝
                               │
                               ▼
              Segmentation Map  (5 weapon classes)
         [Handgun | Rifle | Knife | Revolver | Human]
```

**CBAM Attention** — applied uniformly across all stages (r=16, k=7):

| Module | Formula |
|--------|---------|
| Channel Attention | `Mc = σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))` |
| Spatial Attention | `Ms = σ(Conv7×7([AvgPoolc(F′) ; MaxPoolc(F′)]))` |
| Output | `Fout = Ms ⊗ (Mc ⊗ F)` |

---

## Summary

ArmFormer introduces a CBAM-enhanced MixVisionTransformer backbone paired with a dual-CBAM hamburger decoder for pixel-precise weapon segmentation in security-critical scenarios. The architecture achieves **80.64% mIoU** at **82.26 FPS** with only **3.66M parameters and 4.886G FLOPs** — making it deployable on edge devices like surveillance drones and embedded AI accelerators. It outperforms all compared baselines including heavyweight models with up to 48× more computation, while maintaining superior per-class segmentation across all five weapon categories. The model was trained on a custom 8,097-image dataset annotated semi-automatically using SAM2 and sourced from Google Open Images and IMFDB.

---

## Results

### Overall Performance Comparison

| Model | FLOPs (G) | Params (M) | Speed (FPS) | mIoU (%) | mAcc (%) | mFscore (%) |
|-------|:---------:|:----------:|:-----------:|:--------:|:--------:|:-----------:|
| **ArmFormer (Ours)** | **4.886** | **3.66** | 82.26 | **80.64** | **88.28** | **89.13** |
| EncNet | 54.56 | 12.52 | 90.78 | 77.65 | 81.65 | 80.63 |
| ICNet | 15.434 | 47.52 | 140.88 | 74.16 | 82.41 | 84.63 |
| Uppernet\_Swin | 236.0 | 58.94 | 38.97 | 70.90 | 79.53 | 82.56 |
| HrNet | 6.52 | 1.87 | 64.92 | 69.24 | 78.31 | 81.55 |
| PspNet | 18.14 | 4.571 | 56.96 | 66.20 | 75.12 | 77.95 |
| CGNet | 3.452 | 0.493 | 90.49 | 64.30 | 74.96 | 77.08 |
| Segmenter | 12.266 | 6.685 | 74.15 | 31.46 | 41.46 | 50.86 |

> ArmFormer surpasses the second-best model (EncNet) by **+2.99% mIoU** while using **11.2× fewer FLOPs** and **3.4× fewer parameters**.

### Per-Class IoU (%)

| Model | Handgun | Human | Knife | Rifle | Revolver |
|-------|:-------:|:-----:|:-----:|:-----:|:--------:|
| **ArmFormer (Ours)** | **82.24** | **67.22** | 80.81 | **83.87** | **80.43** |
| EncNet | 83.31 | 58.77 | **81.83** | 80.17 | 72.12 |
| ICNet | 78.72 | 50.12 | 76.83 | 77.46 | 75.65 |
| Uppernet\_Swin | 74.88 | 52.10 | 73.56 | 74.23 | 66.52 |
| HrNet | 72.78 | 58.08 | 61.83 | 70.01 | 69.08 |
| PspNet | 81.82 | 55.81 | 77.45 | 29.80 | 73.18 |
| CGNet | 71.13 | 32.92 | 66.42 | 72.67 | 61.90 |
| Segmenter | 33.19 | 7.41 | 59.24 | 11.56 | 22.02 |

### Ablation Study

| Configuration | mIoU (%) | mFscore (%) | FPS |
|--------------|:--------:|:-----------:|:---:|
| **ArmFormer (Ours)** | **80.64** | **89.13** | 82.26 |
| w/ FPN Neck | 79.92 | 88.70 | 78.85 |
| w/ Lightweight CBAM (r=32, k=3) | 79.74 | 88.57 | 84.28 |
| w/ Adaptive CBAM (stage-specific) | 78.05 | 87.54 | 81.15 |

**Per-Class IoU — Ablation (%):**

| Configuration | Background | Handgun | Human | Knife | Rifle | Revolver |
|--------------|:----------:|:-------:|:-----:|:-----:|:-----:|:--------:|
| **ArmFormer** | **89.29** | **82.24** | **67.22** | 80.81 | **83.87** | **80.43** |
| w/ FPN Neck | 89.01 | 79.65 | 67.97 | **81.60** | 82.38 | 78.94 |
| w/ Lightweight CBAM | **89.46** | 79.88 | 66.90 | **82.87** | 81.32 | 78.04 |
| w/ Adaptive CBAM | 87.20 | 79.30 | 66.77 | 79.95 | 76.52 | 78.53 |

---

## Dataset

Custom dataset curated from **Google Open Images** and **IMFDB** (Internet Movie Firearms Database), annotated semi-automatically using **SAM2 (Segment Anything Model 2)**.

| Class | Train | Val | Test | Total |
|-------|:-----:|:---:|:----:|:-----:|
| Handgun | 1,506 | 440 | 215 | 2,161 |
| Revolver | 1,280 | 345 | 172 | 1,797 |
| Rifle | 1,115 | 328 | 167 | 1,610 |
| Knife | 1,017 | 288 | 145 | 1,450 |
| Human | 928 | 269 | 133 | 1,330 |
| **Total** | **5,846** | **1,670** | **832** | **8,097** |

**Annotation Format:** Grayscale mask IDs — Handgun: 51, Human: 102, Knife: 153, Rifle: 204, Revolver: 255

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/akhilakambhatla/armformer.git
cd armformer

# Install MMSegmentation ecosystem
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install mmsegmentation

# Install remaining dependencies
pip install -r requirements.txt  # if present
```

### Training

```bash
# Using MMSegmentation tools
python tools/train.py armformer_config.py

# Or use the interactive notebook
jupyter notebook train.ipynb
```

### Evaluation

```bash
python tools/test.py armformer_config.py <checkpoint.pth> --eval mIoU

# Or use the interactive notebook
jupyter notebook test.ipynb
```

### Inference & Visualization

```bash
# Run the demo notebook
jupyter notebook inference_demo.ipynb
```

---

## Model Configuration

Key settings from `armformer_config.py`:

| Parameter | Value |
|-----------|-------|
| Input Resolution | 512 × 512 |
| Backbone | CBAM-Enhanced MixVisionTransformer |
| Decoder | CBAM-Integrated Hamburger Head |
| Optimizer | AdamW |
| Learning Rate | 6e-5 |
| Weight Decay | 0.01 |
| LR Schedule | Polynomial decay |
| Training Iterations | 160,000 |
| Batch Size | 2 (train), 1 (val/test) |
| CBAM Reduction Ratio | r = 16 |
| CBAM Spatial Kernel | k = 7 |
| Classes | 5 (Handgun, Rifle, Knife, Revolver, Human) |
| Loss | Pixel-wise Cross-Entropy |
| Parameters | 3.66M |
| FLOPs | 4.886G |
| Inference Speed | 82.26 FPS |

---

## Repository Structure

```
armformer/
├── armformer_config.py             # Main training configuration
├── train.ipynb                     # Interactive training notebook
├── test.ipynb                      # Evaluation and metric computation
├── inference_demo.ipynb            # Inference + segmentation visualization
├── analysis.ipynb                  # Results analysis and plots
├── configs/
│   └── _base_/
│       └── models/
│           └── armformer_base.py   # Base model definition
└── results/
    └── confusion_matrix.png        # Per-class segmentation confusion matrix
```

---

## Citation

If you find ArmFormer useful in your research, please cite our paper:

```bibtex
@article{kambhatla2025armformer,
  title     = {ArmFormer: Lightweight Transformer Architecture for Real-Time
               Multi-Class Weapon Segmentation and Classification},
  author    = {Kambhatla, Akhila and Islam, Taminul and Ahmed, Khaled R},
  journal   = {arXiv preprint arXiv:2510.16854},
  year      = {2025},
  url       = {https://arxiv.org/abs/2510.16854}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
