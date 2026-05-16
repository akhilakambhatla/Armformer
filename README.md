# ArmFormer — CBAM-Enhanced Transformer for Thermal Weapon Segmentation

A custom attention-based segmentation architecture using Convolutional Block Attention Module (CBAM) integrated with a transformer backbone, built on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for multiclass weapon segmentation in thermal imagery.

## Architecture

ArmFormer integrates CBAM attention modules into a transformer-style segmentation framework. Key features:
- **CBAM attention** — channel + spatial attention for thermal feature refinement
- **Multiclass segmentation** — 5-class output (4 weapon classes + background)
- Input resolution: 512×512
- Optimizer: AdamW (lr=6e-5)

## Repository Structure

```
armformer/
├── armformer_config.py             # Main MMSeg config (CBAM + AdamW)
├── train.ipynb                     # Training notebook
├── test.ipynb                      # Evaluation notebook
├── inference_demo.ipynb            # Inference + visualization
├── analysis.ipynb                  # Results analysis
├── configs/
│   └── _base_/
│       └── models/
│           └── armformer_base.py   # Base model definition
└── results/
    └── confusion_matrix.png        # Segmentation confusion matrix
```

## Setup

```bash
# Install MMSegmentation
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install mmsegmentation
```

CBAM modules are integrated into the MMSegmentation framework — they load automatically when the config is registered.

## Training

```bash
python tools/train.py armformer_config.py
```

Or use `train.ipynb` for an interactive session.

## Configuration

Key settings in `armformer_config.py`:
- Input: 512×512 crops
- Classes: 5 (`num_classes=5`)
- Optimizer: AdamW, lr=6e-5, weight_decay=0.01
- Head learning rate multiplier: 10×

## Dataset

Thermal multiclass weapon segmentation dataset (not included). Follows MMSegmentation dataset format.

## Results

See `results/confusion_matrix.png` for class-wise segmentation performance.
