# Partial Cross-Entropy Loss for Weakly-Supervised Remote Sensing Segmentation

## Overview

This project implements and evaluates **Partial Cross-Entropy (pCE)** for weakly-supervised semantic segmentation on the ISPRS Potsdam benchmark. The central idea is to train a dense segmentation model using only sparse point annotations, rather than full pixel-wise supervision.

The training objective is:

```
pCE = Σ(FocalLoss(pred, GT) × MASK_labeled) / Σ(MASK_labeled)
```

`MASK_labeled` is binary (`1` at labeled pixels, `0` elsewhere), so unlabeled pixels contribute zero gradient while labeled pixels carry all supervision.

## System Architecture

For a detailed exploration of the system architecture, including design rationale, data flow, component responsibilities, tradeoffs, and deployment-oriented considerations, refer to the [System Design Document](docs/system_design.md). It provides a complete blueprint of the end-to-end pipeline and links naturally with the [Technical Report](docs/technical_report.md), which covers formulation details, experiment protocol, and result interpretation.

## Key Features

- **Sparse-supervision training objective** using masked focal-weighted pCE
- **End-to-end experiment pipeline** in [notebook.ipynb](notebook.ipynb), from data staging to final plots
- **Two ablation studies**: point-density sensitivity and focal-gamma sensitivity
- **Runtime-aware execution** for local kernels and Colab-style `/content` workflows
- **Resumable experiments** with checkpointing and summary caching for long sweeps
- **Tested loss module** via focused unit tests in [tests/test_loss.py](tests/test_loss.py)

## Technical Components

### Data Pipeline

- Dataset: ISPRS Potsdam (6 semantic classes)
- Conversion scripts: [tools/dataset_converters/custom_potsdam.py](tools/dataset_converters/custom_potsdam.py) and [tools/dataset_converters/mmsegmentation_potsdam.py](tools/dataset_converters/mmsegmentation_potsdam.py)
- Cropped MMSeg-style layout under `data/potsdam/img_dir` and `data/potsdam/ann_dir`

`tools/dataset_converters/mmsegmentation_potsdam.py` is the repository copy of the original Potsdam converter logic from the official MMSegmentation source (`tools/dataset_converters/potsdam.py`). For this project runtime, the upstream conversion path was not directly usable because of mmcv/mmengine compatibility constraints, so `tools/dataset_converters/custom_potsdam.py` was developed to replicate the same core conversion behavior using Pillow, NumPy, and tqdm.

### Model and Optimization

- Model: DeepLabV3+ with ResNet-50 encoder (ImageNet pretrained)
- Loss: Partial Cross-Entropy with configurable focal gamma
- Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-4`)
- Scheduler: CosineAnnealingLR (`eta_min=1e-6`)
- Tile size: 512x512

### Evaluation

- Metrics: validation mIoU and pixel accuracy
- Evaluation target: full validation masks (not sparse point masks)

## Final Reported Results

These are the finalized metrics used across the submission documents.

### Experiment A: Point Density Ablation (`γ = 2.0`)

| Density | Labeled px / 512x512 tile | Best mIoU | Pixel Acc | Convergence Epoch |
|---|---:|---:|---:|---:|
| 0.001 (0.1%) | 262 | 0.068 | 0.19 | 22 |
| 0.005 (0.5%) | 1,310 | 0.074 | 0.21 | 18 |
| 0.01 (1.0%) | 2,621 | 0.076 | 0.22 | 16 |
| 0.05 (5.0%) | 13,107 | 0.072 | 0.20 | 14 |

### Experiment B: Focal Gamma Ablation (`density = 0.01`)

| Gamma | Best mIoU | Pixel Acc | Convergence Epoch | Relative Gain vs Gamma 0 |
|---:|---:|---:|---:|---:|
| 0.0 | 0.072 | 0.20 | 18 | baseline |
| 0.5 | 0.073 | 0.21 | 17 | +1.4% |
| 1.0 | 0.074 | 0.21 | 16 | +2.7% |
| 2.0 | 0.075 | 0.22 | 15 | +4.1% |

Best operating point: `density = 0.01`, `gamma = 2.0`, `best mIoU = 0.076`.

## Repository Structure

```text
remote-sensing-segmentation-pipeline/
├── README.md
├── LICENSE
├── requirements.txt
├── notebook.ipynb
├── src/
│   └── loss.py
├── tests/
│   └── test_loss.py
├── tools/
│   └── dataset_converters/
│       ├── custom_potsdam.py
│       └── mmsegmentation_potsdam.py
├── docs/
│   ├── technical_report.md
│   └── system_design.md
├── data/
│   ├── dataset/
│   └── potsdam/
├── checkpoints/
└── figures/
```

## Getting Started

### 1. Create and activate environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Validate installation

```bash
python -m pytest tests/test_loss.py -v
```

## Data Preparation

The notebook supports:

- Local workflow (VS Code / local kernel)
- Hosted workflow (Colab-style runtime with staged data)

Recommended local layout:

```text
data/potsdam/
├── img_dir/
│   ├── train/
│   └── val/
└── ann_dir/
    ├── train/
    └── val/
```

If starting from raw Potsdam archives:

```bash
python tools/dataset_converters/custom_potsdam.py data/dataset --out_dir data/potsdam --clip_size 512 --stride_size 256
```

Note: the custom converter above mirrors the official MMSegmentation Potsdam conversion logic maintained in `tools/dataset_converters/mmsegmentation_potsdam.py`, and is used in this repository specifically to avoid mmcv/mmengine incompatibility issues in the active environment.

## Running Experiments

Launch notebook:

```bash
jupyter notebook notebook.ipynb
```

Run cells in order from top to bottom.

### Run Modes

- `FAST_DEV_RUN`: reduced epochs/subsets for quick validation
- `FULL_RUN`: full sweep for reportable runs

```bash
export RUN_MODE=FAST_DEV_RUN
# or
export RUN_MODE=FULL_RUN
```

### Optional Single-Run Overrides

```bash
export EXPERIMENT_A_DENSITY_OVERRIDE=0.01
export EXPERIMENT_B_GAMMA_OVERRIDE=2.0
```

Unset to restore full sweeps:

```bash
unset EXPERIMENT_A_DENSITY_OVERRIDE
unset EXPERIMENT_B_GAMMA_OVERRIDE
```

### Output Artifacts

- `checkpoints/`: model checkpoints
- `figures/`: plots and qualitative outputs
- `results/`: experiment summary JSON files

Roots can be overridden via environment variables such as `POTSDAM_DATA_ROOT` and `POTSDAM_OUTPUT_ROOT`.

## Supporting Documents

- [docs/system_design.md](docs/system_design.md): architecture and engineering rationale
- [docs/technical_report.md](docs/technical_report.md): method detail, experiments, and analysis

## License

MIT License.

Dataset usage is governed by ISPRS benchmark terms.
