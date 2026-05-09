# BharatBench — Data-Driven Weather Forecasting Over India

> Benchmarking weather forecasting models on the IMDAA dataset — from Linear Regression to Vision Transformers — for medium-range prediction over the Indian subcontinent.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/maslab/bharatbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What This Project Does

This project trains and evaluates a series of forecasting models — starting from simple baselines all the way to transformer architectures — to predict atmospheric variables 5 days ahead over India using the **BharatBench / IMDAA** dataset.

**Target variables:**
- `HGT_prl` → Geopotential Height at 500 hPa (H500)
- `TMP_prl` → Temperature at 850 hPa (T850) ← primary benchmark
- `TMP_2m` → 2-metre Surface Temperature
- `APCP_sfc` → 6-hourly Accumulated Precipitation

**Lead time:** 20 time steps = **5 days** | **Test period:** 2019–2020 | **Grid:** 32×32 at 1.08°

---

## Models Implemented

| # | Model | Location |
|---|---|---|
| 1 | Climatology & Persistence | `notebooks/1_climatology_persistence.ipynb` |
| 2 | Linear Regression | `notebooks/2_Linear_Regression.ipynb` |
| 3 | CNN + ConvLSTM | `notebooks/3_CNN_ConvLSTM.ipynb` |
| 4 | Vanilla Vision Transformer | `scripts/training/train_transformer.py` |
| 5 | Swin Transformer | `scripts/training/train_swin_transformer.py` |
| 6 | Multi-Variable ViT | `scripts/training/train_multivar_transformer.py` |

---

## Results (T850, 5-day lead time)

| Model | RMSE (K) ↓ | MAE (K) ↓ | ACC ↑ |
|---|---|---|---|
| Climatology | 2.518 | 1.782 | 0.894 |
| Linear Regression | 2.202 | 1.540 | 0.920 |
| CNN | 2.257 | 1.578 | 0.916 |
| ConvLSTM | 2.381 | 1.658 | 0.905 |
| Vanilla ViT | 2.281 | 1.607 | 0.914 |
| Swin Transformer | 2.396 | 1.671 | 0.904 |
| **Multi-Variable ViT** | **2.270** | **1.591** | **0.914** |

Full results in [`results/`](results/).

---

## Repository Structure

```
BharatBench/
│
├── notebooks/                          # Step-by-step Jupyter notebooks
│   ├── 1_climatology_persistence.ipynb
│   ├── 2_Linear_Regression.ipynb
│   ├── 3_CNN_ConvLSTM.ipynb
│   ├── 4_IMDAA_Regrid.ipynb            # Data preprocessing
│   ├── 5_Transformer.ipynb
│   ├── 6_Swin_Transformer.ipynb
│   └── 7_MultiVar_Transformer.ipynb
│
├── scripts/
│   ├── training/                       # Model training scripts
│   │   ├── train_transformer.py        # Vanilla ViT (--target flag)
│   │   ├── train_swin_transformer.py   # Swin Transformer
│   │   ├── train_multivar_transformer.py
│   │   └── train_all.sh                # Train all in sequence
│   │
│   ├── evaluation/                     # Evaluation scripts
│   │   ├── eval_transformer.py
│   │   ├── eval_multivar_transformer.py
│   │   └── eval_all_models.py
│   │
│   └── visualization/                  # Plot generation scripts
│       ├── plot_heatmaps.py            # Geographical heatmaps
│       ├── plot_lead_time_error.py     # Error vs lead time curves
│       ├── plot_metrics.py             # Model comparison bar chart
│       └── plot_paper_heatmaps.py      # Predicted vs actual fields
│
├── results/                            # Saved evaluation metrics
│   ├── transformer_all_metrics.json    # Vanilla ViT: H500, T2m, TP6h
│   ├── transformer_metrics.json        # Vanilla ViT: T850
│   ├── swin_metrics.json               # Swin: T850
│   ├── multivar_metrics.json           # Multi-Var ViT: T850
│   └── paper_benchmark_results.md      # All baseline numbers from paper
│
├── figures/                            # Generated plots
│   ├── geographical_heatmap.png
│   ├── lead_time_error.png
│   ├── metrics_comparison.png
│   └── paper_style_heatmaps.png
│
├── data/
│   └── india_boundary.geojson          # India shapefile for map plotting
│
├── Bharatbenchpaper.pdf                # Original BharatBench paper
├── environment.yaml                    # Conda environment
├── requirements.txt                    # pip dependencies
└── LICENSE
```

---

## Setup

### 1. Clone
```bash
git clone https://github.com/preetyorange/BharatBench.git
cd BharatBench
```

### 2. Install dependencies
```bash
# Conda (recommended)
conda env create -f environment.yaml
conda activate bharatbench

# or pip
pip install -r requirements.txt
```

### 3. Download the dataset
The `.nc` file (~708 MB) is not in the repo. Download from Kaggle:
```bash
pip install kaggle
kaggle datasets download -d maslab/bharatbench
unzip bharatbench.zip -d dataset-bharatbench/
```
Expected path: `dataset-bharatbench/IMDAA_merged_1.08_1990_2020.nc`

---

## Running the Code

### Option A — Notebooks (step-by-step)
Run notebooks in order from `notebooks/1_` → `notebooks/7_`. Each is self-contained.

### Option B — Training Scripts

```bash
# Train all transformer models
bash scripts/training/train_all.sh

# Train individually
python scripts/training/train_transformer.py --target TMP_prl    # T850
python scripts/training/train_transformer.py --target TMP_2m     # T2m
python scripts/training/train_transformer.py --target HGT_prl    # H500
python scripts/training/train_transformer.py --target APCP_sfc   # TP6h
python scripts/training/train_swin_transformer.py
python scripts/training/train_multivar_transformer.py
```

### Evaluate
```bash
python scripts/evaluation/eval_transformer.py
python scripts/evaluation/eval_multivar_transformer.py
python scripts/evaluation/eval_all_models.py
```

### Generate Plots
```bash
python scripts/visualization/plot_heatmaps.py
python scripts/visualization/plot_lead_time_error.py
python scripts/visualization/plot_metrics.py
python scripts/visualization/plot_paper_heatmaps.py
```

---

## Model Architecture Summary

### Vanilla ViT (`train_transformer.py`)
- Input: `(32, 32, 1)` — single variable
- Patches: 4×4 → 64 patches, projected to dim 64
- 4 transformer encoder blocks, 4 attention heads
- Decoder: reshape → 2× Conv2DTranspose → Conv2D output
- Optimizer: Adam lr=1e-4, MSE loss, early stopping patience=5

### Swin Transformer (`train_swin_transformer.py`)
- Input: `(32, 32, 1)`
- PatchExtract 2×2 → 16×16 feature map, dim=64
- 4 SwinTransformerBlocks: alternating shift_size=[0,2,0,2], window=4
- Relative position bias + cyclic shift masking
- Optimizer: Adam lr=5e-5, MSE loss

### Multi-Variable ViT (`train_multivar_transformer.py`)
- Input: `(32, 32, 4)` — HGT_prl, TMP_prl, TMP_2m, APCP_sfc stacked as channels
- Each variable normalized independently using training-set stats
- Same ViT encoder as Vanilla; outputs T850 only
- Optimizer: Adam lr=1e-3, MSE loss

---

## Data Split

| Split | Years | Purpose |
|---|---|---|
| Train | 1990–2017 | Model fitting |
| Validation | 2018 | Early stopping |
| Test | 2019–2020 | Final evaluation |

Normalization (mean/std) computed on training split only.

---

## Acknowledgements

Dataset: [IMDAA Reanalysis](https://rds.ncmrwf.gov.in/datasets) — NCMRWF, Ministry of Earth Sciences, Government of India.
Original paper: [arXiv:2405.07534](https://arxiv.org/abs/2405.07534)

---

## License
MIT — see [LICENSE](LICENSE)
