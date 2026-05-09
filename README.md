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

**Lead time:** 20 time steps = **5 days**  
**Test period:** 2019–2020  
**Grid:** 32×32 at 1.08° resolution, domain 5°N–40°N, 65°E–100°E

---

## Models Implemented

| # | Model | Script / Notebook | Notes |
|---|---|---|---|
| 1 | Climatology & Persistence | `1_climatology_persistence.ipynb` | Simplest baselines |
| 2 | Linear Regression | `2_Linear_Regression.ipynb` | Flattens 32×32 → 1024 |
| 3 | CNN + ConvLSTM | `3_CNN_ConvLSTM.ipynb` | Encoder-decoder, 17 layers |
| 4 | Vanilla Vision Transformer | `train_transformer.py` / `5_Transformer.ipynb` | Patch-based, global attention |
| 5 | Swin Transformer | `train_swin_transformer.py` / `6_Swin_Transformer.ipynb` | Shifted window attention |
| 6 | Multi-Variable ViT | `train_multivar_transformer.py` / `7_MultiVar_Transformer.ipynb` | 4-channel input fusion |

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

**T2m result:** Vanilla ViT achieves ACC **0.941** vs Linear Regression **0.273** — a 3.4× improvement.

Full per-variable results in:
- `transformer_all_metrics.json` — H500, T2m, TP6h
- `transformer_metrics.json` — T850 (Vanilla ViT)
- `swin_metrics.json` — T850 (Swin)
- `multivar_metrics.json` — T850 (Multi-Var ViT)
- `paper_benchmark_results.md` — all baseline numbers

---

## Repository Structure

```
├── 1_climatology_persistence.ipynb   # Climatology & persistence baselines
├── 2_Linear_Regression.ipynb         # Linear regression baseline
├── 3_CNN_ConvLSTM.ipynb              # CNN and ConvLSTM models
├── 4_IMDAA_Regrid.ipynb              # Data preprocessing / regridding
├── 5_Transformer.ipynb               # Vanilla ViT training + eval
├── 6_Swin_Transformer.ipynb          # Swin Transformer training + eval
├── 7_MultiVar_Transformer.ipynb      # Multi-variable ViT
│
├── train_transformer.py              # Train Vanilla ViT (--target flag for variable)
├── train_swin_transformer.py         # Train Swin Transformer (T850)
├── train_multivar_transformer.py     # Train Multi-Variable ViT (T850)
├── train_all.sh                      # Run all transformer training in sequence
│
├── eval_transformer.py               # Evaluate Vanilla ViT
├── eval_multivar_transformer.py      # Evaluate Multi-Variable ViT
├── eval_all_models.py                # Unified evaluation runner
│
├── plot_heatmaps.py                  # Geographical heatmaps (Cartopy + GeoJSON)
├── plot_lead_time_error.py           # RMSE/MAE vs lead time curves
├── plot_metrics.py                   # Bar chart comparing all models
├── plot_paper_heatmaps.py            # Predicted vs actual field maps
│
├── geographical_heatmap.png          # Output: mean variable fields over India
├── lead_time_error.png               # Output: error growth with lead time
├── metrics_comparison.png            # Output: model comparison chart
├── paper_style_heatmaps.png          # Output: prediction vs ground truth
│
├── transformer_all_metrics.json      # Vanilla ViT results (H500, T2m, TP6h)
├── transformer_metrics.json          # Vanilla ViT T850
├── swin_metrics.json                 # Swin T850
├── multivar_metrics.json             # Multi-Var ViT T850
├── paper_benchmark_results.md        # All paper baseline numbers
│
├── india_boundary.geojson            # India shapefile for map plotting
├── Bharatbenchpaper.pdf              # Original BharatBench paper (reference)
├── environment.yaml                  # Conda environment
└── requirements.txt                  # pip dependencies
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
The code expects: `dataset-bharatbench/IMDAA_merged_1.08_1990_2020.nc`

---

## Running the Code

### Option A — Notebooks (recommended for step-by-step)
Run the notebooks in order `1_` → `7_`. Each is self-contained.

### Option B — Scripts (for training transformers)

**Train all transformer models:**
```bash
bash train_all.sh
```

**Train individually:**
```bash
# Vanilla ViT — choose target variable
python train_transformer.py --target TMP_prl    # T850
python train_transformer.py --target TMP_2m     # T2m
python train_transformer.py --target HGT_prl    # H500
python train_transformer.py --target APCP_sfc   # TP6h

# Swin Transformer (T850 only)
python train_swin_transformer.py

# Multi-Variable ViT (T850, uses 4 input channels)
python train_multivar_transformer.py
```

**Evaluate:**
```bash
python eval_transformer.py
python eval_multivar_transformer.py
python eval_all_models.py
```

**Generate plots:**
```bash
python plot_heatmaps.py           # Geographical mean field heatmap
python plot_lead_time_error.py    # Error vs lead time
python plot_metrics.py            # Model comparison bar chart
python plot_paper_heatmaps.py     # Predicted vs actual maps
```

---

## Model Architecture Summary

### Vanilla ViT (`train_transformer.py`)
- Input: `(32, 32, 1)` — single variable
- Patches: 4×4 → 64 patches, projected to dim 64
- 4 transformer encoder blocks, 4 attention heads
- Decoder: reshape → 2× Conv2DTranspose → Conv2D output
- Optimizer: Adam lr=1e-4, loss=MSE, early stopping patience=5

### Swin Transformer (`train_swin_transformer.py`)
- Input: `(32, 32, 1)`
- PatchExtract 2×2 → 16×16 feature map, dim=64
- 4 SwinTransformerBlocks alternating shift_size=[0,2,0,2], window=4
- Relative position bias; shifted cyclic masking
- Decoder: Conv2DTranspose → Conv2D
- Optimizer: Adam lr=5e-5, loss=MSE

### Multi-Variable ViT (`train_multivar_transformer.py`)
- Input: `(32, 32, 4)` — HGT_prl, TMP_prl, TMP_2m, APCP_sfc stacked
- Each variable normalized independently using training-set stats
- Same ViT architecture as Vanilla; output is T850 only
- Optimizer: Adam lr=1e-3, loss=MSE

---

## Data Split

| Split | Years | Purpose |
|---|---|---|
| Train | 1990–2017 | Model fitting |
| Validation | 2018 | Hyperparameter tuning / early stopping |
| Test | 2019–2020 | Final evaluation (never seen during training) |

Normalization (mean/std) computed on training split only and applied to val/test.

---

## Evaluation Metrics

- **RMSE** — Root Mean Square Error (spatial + temporal average)
- **MAE** — Mean Absolute Error
- **ACC** — Anomaly Correlation Coefficient (skill relative to climatology, range −1 to +1)

---

## Acknowledgements

Dataset: [IMDAA Reanalysis](https://rds.ncmrwf.gov.in/datasets) — NCMRWF, Ministry of Earth Sciences, Government of India.  
Original BharatBench paper: [arXiv:2405.07534](https://arxiv.org/abs/2405.07534)

---

## License
MIT — see [LICENSE](LICENSE)
