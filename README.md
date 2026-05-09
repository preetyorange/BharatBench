# BharatBench — Data-Driven Weather Forecasting Over India

> **A comprehensive benchmarking study of statistical and deep learning models — from Linear Regression to Vision Transformers — for medium-range weather forecasting over the Indian subcontinent, built on the IMDAA reanalysis dataset.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/maslab/bharatbench)

---

## 📌 Overview

This project extends the original [BharatBench paper](https://arxiv.org/abs/2405.07534) by implementing and evaluating **three transformer-based architectures** alongside the paper's original baselines (climatology, linear regression, CNN, ConvLSTM) for 5-day lead-time weather forecasting over India.

### Target Variables
| Variable | Description | Unit |
|---|---|---|
| **H500** | Geopotential Height at 500 hPa | m |
| **T850** | Temperature at 850 hPa | K |
| **T2m** | 2-metre Surface Temperature | K |
| **TP6h** | 6-hourly Accumulated Precipitation | kg/m² |

### Models Implemented
| Model | Type | Notes |
|---|---|---|
| Climatology & Persistence | Statistical baseline | Paper baseline |
| Linear Regression | Statistical | Paper baseline |
| CNN | Deep Learning | Paper baseline |
| ConvLSTM | Deep Learning | Paper baseline |
| **Vanilla ViT** | Transformer | **This work** |
| **Swin Transformer** | Transformer | **This work** |
| **Multi-Variable ViT** | Transformer | **This work** |

---

## 🗂 Repository Structure

```
BharatBench-main/
│
├── 📓 Notebooks (step-by-step)
│   ├── 1_climatology_persistence.ipynb   # Baseline: Climatology & Persistence
│   ├── 2_Linear_Regression.ipynb         # Baseline: Linear Regression
│   ├── 3_CNN_ConvLSTM.ipynb              # Deep Learning: CNN & ConvLSTM
│   ├── 4_IMDAA_Regrid.ipynb              # Dataset preprocessing & regridding
│   ├── 5_Transformer.ipynb               # Vanilla Vision Transformer (ViT)
│   ├── 6_Swin_Transformer.ipynb          # Swin Transformer
│   └── 7_MultiVar_Transformer.ipynb      # Multi-Variable ViT
│
├── 🐍 Training Scripts
│   ├── train_transformer.py              # Train Vanilla ViT (all variables)
│   ├── train_swin_transformer.py         # Train Swin Transformer (T850)
│   └── train_multivar_transformer.py     # Train Multi-Variable ViT
│
├── 📊 Evaluation Scripts
│   ├── eval_transformer.py               # Evaluate Vanilla ViT
│   ├── eval_multivar_transformer.py      # Evaluate Multi-Variable ViT
│   └── eval_all_models.py                # Unified evaluation runner
│
├── 📈 Visualization Scripts
│   ├── plot_heatmaps.py                  # Geographical heatmaps over India
│   ├── plot_lead_time_error.py           # RMSE/MAE vs lead-time plots
│   ├── plot_metrics.py                   # Bar chart model comparison
│   └── plot_paper_heatmaps.py            # Paper-style prediction heatmaps
│
├── 📉 Results
│   ├── transformer_all_metrics.json      # Vanilla ViT results (H500, T2m, TP6h)
│   ├── transformer_metrics.json          # Vanilla ViT T850 results
│   ├── swin_metrics.json                 # Swin Transformer T850 results
│   ├── multivar_metrics.json             # Multi-Variable ViT T850 results
│   └── paper_benchmark_results.md        # All baseline results from paper
│
├── 🗺 Visualizations
│   ├── geographical_heatmap.png          # Mean field heatmap over India
│   ├── lead_time_error.png               # Lead-time vs error curves
│   ├── metrics_comparison.png            # Model comparison bar chart
│   └── paper_style_heatmaps.png          # Predicted vs actual fields
│
├── 📄 Report
│   └── Project_Report.pdf                # Full 44-page academic project report
│
├── india_boundary.geojson                # India map boundary for plotting
├── environment.yaml                      # Conda environment specification
├── requirements.txt                      # pip requirements
├── train_all.sh                          # Script to train all models sequentially
└── Bharatbenchpaper.pdf                  # Original BharatBench paper (reference)
```

---

## 📊 Key Results

All models evaluated on **5-day lead time**, test period **2019–2020**.

### T850 (Temperature at 850 hPa) — Primary Benchmark

| Model | RMSE (K) ↓ | MAE (K) ↓ | ACC ↑ |
|---|---|---|---|
| Climatology | 2.518 | 1.782 | 0.894 |
| Linear Regression | 2.202 | 1.540 | 0.920 |
| CNN | 2.257 | 1.578 | 0.916 |
| ConvLSTM | 2.381 | 1.658 | 0.905 |
| Vanilla ViT | 2.281 | 1.607 | 0.914 |
| Swin Transformer | 2.396 | 1.671 | 0.904 |
| **Multi-Variable ViT** | **2.270** | **1.591** | **0.914** |

### T2m (Surface Temperature) — Most Dramatic Improvement

| Model | RMSE (K) ↓ | ACC ↑ |
|---|---|---|
| Linear Regression | 3.265 | 0.273 |
| Climatology | 4.135 | 0.814 |
| **Vanilla ViT** | **2.421** | **0.941** |

> 🔑 **The Vision Transformer achieves 3.4× better ACC than Linear Regression for T2m**, demonstrating the power of global self-attention for capturing non-linear surface temperature dynamics.

### All Variables — Vanilla ViT (5-day)

| Variable | RMSE | MAE | ACC |
|---|---|---|---|
| H500 | 29.909 m | 20.516 m | 0.832 |
| T850 | 2.281 K | 1.607 K | 0.914 |
| T2m | 2.421 K | 1.615 K | **0.941** |
| TP6h | 3.131 kg/m² | 1.158 kg/m² | 0.334 |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/preetyorange/BharatBench.git
cd BharatBench
```

### 2. Set Up Environment

**Using Conda (recommended):**
```bash
conda env create -f environment.yaml
conda activate bharatbench
```

**Using pip:**
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

The dataset is **not included** in this repository due to its large size (~20 GB).

Download from Kaggle:
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d maslab/bharatbench
unzip bharatbench.zip -d dataset-bharatbench/
```

Or download manually from: https://www.kaggle.com/datasets/maslab/bharatbench

Place the NetCDF file at: `dataset-bharatbench/IMDAA_merged_1.08_1990_2020.nc`

### 4. Run the Notebooks

Follow the numbered notebooks in order:
```
1_climatology_persistence.ipynb  → Statistical baselines
2_Linear_Regression.ipynb        → Linear model
3_CNN_ConvLSTM.ipynb             → Deep learning baselines
4_IMDAA_Regrid.ipynb             → Data preprocessing
5_Transformer.ipynb              → Vanilla ViT
6_Swin_Transformer.ipynb         → Swin Transformer
7_MultiVar_Transformer.ipynb     → Multi-Variable ViT
```

### 5. Train Models from Scripts

```bash
# Train Vanilla ViT for all variables
bash train_all.sh

# Train individual models
python train_transformer.py --target TMP_prl    # T850
python train_transformer.py --target TMP_2m     # T2m
python train_transformer.py --target HGT_prl    # H500
python train_transformer.py --target APCP_sfc   # TP6h
python train_swin_transformer.py                # Swin (T850)
python train_multivar_transformer.py            # Multi-Var ViT (T850)
```

### 6. Evaluate

```bash
python eval_transformer.py
python eval_multivar_transformer.py
python eval_all_models.py
```

### 7. Generate Visualizations

```bash
python plot_heatmaps.py          # Geographical heatmaps
python plot_lead_time_error.py   # Lead-time error curves
python plot_metrics.py           # Model comparison chart
python plot_paper_heatmaps.py    # Paper-style prediction maps
```

---

## 🧠 Model Architectures

### Vanilla Vision Transformer (ViT)
- Input: 32×32 single-channel atmospheric field
- Patch size: 4×4 → 64 patches
- Embedding dim: 64, Heads: 4, Layers: 4
- Decoder: Reshape + 2× transposed convolutions → 32×32 output
- Optimizer: Adam (lr=1e-4), Loss: MSE

### Swin Transformer
- Input: 32×32 single-channel
- Patch embed: 2×2 → 16×16 feature map (dim=64)
- 4 alternating Swin blocks (window=4, shift=[0,2,0,2])
- Relative position bias for spatial encoding
- Optimizer: Adam (lr=5e-5), Loss: MSE

### Multi-Variable ViT
- Input: 32×32×4 (H500, T850, T2m, TP6h as channels)
- Architecture: identical to Vanilla ViT
- Output: single-channel T850 prediction
- Optimizer: Adam (lr=1e-3), Loss: MSE

---

## 📐 Evaluation Metrics

**RMSE** — Root Mean Square Error (penalises large errors):
$$\text{RMSE} = \frac{1}{N} \sum_i \sqrt{\frac{1}{N_{lat}N_{lon}} \sum_{j,k}(f_{ijk} - t_{ijk})^2}$$

**MAE** — Mean Absolute Error (linear error measure):
$$\text{MAE} = \frac{1}{N} \sum_i \frac{1}{N_{lat}N_{lon}} \sum_{j,k}|f_{ijk} - t_{ijk}|$$

**ACC** — Anomaly Correlation Coefficient (spatial skill vs climatology):
$$\text{ACC} = \frac{\sum f'_{ijk} t'_{ijk}}{\sqrt{\sum f'^2_{ijk} \cdot \sum t'^2_{ijk}}}$$

---

## 📁 Dataset Details

| Property | Value |
|---|---|
| Source | IMDAA Reanalysis (NCMRWF) |
| Domain | 5°N–40°N, 65°E–100°E |
| Resolution | 1.08° (32×32 grid) |
| Period | 1990–2020 |
| Temporal resolution | 6-hourly |
| Format | NetCDF (.nc) |
| Train / Val / Test | 1990–2017 / 2018 / 2019–2020 |

---

## 📄 Project Report

A full **44-page academic project report** is included at [`Project_Report.pdf`](Project_Report.pdf), covering:
- Introduction & Motivation
- Literature Review (NWP → Deep Learning → Transformers)
- Dataset & Methodology
- Results & Discussions
- Summary & Conclusions
- 25 APA-format References

---

## 📚 References

- Choudhury, A., Panda, J., & Mukherjee, A. (2024). *BharatBench: Dataset for data-driven weather forecasting over India*. arXiv:2405.07534
- Rani et al. (2021). *IMDAA: High-resolution satellite-era reanalysis for the Indian monsoon region*. Journal of Climate.
- Vaswani et al. (2017). *Attention is all you need*. NeurIPS.
- Liu et al. (2021). *Swin Transformer: Hierarchical vision transformer using shifted windows*. ICCV.
- Rasp et al. (2020). *WeatherBench: A benchmark dataset for data-driven weather forecasting*. JAMES.

---

## 🏛 Acknowledgements

Dataset provided by **NCMRWF, Ministry of Earth Sciences, Government of India** under the National Monsoon Mission. IMDAA reanalysis produced in collaboration with the UK Met Office and IMD.

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Department of Earth and Atmospheric Sciences</b><br>
  National Institute of Technology, Rourkela — Odisha 769008, India
</p>
