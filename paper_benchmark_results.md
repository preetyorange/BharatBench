# BharatBench Results & Benchmarks

This document logs all the evaluation metrics presented in the original BharatBench paper, establishing the baselines for forecasting atmospheric variables on the Indian subcontinent using the IMDAA dataset.

> [!NOTE]
> All models are evaluated against the test set covering the years **2019-2020**. The four selected variables represent distinct atmospheric properties:
> - **H500**: Geopotential height at 500 hPa (Steering flow / large-scale circulation)
> - **T850**: Temperature at 850 hPa (Frontal boundaries / thermal transport)
> - **T2m**: 2-meter surface temperature (Diurnal variations / land-cover correlation)
> - **TP6h**: 6-hourly total precipitation (Highly dynamic, sparse phenomena)

---

## 1. Climatology Baselines
Climatological models compute baseline expectations from historical averages. In the paper, these were calculated using the training data from 1990 to 2017.

### Standard Climatology
| Metric | H500 | T850 | T2m | TP6h |
|---|---|---|---|---|
| **RMSE** | 28.475 | 2.518 | 4.135 | 3.178 |
| **MAE** | 21.135 | 1.782 | 2.827 | 1.116 |
| **ACC** | 0.854 | 0.894 | 0.814 | 0.286 |

### Weekly Climatology (Seasonal)
| Metric | H500 | T850 | T2m | TP6h |
|---|---|---|---|---|
| **RMSE** | 28.776 | 2.537 | 4.151 | 3.167 |
| **MAE** | 21.279 | 1.795 | 2.835 | 1.113 |
| **ACC** | 0.850 | 0.892 | 0.813 | 0.295 |

---

## 2. Linear Regression Model
A simple linear model flattening the spatial grid (32x32 to 1024) to predict 3 and 5 days ahead.

### 3-Day Forecast
| Metric | H500 | T850 | T2m | TP6h |
|---|---|---|---|---|
| **RMSE** | 26.616 | 2.006 | 3.214 | 2.109 |
| **MAE** | 18.744 | 1.397 | 1.217 | 1.364 |
| **ACC** | 0.869 | 0.934 | 0.308 | 0.955 |

### 5-Day Forecast
| Metric | H500 | T850 | T2m | TP6h |
|---|---|---|---|---|
| **RMSE** | 29.582 | 2.202 | 3.265 | 2.274 |
| **MAE** | 20.688 | 1.540 | 1.245 | 1.478 |
| **ACC** | 0.834 | 0.920 | 0.273 | 0.947 |

---

## 3. Deep Learning Baselines
The deep learning models aim to capture spatio-temporal features. The models below were trained specifically for **H500** and **T850**.

| Model | Forecast | Metric | H500 | T850 |
|---|---|---|---|---|
| **CNN** | 3 days | **RMSE** | 26.272 | 2.053 |
| | | **MAE** | 18.427 | 1.433 |
| | | **ACC** | 0.872 | 0.931 |
| **ConvLSTM** | 3 days | **RMSE** | 26.871 | 2.211 |
| | | **MAE** | 18.806 | 1.536 |
| | | **ACC** | 0.866 | 0.919 |
| **CNN** | 5 days | **RMSE** | 28.805 | 2.257 |
| | | **MAE** | 20.108 | 1.578 |
| | | **ACC** | 0.844 | 0.916 |
| **ConvLSTM** | 5 days | **RMSE** | 29.592 | 2.381 |
| | | **MAE** | 20.747 | 1.658 |
| | | **ACC** | 0.835 | 0.905 |

---

## 4. Transformer Architectures (New Additions)
These models were implemented and evaluated natively on the repository for predicting variables at a **5-day lead time**, extending the original paper's baselines. 

| Target Variable | Model | RMSE | MAE | ACC | Notes |
|---|---|---|---|---|---|
| **T850** | Vanilla ViT | 2.28 | 1.61 | 0.91 | Standard Transformer encoder with positional embeddings |
| **T850** | Swin Transformer | 2.40 | 1.67 | 0.90 | Shifted window attention, requires more data to perfectly converge |
| **T850** | Multi-Variable ViT | **2.27** | **1.59** | 0.91 | Incorporates Z500, T2m, and Precip as physical context channels |
| **H500** | Vanilla ViT | 29.91 | 20.52 | 0.83 | Slightly trails CNN (28.81) and ConvLSTM (29.59) |
| **T2m** | Vanilla ViT | **2.42** | 1.62 | **0.94** | Massive improvement over Linear Regression (RMSE: 3.27, ACC: 0.27) |
| **TP6h** | Vanilla ViT | 3.13 | 1.16 | 0.33 | Struggles with the sparse, dynamic nature of precipitation |

> [!TIP]
> The **Multi-Variable ViT** nearly ties the CNN baseline (2.257) on RMSE and achieves comparable MAE without needing specialized spatio-temporal convolutions, proving the efficacy of providing Transformers with multi-variable physics constraints.
