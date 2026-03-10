# Weather Temperature Prediction Using Apache Spark MLlib

A scalable, end-to-end machine learning pipeline for predicting surface air temperature using large-scale meteorological data from the **NOAA Global Hourly Integrated Surface Dataset (ISD)**. Built on **Apache Spark MLlib** and deployed on **Google Cloud Platform**, the project benchmarks three regression models across 100M+ weather observations.

**Group Members:**
- Akshat Atul Bhargava (A0326699H)
- Choo Li Ying (A0172633H)
- Goh Chi Min (A0183716Y)

---

## Project Overview

| Detail | Value |
|---|---|
| **Dataset** | NOAA ISD Global Hourly 2024 |
| **Raw Dataset Size** | ~130M rows |
| **Post-Cleaning Size** | 17,953,096 rows |
| **Target Variable** | Air Temperature (°C) |
| **Best Model** | Gradient Boosted Trees |
| **Best RMSE** | 3.43°C |
| **Best R²** | 0.9249 |
| **Platform** | Google Cloud Dataproc (asia-southeast1) |

---

## Architecture

```
NOAA ISD Dataset (GCS Bucket: dsa5208-mllib-proj)
        │
        ▼
  Dataproc Cluster (Apache Spark)
        │
        ├── Schema Alignment & Data Ingestion
        ├── Sentinel Value Removal
        ├── Feature Engineering & Parsing
        ├── Parquet Storage (GCS)
        │
        ├── Generalized Linear Regression (GLR)
        ├── Random Forest Regressor (RF)
        └── Gradient Boosted Trees (GBT)
```

---

## Features

- **Large-Scale Ingestion** of thousands of CSV files from GCS with automatic schema alignment
- **Robust Preprocessing** — sentinel value removal, column parsing, feature engineering
- **Three ML Models** benchmarked: GLR (linear baseline), RF (bagging), GBT (boosting)
- **Hyperparameter Tuning** via CrossValidator (GLR) and TrainValidationSplit (RF, GBT)
- **Feature Importance Analysis** across all models
- **Residual Analysis** and RMSE heatmap visualizations
- **Parquet Checkpointing** to avoid rerunning heavy preprocessing on large datasets
- **Distributed Computing** across primary and preemptible worker nodes on GCP Dataproc

---

## Dataset

**Source:** [NOAA Global Hourly ISD — 2024](https://www.ncei.noaa.gov/data/global-hourly/archive/csv/)

Hourly surface weather observations from thousands of meteorological stations worldwide.

**Features Used:**

| Feature | Description |
|---|---|
| `latitude` | Station latitude |
| `longitude` | Station longitude |
| `elevation` | Station elevation |
| `date_numeric` | Days since 1970-01-01 (engineered) |
| `wnd_speed` | Wind speed |
| `cig_height` | Cloud ceiling height |
| `vis_dist` | Visibility distance |
| `dew_cel` | Dew point temperature (°C) |
| `slp_hpa` | Sea-level pressure (hPa) |
| `tmp_cel` | **Air temperature (°C) — target** |

---

## Preprocessing Pipeline

### 1. Schema Alignment
- Canonicalized column names (unicode normalization, BOM removal, lowercase + underscore)
- Missing columns added as NULL to enforce a fixed schema across all CSV files

### 2. Multi-Value Column Parsing
- `split_column()` — parses WND, CIG, VIS, SLP into distinct subcolumns
- `split_signed_tenths()` — converts TMP and DEW from tenths-of-degrees integers to double-precision Celsius

### 3. Sentinel Value Removal
- Referenced NOAA ISD Federal Climate Complex Data Documentation
- Removed rows containing sentinel codes: `['3', '7', '9', '99', '999', '9999', '99999', '999999', '+3', '+7', ...]`
- **112,269,010 rows removed** → 17,953,096 clean rows retained
- Rows dropped (not imputed) since sentinels represent instrument failures, not statistical missingness

### 4. Feature Engineering
- `date_numeric`: timestamp → days since epoch (continuous temporal feature)
- `slp_hpa`: rescaled from tenths of hPa → standard hPa (÷10)
- `wnd_dir`: encoded as sin/cos components to handle circularity (optional, disabled for speed)

### 5. Parquet Storage
- Cleaned dataset saved to GCS in Parquet format
- Avoids re-running preprocessing across experiments; prevents kernel disconnections on large data

---

## ML Models

### a. Generalized Linear Regression (GLR)
- Gaussian family, identity link (standard multiple linear regression)
- Tuned via **5-fold CrossValidator**
- Parameter grid: `regParam ∈ {0.0, 0.01, 0.1, 0.5}`, `maxIter ∈ {50, 100}`
- Best config: `regParam=0.0`, `maxIter=100`

### b. Random Forest Regressor (RF)
- Ensemble of decision trees (bagging); averages predictions to reduce variance
- Tuned via **TrainValidationSplit** (80/20) for computational efficiency at scale
- Parameter grid: `maxDepth ∈ {5,10}`, `numTrees ∈ {5,10,15}`, `subsamplingRate ∈ {0.6,0.8}`
- Best config: `maxDepth=10`, `numTrees=15`, `subsamplingRate=0.6`

### c. Gradient Boosted Trees (GBT) -> Best Model
- Sequential boosting — each tree corrects residual errors of the previous
- Tuned via **TrainValidationSplit** (80/20)
- Parameter grid: `maxDepth ∈ {7,9,11}`, `maxIter ∈ {50,60,70}`, `stepSize ∈ {0.05}`
- Best config: `maxDepth=11`, `maxIter=70`, `stepSize=0.05`, `subsamplingRate=0.6`

---

## Results

### Model Performance Comparison

| Model | Test RMSE (°C) | R² | MAE (°C) | Key Takeaway |
|---|---|---|---|---|
| Generalized Linear Regression | 5.30 | 0.822 | 3.75 | Interpretable linear baseline |
| Random Forest | 4.03 | 0.897 | 2.95 | Captures non-linearities well |
| **Gradient Boosted Trees** | **3.43** | **0.925** | **2.45** | Best performance overall |

### Top Feature Importances (consistent across all models)
| Rank | Feature | Description |
|---|---|---|
| 1 | `dew_cel` | Dew point temperature — dominant predictor |
| 2 | `latitude` | Spatial dependency (equator = warmer) |
| 3 | `date_numeric` | Temporal/seasonal patterns |
| 4 | `cig_height` | Cloud ceiling height |
| 5 | `slp_hpa` | Sea-level pressure |

---

## GCP Setup

### Cloud Storage
- Bucket: `dsa5208-mllib-proj` (asia-southeast1, Singapore)
- Contains raw CSVs, cleaned Parquet files, and model checkpoints

### Dataproc Cluster Config
| Setting | Value |
|---|---|
| Region | asia-southeast1 (Singapore) |
| Master Node | n4-standard-2 |
| Primary Worker Nodes | 2 × n4-standard-2 |
| Secondary (Preemptible) Nodes | 2–4 |
| Disk | 50GB hyperdisk-balanced |
| Components | Jupyter |

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **Big Data** | Apache Spark (PySpark), Spark MLlib |
| **Cloud** | Google Cloud Platform — Dataproc, GCS |
| **ML Models** | GLR, Random Forest, Gradient Boosted Trees |
| **Tuning** | CrossValidator, TrainValidationSplit, ParamGridBuilder |
| **Visualization** | Matplotlib, Seaborn |
| **Storage** | Parquet (columnar, GCS-backed) |
| **Environment** | Jupyter Notebook (Dataproc Component Gateway) |

---

## Repository Structure

```
├── notebooks/
│   └── weather_ml_pipeline.ipynb     # Main Jupyter notebook
├── data/
│   └── 2024.tar.gz                   # Raw NOAA ISD dataset (download separately)
├── parquet/                          # Cleaned dataset (Parquet format, GCS)
├── models/                           # Saved model checkpoints
├── outputs/                          # Visualizations, RMSE heatmaps, residual plots
└── README.md
```

---

## Notes

- Raw dataset must be downloaded from [NOAA ISD](https://www.ncei.noaa.gov/data/global-hourly/archive/csv/) and uploaded to GCS before running the pipeline
- Sentinel value removal is designed for the 2024 ISD release — verify codes against the [NOAA ISD documentation](https://www.ncei.noaa.gov/pub/data/ish/ish-format-document.pdf) for other years
- Wind direction sin/cos encoding is implemented but disabled by default for speed; can be re-enabled in the feature engineering cell

---

## References

1. NOAA National Centers for Environmental Information (NCEI). *Global Hourly Integrated Surface Dataset (ISD), 2024.* https://www.ncei.noaa.gov/data/global-hourly/archive/csv/
2. Federal Climate Complex Data Documentation for Integrated Surface Data (ISD), NOAA.
3. [Apache Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)

---

## Authors

**Akshat Atul Bhargava** · **Choo Li Ying** · **Goh Chi Min**
M.Sc. Data Science — National University of Singapore (NUS)
DSA5208 — Scalable Distributed Systems
