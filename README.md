# Turbofan Remaining Useful Life Prediction

Course project for **COSC 434 — Introduction to Machine Learning**.

**Students:** Karam Hasan (100064840) and Lara Al Omari (100064797)  
**Presented to:** Prof. Panos Liatsis, Dr. Ibrar Amin, and Dr. Kunlin Yang  
**Date:** 27 April 2026

## Problem

Aircraft engines fail gradually. The task is to estimate the remaining useful life (RUL): how many operating cycles are left before failure. That estimate supports maintenance before a breakdown.

Each row is one engine at one cycle. The inputs are three operational settings and 21 sensor readings. The target is a cycle count, so this is a regression problem. There is no class label and no confusion matrix. Error is reported with MAE, RMSE, and R².

## Dataset

The data is the NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set. It has four subsets. Training trajectories run until failure. Test trajectories stop early, and a separate file gives the true RUL of each test engine.

| Subset | Operating conditions | Fault modes | Difficulty |
| --- | --- | --- | --- |
| FD001 | 1 | 1 | Easy |
| FD002 | Multiple | 1 | Medium |
| FD003 | 1 | Multiple | Medium |
| FD004 | Multiple | Multiple | Hard |

Columns are `unit_nr`, `cycle`, `op_setting_1` to `op_setting_3`, and `sensor_1` to `sensor_21`. The files have no missing values. Some sensors are constant and are removed.

Training RUL is `max cycle − current cycle`. Early-life values are capped at 125 so the model focuses on degradation near failure. Official test scores use the last recorded cycle of each test engine.

The NASA text files are not stored in this repository. Download instructions are below.

## Methods

FD001 is explored in full in `docs/PreFinal-notebook.pdf`: loading, RUL construction, constant-feature removal, correlation, Random Forest feature importance, an engine-level 80/20 split, scaling, model comparison, 5-fold cross-validation, and Optuna tuning.

Splits are by engine id. Cycles from one engine stay in one split, so later cycles of the same unit cannot leak into validation.

`rul_functions.py` repeats that pipeline for FD002, FD003, and FD004: load, build RUL, drop low-information columns, split by engine, scale on the training engines only, train, and score.

Classical models:

- Linear Regression and Ridge
- K-Nearest Neighbors and RBF SVR
- Decision Tree, Random Forest, and XGBoost

Distance-based models use standardized features. Tree models use the selected features without scaling.

The sequence model is a 1D CNN over windows of 30 cycles, with two convolutional layers, max pooling, dropout, and a linear head. It is trained with Adam and MSE, with early stopping. Optuna searches filter counts, hidden size, learning rate, dropout, and batch size.

On FD001, Random Forest importance is led by `sensor_11`, then `cycle`, with `sensor_9`, `sensor_4`, `sensor_12`, and `sensor_7` also contributing.

## Results

Scores below are the test-set numbers saved in the analysis notebook.

| Dataset | Best model | MAE | RMSE | R² |
| --- | --- | --- | --- | --- |
| FD001 | 1D CNN, Optuna | 9.68 | 12.89 | 0.897 |
| FD002 | 1D CNN | 21.12 | 29.98 | 0.689 |
| FD003 | SVR | 15.40 | 20.39 | 0.757 |
| FD004 | XGBoost | 25.78 | 35.96 | 0.565 |

FD001 detail:

- Validation, one engine-level split: SVR reached R² 0.868 (MAE 9.91). Random Forest and XGBoost were close, at R² 0.847.
- 5-fold cross-validation: SVR, Random Forest, and XGBoost all sat near R² 0.83. The tree models had a smaller RMSE spread.
- Optuna Random Forest on the official test set: MAE 12.73, RMSE 17.25, R² 0.815.
- Optuna CNN on the same test set: MAE 9.68, RMSE 12.89, R² 0.897. Validation before the final test run was MAE 9.88, RMSE 13.39, R² 0.897.

FD002 test, classical models: SVR R² 0.671, XGBoost R² 0.666, Random Forest R² 0.659. The CNN improved that to R² 0.689.

FD003 test: SVR R² 0.757 beat the CNN test score of R² 0.691, even though CNN validation on FD003 reached R² 0.888.

FD004 test: XGBoost R² 0.565, Random Forest R² 0.547.

Accuracy falls as operating conditions and fault modes increase. The CNN is strongest when degradation is a clean trend. No single model wins every subset.

Plots from the notebook are in `results/`:

- `fd001_sensor4_over_cycles.png` shows sensor drift across cycles.
- `fd001_rul_capping.png` compares raw and capped RUL.
- `fd001_feature_importance.png` ranks Random Forest importance.
- `fd001_validation_rmse.png` compares validation RMSE.
- `fd001_random_forest_actual_vs_predicted.png` and `fd001_random_forest_sorted.png` show the tuned Random Forest.
- `fd001_cnn_training_loss.png`, `fd001_cnn_actual_vs_predicted.png`, and `fd001_cnn_sorted.png` show the FD001 CNN.
- `fd002_cnn_actual_vs_predicted.png` and `fd003_cnn_actual_vs_predicted.png` show the later subsets.

## Contribution

Karam Hasan and Lara Al Omari built this project together. The work covers the FD001 study, the shared functions in `rul_functions.py`, the comparison of classical models and the 1D CNN on all four subsets, and the course slides in `docs/`.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set from the [NASA Prognostics Center of Excellence data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and place the text files in `data/CMAPS/`:

```
data/CMAPS/train_FD001.txt
data/CMAPS/test_FD001.txt
data/CMAPS/RUL_FD001.txt
```

The same three names are required for FD002, FD003, and FD004.

The full written analysis, code, and saved figures are in `docs/PreFinal-notebook.pdf`. The course slides are in `docs/COSC434-RUL-presentation.pdf`. Import `rul_functions` from the repository root when reusing the FD002–FD004 pipeline. A full rerun includes Random Forest training and Optuna searches, so it takes a while on CPU.

## Repository layout

```
rul_functions.py                 Shared loading, RUL, split, scaling, models, and CNN helpers
requirements.txt
.gitignore
data/CMAPS/                      NASA text files go here
docs/PreFinal-notebook.pdf       Full notebook export
docs/COSC434-RUL-presentation.pdf
results/                         Plots saved from the analysis
```

## Dataset citation

A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,” in *Proceedings of the 1st International Conference on Prognostics and Health Management*, Denver, CO, 2008.
