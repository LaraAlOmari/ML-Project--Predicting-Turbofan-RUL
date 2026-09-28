# Predicting Remaining Useful Life of Turbofan Jet Engines

Machine learning course project for **COSC 434 — Introduction to Machine Learning**.

**Students:** Karam Hasan (100064840) and Lara Al Omari (100064797)  
**Presented to:** Prof. Panos Liatsis, Dr. Ibrar Amin, and Dr. Kunlin Yang  
**Presentation date:** 27 April 2026

The goal is to predict how many operating cycles a turbofan engine has left before failure. That quantity is the Remaining Useful Life (RUL). The work uses the NASA C-MAPSS turbofan degradation dataset and compares classical regression models with a 1D convolutional network.

## Problem

Unexpected engine failure is a safety and cost problem. Predictive maintenance estimates RUL from operational settings and sensor readings so maintenance can happen before a breakdown.

Each row is one engine at one cycle. Training trajectories run until failure. Test trajectories stop before failure, and a separate file gives the true RUL of each test engine at its last recorded cycle. This is a regression problem: predict a cycle count, not a class label.

## Dataset

The project uses all four C-MAPSS subsets. They share the same column layout and differ in operating conditions and fault modes.

| Subset | Operating conditions | Fault modes | Difficulty |
| --- | --- | --- | --- |
| FD001 | 1 | 1 | Easy |
| FD002 | Multiple | 1 | Medium |
| FD003 | 1 | Multiple | Medium |
| FD004 | Multiple | Multiple | Hard |

Each observation has:

- `unit_nr` — engine id
- `cycle` — time step within that engine
- `op_setting_1` to `op_setting_3` — operational settings
- `sensor_1` to `sensor_21` — sensor measurements

There are no missing values. Some sensors are constant and are dropped before modeling.

Training RUL for an engine is `max cycle − current cycle`, so the last cycle of a failed engine has RUL 0. Early-life RUL values are capped at 125 (`RUL = min(RUL, 125)`). That cap reduces the influence of healthy early cycles and focuses learning on the degradation region near failure.

The official test score uses the last available cycle of each test engine, compared with the RUL file (also capped at 125 where the analysis applies that cap).

## Method

FD001 is explored in full in `docs/PreFinal-notebook.pdf`: loading, RUL construction, constant-feature removal, correlation, Random Forest feature importance, an engine-level train/validation split, scaling, model comparison, 5-fold cross-validation, and Optuna tuning. That PDF is the notebook, with the code, notes, and results.

Splits are by engine id (80% train / 20% validation). Cycles from one engine stay in a single split, so the model cannot leak future cycles of the same unit into validation.

`rul_functions.py` packages the same steps so FD002, FD003, and FD004 reuse one pipeline: load data, build RUL, drop low-information columns, split by engine, scale on the training engines only, train, and score.

### Models

Classical models:

- Linear Regression and Ridge
- K-Nearest Neighbors and RBF SVR
- Decision Tree, Random Forest, and XGBoost

Distance-based models are trained on standardized features. Tree models use the unscaled selected features.

Deep model:

- 1D CNN over sliding windows of 30 cycles
- Two convolutional layers, max pooling, dropout, and a linear head
- Adam, MSE loss, and early stopping
- Optuna search over filter counts, hidden size, learning rate, dropout, and batch size on FD001 (and an experimental search on FD002)

### Metrics

- **MAE** — average absolute error in cycles
- **RMSE** — penalizes large misses
- **R²** — share of RUL variance explained by the model

## Results

Best reported test result on each subset, from the notebook PDF and the course slides:

| Dataset | Best model | MAE | RMSE | R² |
| --- | --- | --- | --- | --- |
| FD001 | 1D CNN (Optuna) | ~9.7 | ~12.9 | ~0.90 |
| FD002 | 1D CNN | ~21.1 | ~30.0 | ~0.69 |
| FD003 | SVR | ~15.4 | ~20.4 | ~0.76 |
| FD004 | XGBoost | ~25.8 | ~36.0 | ~0.57 |

On FD001, the Optuna-tuned Random Forest reached about MAE 12.7, RMSE 17.25, and R² 0.81. The CNN improved on that (about MAE 9.68, RMSE 12.89, R² 0.90) because it sees a short history of sensor readings instead of a single row.

Accuracy falls as the subset gets harder. Multiple operating conditions (FD002, FD004) and multiple fault modes (FD003, FD004) weaken the link between a sensor snapshot and RUL. CNN is strongest when the degradation pattern is clean (FD001, and still best on FD002). SVR is the best FD003 model. XGBoost is the best FD004 model. No single model wins every subset.

Feature importance on FD001 is dominated by `sensor_11`, then `cycle`, with `sensor_9`, `sensor_4`, `sensor_12`, and `sensor_7` also contributing. Correlation points the same way: `sensor_11`, `sensor_4`, and `cycle` move against RUL, while `sensor_12` and `sensor_7` move with it.

## Repository layout

```
rul_functions.py                    Shared loading, RUL, split, scaling, models, and CNN helpers
docs/COSC434-RUL-presentation.pdf   Course slideshow
docs/PreFinal-notebook.pdf          Full notebook: code, notes, and results
data/CMAPS/                         Place the NASA text files here (not included)
```

The notebook is `docs/PreFinal-notebook.pdf`. It is not stored as an `.ipynb` file.

## How to run

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download the NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set from the [NASA Prognostics Center of Excellence data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and unzip the `CMAPSSData` files into `data/CMAPS/`. The analysis expects these names:

```
data/CMAPS/train_FD001.txt
data/CMAPS/test_FD001.txt
data/CMAPS/RUL_FD001.txt
```

The same three files are required for FD002, FD003, and FD004.

3. Read the full analysis in `docs/PreFinal-notebook.pdf`. Import `rul_functions` from the repository root when reusing the FD002–FD004 pipeline. The data paths in that PDF are `data/CMAPS/...`.

FD001 training includes Random Forest fitting and Optuna studies (30 trials for Random Forest, 10 for the CNN). FD002–FD004 repeat model training and, for FD002 and FD003, CNN training. A full rerun takes a while on CPU.

## Dependencies

Listed in `requirements.txt`:

- numpy, pandas, matplotlib
- scikit-learn
- xgboost
- torch
- optuna

The notebook was developed in an Anaconda environment (`dsenv`) with Optuna 4.8 and NumPy 2.2.

## Takeaways

- Capping RUL and dropping constant sensors makes the learning problem focus on degradation.
- Engine-level splits are required. A random row split would leak the same engine into train and validation.
- Sequence models help when degradation is a clear trend over time.
- On the harder subsets, dataset structure limits accuracy more than the choice of a deeper model.
- Later work suggested in the notebook PDF and the slides: LSTM or other sequence models, richer temporal features, and a more careful treatment of multiple operating conditions.

## Dataset citation

A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,” in *Proceedings of the 1st International Conference on Prognostics and Health Management*, Denver, CO, 2008.

Data: NASA Prognostics Center of Excellence, Turbofan Engine Degradation Simulation Data Set.
