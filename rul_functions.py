import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

from IPython.display import display

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# 1. COLUMN NAMES
# =========================================================

def get_cmapss_column_names():
    """
    Return the standard C-MAPSS column names.
    """
    base_cols = ['unit_nr', 'cycle']
    setting_cols = [f'op_setting_{i}' for i in range(1, 4)]
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    return base_cols + setting_cols + sensor_cols


# =========================================================
# 2. DATA LOADING
# =========================================================

def load_cmapss_data(train_path, test_path, rul_path):
    """
    Load train, test, and RUL files for a C-MAPSS dataset.
    Removes trailing empty columns if they exist.
    """
    col_names = get_cmapss_column_names()

    train_df = pd.read_csv(train_path, sep=r'\s+', header=None)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None)
    rul_df = pd.read_csv(rul_path, sep=r'\s+', header=None)

    train_df = train_df.dropna(axis=1, how='all')
    test_df = test_df.dropna(axis=1, how='all')
    rul_df = rul_df.dropna(axis=1, how='all')

    train_df.columns = col_names
    test_df.columns = col_names
    rul_df.columns = ['RUL']

    return train_df, test_df, rul_df


# =========================================================
# 3. RUL CREATION
# =========================================================

def add_train_rul(train_df):
    """
    Compute RUL for training data:
    RUL = max cycle per engine - current cycle
    """
    train_df = train_df.copy()

    max_cycle = train_df.groupby('unit_nr')['cycle'].max().reset_index()
    max_cycle.columns = ['unit_nr', 'max_cycle']

    train_df = train_df.merge(max_cycle, on='unit_nr', how='left')
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
    train_df.drop(columns=['max_cycle'], inplace=True)

    return train_df


def add_test_rul(test_df, rul_df):
    """
    Compute true RUL for the official test set using the provided RUL file.
    """
    test_df = test_df.copy()

    max_cycle = test_df.groupby('unit_nr')['cycle'].max().reset_index()
    max_cycle.columns = ['unit_nr', 'max_cycle']
    max_cycle['final_rul'] = rul_df['RUL'].values

    test_df = test_df.merge(max_cycle, on='unit_nr', how='left')
    test_df['RUL'] = test_df['final_rul'] + (test_df['max_cycle'] - test_df['cycle'])
    test_df.drop(columns=['max_cycle', 'final_rul'], inplace=True)

    return test_df


# =========================================================
# 4. BASIC INSPECTION FUNCTIONS
# =========================================================

def basic_dataset_summary(df, name="Dataset"):
    """
    Print shape, head, and info summary.
    """
    print(f"{name} Shape: {df.shape}")
    display(df.head())
    print()
    print(df.info())


def missing_values_summary(df):
    """
    Return and display missing value counts and percentages.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100

    summary = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing %': missing_pct
    }).sort_values(by='Missing Count', ascending=False)

    display(summary[summary['Missing Count'] > 0])

    if summary['Missing Count'].sum() == 0:
        print("No missing values found.")

    return summary


def unique_value_summary(df):
    """
    Return unique value counts for each column.
    Useful for spotting constant / near-constant columns.
    """
    summary = pd.DataFrame({
        'Column': df.columns,
        'Unique Values': [df[col].nunique() for col in df.columns]
    }).sort_values(by='Unique Values')

    display(summary)
    return summary


def low_variance_summary(df, threshold=1e-8, exclude_cols=None):
    """
    Show numerical columns with very low variance.
    Does not drop anything.
    """
    if exclude_cols is None:
        exclude_cols = ['unit_nr', 'cycle', 'RUL']

    candidate_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_cols = df[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()

    variances = df[numeric_cols].var()

    summary = pd.DataFrame({
        'Column': variances.index,
        'Variance': variances.values
    }).sort_values(by='Variance')

    low_var = summary[summary['Variance'] <= threshold]

    print(f"Columns with variance <= {threshold}:")
    display(low_var)

    return summary


def zero_std_columns(df, exclude_cols=None):
    """
    Show columns with exactly zero standard deviation.
    """
    if exclude_cols is None:
        exclude_cols = ['unit_nr', 'cycle', 'RUL']

    candidate_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_cols = df[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()

    stds = df[numeric_cols].std()

    zero_std = stds[stds == 0].index.tolist()

    print("Zero standard deviation columns:")
    print(zero_std)

    return zero_std


# =========================================================
# 5. CORRELATION FUNCTIONS
# =========================================================

def correlation_with_rul(df, method='pearson'):
    """
    Return feature correlations with RUL, sorted by absolute value.
    """
    if 'RUL' not in df.columns:
        raise ValueError("RUL column not found. Please add RUL first.")

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    corr = numeric_df.corr(method=method)['RUL'].drop('RUL')
    corr_df = pd.DataFrame({
        'Feature': corr.index,
        'Correlation_with_RUL': corr.values,
        'Abs_Correlation': np.abs(corr.values)
    }).sort_values(by='Abs_Correlation', ascending=False)

    display(corr_df)
    return corr_df


def plot_top_correlations_with_rul(df, top_n=15, method='pearson', figsize=(10, 6)):
    """
    Plot top features by absolute correlation with RUL.
    """
    corr_df = correlation_with_rul(df, method=method).head(top_n)

    plt.figure(figsize=figsize)
    plt.barh(corr_df['Feature'][::-1], corr_df['Correlation_with_RUL'][::-1])
    plt.xlabel('Correlation with RUL')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Correlations with RUL')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap_subset(df, cols, figsize=(10, 8)):
    """
    Plot correlation heatmap for a selected set of columns.
    This uses matplotlib only.
    """
    corr = df[cols].corr()

    plt.figure(figsize=figsize)
    im = plt.imshow(corr, aspect='auto')
    plt.colorbar(im)

    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# =========================================================
# 6. MANUAL PREPROCESSING HELPERS
# =========================================================

def drop_columns(df, cols_to_drop):
    """
    Drop columns safely if they exist.
    """
    df = df.copy()
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    return df


def get_feature_columns(df):
    """
    Return feature columns excluding identifiers and target.
    """
    excluded = ['unit_nr', 'cycle', 'RUL']
    return [col for col in df.columns if col not in excluded]


def get_last_cycle_per_engine(df):
    """
    Keep only the last available row for each engine.
    Used for official tabular model test evaluation.
    """
    return df.groupby('unit_nr').tail(1).copy()


# =========================================================
# 7. ENGINE-LEVEL TRAIN/VALIDATION SPLIT
# =========================================================

def engine_level_train_val_split(train_df, test_size=0.2, random_state=42):
    """
    Split the training set by engine ID to prevent leakage.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = train_df['unit_nr']

    train_idx, val_idx = next(splitter.split(train_df, groups=groups))

    train_split = train_df.iloc[train_idx].copy()
    val_split = train_df.iloc[val_idx].copy()

    return train_split, val_split


# =========================================================
# 8. SCALING
# =========================================================

def scale_tabular_features(train_df, val_df, test_df, feature_cols):
    """
    Scale features using StandardScaler fitted on training split only.
    """
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val = scaler.transform(val_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    y_train = train_df['RUL'].values
    y_val = val_df['RUL'].values
    y_test = test_df['RUL'].values

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# =========================================================
# 9. METRICS
# =========================================================

def compute_regression_metrics(y_true, y_pred):
    """
    Compute MAE, RMSE, and R².
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }


# =========================================================
# 10. REGULAR ML MODELS
# =========================================================

def get_regression_models(random_state=42, include_xgboost=True):
    models = {
        'Linear Regression': LinearRegression(),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=random_state),
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=random_state,
            n_jobs=-1
        ),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    }

    if include_xgboost and XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=random_state
        )

    return models

def train_regression_models(X_train, y_train, random_state=42, include_xgboost=True):
    """
    Train all regular regression models.
    """
    models = get_regression_models(random_state=random_state, include_xgboost=include_xgboost)

    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def evaluate_regression_model(model, X, y_true):
    """
    Evaluate one regression model.
    """
    y_pred = model.predict(X)
    metrics = compute_regression_metrics(y_true, y_pred)

    return {
        'y_pred': y_pred,
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'R2': metrics['R2']
    }


def evaluate_all_models(models, X, y_true):
    """
    Evaluate all trained regular models and return a summary DataFrame.
    """
    results = []

    for name, model in models.items():
        output = evaluate_regression_model(model, X, y_true)
        results.append({
            'Model': name,
            'MAE': output['MAE'],
            'RMSE': output['RMSE'],
            'R2': output['R2']
        })

    results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False).reset_index(drop=True)
    display(results_df)
    return results_df


# =========================================================
# 11. PLOTTING FOR REGULAR MODELS
# =========================================================

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title(title)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, title="Residual Plot", figsize=(7, 5)):
    residuals = y_true - y_pred

    plt.figure(figsize=figsize)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, linestyle='--')
    plt.xlabel("Predicted RUL")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_residual_histogram(y_true, y_pred, title="Residual Histogram", bins=30, figsize=(7, 5)):
    residuals = y_true - y_pred

    plt.figure(figsize=figsize)
    plt.hist(residuals, bins=bins)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_sorted_actual_vs_predicted(y_true, y_pred, title="Sorted Actual vs Predicted", figsize=(10, 5)):
    sorted_idx = np.argsort(y_true)

    plt.figure(figsize=figsize)
    plt.plot(np.array(y_true)[sorted_idx], label='Actual')
    plt.plot(np.array(y_pred)[sorted_idx], label='Predicted')
    plt.xlabel("Samples sorted by Actual RUL")
    plt.ylabel("RUL")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()



# =========================================================
# 12. FINAL COMPARISON HELPERS
# =========================================================

def add_result_row(results_list, dataset_name, model_name, metrics_dict, split_name='Test'):
    """
    Append one result row into a list for later comparison.
    """
    results_list.append({
        'Dataset': dataset_name,
        'Split': split_name,
        'Model': model_name,
        'MAE': metrics_dict['MAE'],
        'RMSE': metrics_dict['RMSE'],
        'R2': metrics_dict['R2']
    })


def results_list_to_dataframe(results_list):
    """
    Convert accumulated result rows into a DataFrame.
    """
    df = pd.DataFrame(results_list)
    if not df.empty:
        df = df.sort_values(by=['Dataset', 'Split', 'R2'], ascending=[True, True, False]).reset_index(drop=True)
    display(df)
    return df
# =========================================================
# 13. CNN-SPECIFIC SCALING
# =========================================================

def scale_sequence_features(train_df, val_df, test_df, feature_cols):
    """
    Scale sequence-model features using StandardScaler fitted on
    the training split only, then return transformed dataframes.
    """
    scaler = StandardScaler()

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_scaled, val_scaled, test_scaled, scaler

# =========================================================
# 16. CNN TRAINING FUNCTION
# =========================================================

def train_cnn_model(
    X_train_seq,
    y_train_seq,
    X_val_seq,
    y_val_seq,
    num_features,
    seq_len,
    batch_size=64,
    lr=0.0005,
    num_epochs=40,
    patience=8,
    device=None
):
    """
    Train CNN with the same logic used in FD001.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_seq, dtype=torch.float32),
        torch.tensor(y_val_seq, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNN1DRegressor(num_features=num_features, seq_len=seq_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_state)

    return model, train_losses, val_losses

# =========================================================
# 17. CNN EVALUATION FUNCTION
# =========================================================

def evaluate_cnn_model(model, X_seq, y_true, device=None):
    """
    Evaluate trained CNN on sequence data.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    metrics = compute_regression_metrics(y_true, y_pred)

    return {
        'y_pred': y_pred,
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'R2': metrics['R2']
    }

class CNN1DRegressor(nn.Module):
    def __init__(self, num_features, seq_len):
        super(CNN1DRegressor, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(64 * (seq_len // 2), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (batch, features, seq_len)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x.squeeze(1)
    
def filter_late_cycles(df, threshold=125):
    return df[df['RUL'] <= threshold].copy()

def add_rolling_features(df, sensor_cols, window=5):
    df = df.copy()
    
    for col in sensor_cols:
        new_col = f"{col}_roll"
        df[new_col] = (
            df.groupby('unit_nr')[col]
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[new_col] = df.groupby('unit_nr')[new_col].fillna(method='bfill')
    
    return df