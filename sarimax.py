import pandas as pd
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

# Ensure output directory exists
output_dir = "output_graphs"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
file_path = "data.csv"  # Ensure the correct file path
ridge_results_path = "ridge_results.csv"
data = pd.read_csv(file_path)
data['DATE'] = pd.to_datetime(data['DATE'])
data.sort_values(by=['ATM_ID', 'DATE'], inplace=True)


def check_stationarity(series):
    """Perform Augmented Dickey-Fuller test to check stationarity."""
    result = adfuller(series.dropna())
    return result[1] < 0.05  # If p-value < 0.05, the series is stationary


# Define Feature Sets
feature_sets = {
    "F0": ["ATM_WITHDRWLS_1WEEKAGO"],
    "F1": ["ATM_WITHDRWLS_1WEEKAGO", "HOLIDAY", "AVG_ATM_WITHDRW_PREVMONTH", "DAY_OF_WEEK"],
    "F2": ["ATM_WITHDRWLS_1WEEKAGO", "HOLIDAY", "AVG_ATM_WITHDRW_PREVMONTH", "DAY_OF_WEEK", "MONTH", "FIRSTWORKDAY",
           "LASTWORKDAY"],
    "F3": ["ATM_WITHDRWLS_1WEEKAGO", "ATM_WITHDRWLS_2WEEKAGO", "HOLIDAY", "AVG_ATM_WITHDRW_PREVMONTH", "DAY_OF_WEEK",
           "IsSaturday", "IsSunday"],
    "F4": ["ATM_WITHDRWLS_1WEEKAGO", "HOLIDAY", "AVG_ATM_WITHDRW_PREVMONTH", "DAY_OF_WEEK", "MONTH", "FIRSTWORKDAY",
           "LASTWORKDAY", "ATM_WITHDRWLS_2WEEKAGO", "BRA_WITHDRWLS", "BRA_DEPOSITS", "IsSaturday", "IsSunday"],
    "F5": ["ATM_WITHDRWLS_1WEEKAGO", "HOLIDAY", "AVG_ATM_WITHDRW_PREVMONTH", "ATM_WITHDRWLS_2WEEKAGO", "DAY_OF_WEEK",
           "MONTH", "FIRSTWORKDAY", "LASTWORKDAY", "BRA_WITHDRWL_RATIO", "AVG_BRA_WITHDRWL_PREVMONTH", "BRA_WITHDRWLS",
           "BRA_DEPOSITS", "IsSaturday", "IsSunday"]
}

# ATMs to analyze
atms_to_analyze = data["ATM_ID"].unique()

# Hyperparameter tuning grid
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]
param_combinations = list(itertools.product(p_values, d_values, q_values))

# Results storage
combined_results = []

# Process each ATM
for atm_id in atms_to_analyze:
    atm_data = data[data['ATM_ID'] == atm_id]
    atm_data.set_index('DATE', inplace=True)

    # Check if differencing is needed
    d_param = 1 if not check_stationarity(atm_data['ATM_WITHDRWLS']) else 0

    # Iterate through each feature set
    for feature_set_name, feature_set in feature_sets.items():
        selected_columns = ["ATM_WITHDRWLS"] + [col for col in feature_set if col in atm_data.columns]
        atm_data_filtered = atm_data[selected_columns].dropna()

        # Split dynamically (80% train, 20% test)
        split_index = int(len(atm_data_filtered) * 0.8)
        train = atm_data_filtered.iloc[:split_index]
        test = atm_data_filtered.iloc[split_index:]

        best_model = None
        best_aic = float("inf")
        best_order = None

        # Grid search for best SARIMAX parameters
        for order in param_combinations:
            try:
                model = SARIMAX(train["ATM_WITHDRWLS"], exog=train[feature_set], order=(order[0], d_param, order[2]))
                fitted_model = model.fit(disp=False)
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
                    best_order = order
            except:
                continue

        # Predict on both train and test set
        train_pred = best_model.predict(start=0, end=len(train) - 1, exog=train[feature_set])
        test_pred = best_model.predict(start=len(train), end=len(train) + len(test) - 1, exog=test[feature_set])

        # Calculate errors
        train_mae = mean_absolute_error(train["ATM_WITHDRWLS"], train_pred)
        test_mae = mean_absolute_error(test["ATM_WITHDRWLS"], test_pred)
        train_rmse = np.sqrt(mean_squared_error(train["ATM_WITHDRWLS"], train_pred))
        test_rmse = np.sqrt(mean_squared_error(test["ATM_WITHDRWLS"], test_pred))
        train_mape = (train_mae / train["ATM_WITHDRWLS"].mean()) * 100 if train["ATM_WITHDRWLS"].mean() != 0 else None
        test_mape = (test_mae / test["ATM_WITHDRWLS"].mean()) * 100 if test["ATM_WITHDRWLS"].mean() != 0 else None

        # Store results
        combined_results.append({
            'ATM_ID': atm_id,
            'Feature_Set': feature_set_name,
            'Best Order': best_order,
            'MAE (Train/Test)': f"{train_mae:.2f} / {test_mae:.2f}",
            'RMSE (Train/Test)': f"{train_rmse:.2f} / {test_rmse:.2f}",
            'MAPE (Train/Test)': f"{train_mape:.2f} / {test_mape:.2f}"
        })

# Save results
results_df = pd.DataFrame(combined_results)
results_df.to_csv("SARIMAX_optimized_results.csv", index=False)
print("Optimized SARIMAX model results with Train/Test metrics saved.")
