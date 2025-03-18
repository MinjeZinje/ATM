import pandas as pd
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Ensure output directory exists
output_dir = "sarimax_rolling_results"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
file_path = "data.csv"  # Ensure the correct file path
data = pd.read_csv(file_path)
data['DATE'] = pd.to_datetime(data['DATE'])
data.sort_values(by=['ATM_ID', 'DATE'], inplace=True)

# Define Feature Set F5
feature_set = [
    "ATM_WITHDRWLS_1WEEKAGO", "HOLIDAY", "AVG_ATM_WITHDRW_PREVMONTH",
    "ATM_WITHDRWLS_2WEEKAGO", "DAY_OF_WEEK", "MONTH", "FIRSTWORKDAY", "LASTWORKDAY",
    "BRA_WITHDRWL_RATIO", "AVG_BRA_WITHDRWL_PREVMONTH", "BRA_WITHDRWLS",
    "BRA_DEPOSITS", "IsSaturday", "IsSunday"
]

# Reduced hyperparameter grid for SARIMAX orders (p and q in [0,1])
p_values = [0, 1]
q_values = [0, 1]
param_combinations = list(itertools.product(p_values, q_values))

# Rolling window size (90 days)
rolling_window_size = 90

# Function to determine stationarity (returns True if series is stationary)
def is_stationary(series):
    result = adfuller(series.dropna())
    return result[1] < 0.05

# Storage for rolling window results
results = []

# Process each ATM
for atm_id in data["ATM_ID"].unique():
    atm_data = data[data['ATM_ID'] == atm_id].copy()
    atm_data.set_index('DATE', inplace=True)

    # Filter for F5 columns (target + features) and drop missing values
    columns_required = ["ATM_WITHDRWLS"] + feature_set
    atm_data_filtered = atm_data[columns_required].dropna()

    if len(atm_data_filtered) < rolling_window_size + 1:
        print(f"Skipping ATM {atm_id}: insufficient data for rolling window analysis.")
        continue

    # Rolling window forecast: train on 90 days, forecast next day
    for i in range(rolling_window_size, len(atm_data_filtered)):
        train = atm_data_filtered.iloc[i - rolling_window_size: i]
        test = atm_data_filtered.iloc[i: i + 1]

        # Determine d parameter based on stationarity test on training target
        d_param = 0 if is_stationary(train["ATM_WITHDRWLS"]) else 1

        best_aic = np.inf
        best_order = None
        best_model = None

        # Grid search for best SARIMAX order in current window using reduced grid
        for order in param_combinations:
            candidate_order = (order[0], d_param, order[1])
            try:
                model = SARIMAX(train["ATM_WITHDRWLS"],
                                exog=train[feature_set],
                                order=candidate_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                fitted_model = model.fit(disp=False)
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_order = candidate_order
                    best_model = fitted_model
            except Exception:
                continue

        # If a suitable model was found, predict the next day's withdrawal
        if best_model is not None:
            try:
                y_pred_series = best_model.predict(start=len(train), end=len(train), exog=test[feature_set])
                y_pred = np.array(y_pred_series)
                y_true = test["ATM_WITHDRWLS"].values
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                denominator = abs(y_true[0]) + abs(y_pred[0]) + 1e-6  # Avoid division by near-zero values
                smape = (200 * abs(y_true[0] - y_pred[0]) / denominator)
                results.append({
                    "ATM_ID": atm_id,
                    "Date": test.index[0],
                    "MAE": mae,
                    "RMSE": rmse,
                    "SMAPE": smape,
                    "Best Order": best_order
                })
            except Exception as e:
                print(f"Error for ATM {atm_id} at {test.index[0]}: {e}")
                continue
        else:
            print(f"No suitable model found for ATM {atm_id} at window ending {test.index[0]}")

    # Save results for the current ATM
    atm_results = pd.DataFrame([r for r in results if r["ATM_ID"] == atm_id])
    atm_results.to_csv(os.path.join(output_dir, f"SARIMAX_results_{atm_id}_F5.csv"), index=False)

# Save combined results for all ATMs
combined_results = pd.DataFrame(results)
combined_results.to_csv("SARIMAX_rolling_results_F5.csv", index=False)
print("Rolling window analysis for SARIMAX on F5 completed.")
