import pandas as pd
import numpy as np
import os
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = "data.csv"  # Ensure correct path
data = pd.read_csv(file_path)
data['DATE'] = pd.to_datetime(data['DATE'], dayfirst=True, errors='coerce')
data.sort_values(by=['ATM_ID', 'DATE'], inplace=True)

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

# RidgeCV alpha values
tuned_alphas = [0.1, 1, 10, 50, 100, 200, 500]

# Store results
combined_results = []

# Process each ATM
for atm_id in atms_to_analyze:
    atm_data = data[data['ATM_ID'] == atm_id]
    atm_data.set_index('DATE', inplace=True)

    for feature_set_name, feature_set in feature_sets.items():
        selected_columns = ["ATM_WITHDRWLS"] + [col for col in feature_set if col in atm_data.columns]
        atm_data_filtered = atm_data[selected_columns].dropna()

        # Split dynamically (80% train, 20% test)
        split_index = int(len(atm_data_filtered) * 0.8)
        train = atm_data_filtered.iloc[:split_index]
        test = atm_data_filtered.iloc[split_index:]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feature_set])
        X_test = scaler.transform(test[feature_set])
        y_train = train["ATM_WITHDRWLS"].values
        y_test = test["ATM_WITHDRWLS"].values

        # Ridge Regression
        ridge = RidgeCV(alphas=tuned_alphas, store_cv_results=True)
        ridge.fit(X_train, y_train)

        # Predictions
        train_pred = ridge.predict(X_train)
        test_pred = ridge.predict(X_test)

        # Calculate errors
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mape = (train_mae / np.mean(y_train)) * 100 if np.mean(y_train) != 0 else None
        test_mape = (test_mae / np.mean(y_test)) * 100 if np.mean(y_test) != 0 else None

        # Store results
        combined_results.append({
            'ATM_ID': atm_id,
            'Feature_Set': feature_set_name,
            'Best Ridge Alpha': ridge.alpha_,
            'MAE (Train/Test)': f"{train_mae:.2f} / {test_mae:.2f}",
            'RMSE (Train/Test)': f"{train_rmse:.2f} / {test_rmse:.2f}",
            'MAPE (Train/Test)': f"{train_mape:.2f} / {test_mape:.2f}"
        })

# Save results
results_df = pd.DataFrame(combined_results)
results_df.to_csv("ridge_results.csv", index=False)
print("Updated Ridge Regression results saved with Train/Test MAPE.")
