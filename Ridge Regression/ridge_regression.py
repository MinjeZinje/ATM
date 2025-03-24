import pandas as pd
import numpy as np
import os
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = "data.csv"
data = pd.read_csv(file_path)
data['DATE'] = pd.to_datetime(data['DATE'], dayfirst=True, errors='coerce')
data.sort_values(by=['ATM_ID', 'DATE'], inplace=True)

# Define Feature Sets
feature_sets = {
    "F0": ["ATM_WITHDRWLS_1WEEKAGO"],
    "F1": ["ATM_WITHDRWLS_1WEEKAGO", "HOLIDAY", "AVG_ATM_WITHDRW_PREVMONTH", "DAY_OF_WEEK"],
    "F2": ["ATM_WITHDRWLS_1WEEKAGO", "HOLIDAY", "AVG_ATM_WITHDRW_PREVMONTH", "DAY_OF_WEEK", "MONTH", "FIRSTWORKDAY", "LASTWORKDAY"],
    "F3": ["ATM_WITHDRWLS_1WEEKAGO", "ATM_WITHDRWLS_2WEEKAGO", "HOLIDAY", "AVG_ATM_WITHDRW_PREVMONTH", "DAY_OF_WEEK", "IsSaturday", "IsSunday"],
    "F4": ["ATM_WITHDRWLS_1WEEKAGO", "HOLIDAY", "AVG_ATM_WITHDRW_PREVMONTH", "DAY_OF_WEEK", "MONTH", "FIRSTWORKDAY", "LASTWORKDAY", "ATM_WITHDRWLS_2WEEKAGO", "BRA_WITHDRWLS", "BRA_DEPOSITS", "IsSaturday", "IsSunday"],
    "F5": ["ATM_WITHDRWLS_1WEEKAGO", "HOLIDAY", "AVG_ATM_WITHDRW_PREVMONTH", "ATM_WITHDRWLS_2WEEKAGO", "DAY_OF_WEEK", "MONTH", "FIRSTWORKDAY", "LASTWORKDAY", "BRA_WITHDRWL_RATIO", "AVG_BRA_WITHDRWL_PREVMONTH", "BRA_WITHDRWLS", "BRA_DEPOSITS", "IsSaturday", "IsSunday"]
}

# ATMs to analyze
atms_to_analyze = data["ATM_ID"].unique()

# RidgeCV alpha values
tuned_alphas = [0.1, 1, 10, 50, 100, 200, 500]

# Store results and predictions
combined_results = []
prediction_rows = []

# Process each ATM
for atm_id in atms_to_analyze:
    atm_data = data[data['ATM_ID'] == atm_id]
    atm_data.set_index('DATE', inplace=True)

    for feature_set_name, feature_set in feature_sets.items():
        selected_columns = ["ATM_WITHDRWLS"] + [col for col in feature_set if col in atm_data.columns]

        # Filter by fixed date range
        start_date = pd.to_datetime("2006-01-01")
        end_date = pd.to_datetime("2008-02-25")
        atm_data_filtered = atm_data[selected_columns].loc[start_date:end_date].dropna()

        # Apply fixed train/test date split
        train = atm_data_filtered.loc[:pd.to_datetime("2007-05-31")]
        test = atm_data_filtered.loc[pd.to_datetime("2007-06-01"):pd.to_datetime("2008-02-25")]

        if train.empty or test.empty:
            continue

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

        # Save predictions
        train_df = pd.DataFrame({
            'ATM_ID': atm_id,
            'Feature_Set': feature_set_name,
            'DATE': train.index,
            'Set_Type': 'Train',
            'Actual': y_train,
            'Predicted': train_pred
        })

        test_df = pd.DataFrame({
            'ATM_ID': atm_id,
            'Feature_Set': feature_set_name,
            'DATE': test.index,
            'Set_Type': 'Test',
            'Actual': y_test,
            'Predicted': test_pred
        })

        prediction_rows.extend([train_df, test_df])

        # Errors
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mape = (train_mae / np.mean(y_train)) * 100 if np.mean(y_train) != 0 else None
        test_mape = (test_mae / np.mean(y_test)) * 100 if np.mean(y_test) != 0 else None

        combined_results.append({
            'ATM_ID': atm_id,
            'Feature_Set': feature_set_name,
            'Best Ridge Alpha': ridge.alpha_,
            'MAE (Train/Test)': f"{train_mae:.2f} / {test_mae:.2f}",
            'RMSE (Train/Test)': f"{train_rmse:.2f} / {test_rmse:.2f}",
            'MAPE (Train/Test)': f"{train_mape:.2f} / {test_mape:.2f}"
        })

# Save errors
results_df = pd.DataFrame(combined_results)
results_df.to_csv("ridge_results.csv", index=False)

# Save predictions
pd.concat(prediction_rows).to_csv("ridge_predictions.csv", index=False)

print(" Ridge Regression results and predictions saved.")
