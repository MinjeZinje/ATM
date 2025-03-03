import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Load dataset
file_path = 'data.csv'  # Adjust the file path accordingly
data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()  # Remove any leading/trailing spaces from column names
print("Columns in dataset:", data.columns.tolist())  # Debugging: Check actual column names

# Ensure DATE column is correctly formatted and set as index
data['DATE'] = pd.to_datetime(data['DATE'], format='%d/%m/%Y', errors='coerce')
data.sort_values(by=['ATM_ID', 'DATE'], inplace=True)

# Define ATMs to process
atms_to_analyze = ['Z0241002', 'Z0951001', 'Z1031001', 'Z1119001', 'Z1899001']

# Define feature sets
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

# Ensure output directory exists
output_dir = "ridge_graphs"
os.makedirs(output_dir, exist_ok=True)

# Store results
results = []

# Train Ridge Regression per ATM
for atm_id in atms_to_analyze:
    print(f"\nProcessing ATM: {atm_id}")
    atm_data = data[data['ATM_ID'] == atm_id].copy()
    if atm_data.empty:
        print(f"No data for {atm_id}. Skipping.")
        continue

    for feature_set_name, feature_set in feature_sets.items():
        print(f"  Feature Set: {feature_set_name}")
        selected_columns = ["DATE", "ATM_WITHDRWLS"] + [col for col in feature_set if col in atm_data.columns]
        atm_data_filtered = atm_data[selected_columns].dropna().copy()

        # ✅ Ensure DATE is still present before setting it as index
        print("Filtered dataset columns before setting index:", atm_data_filtered.columns.tolist())

        atm_data_filtered.set_index('DATE', inplace=True)

        # ✅ Convert split_date to Timestamp
        split_date = pd.Timestamp('2007-06-01')

        # Train-test split
        train = atm_data_filtered[atm_data_filtered.index < split_date]
        test = atm_data_filtered[atm_data_filtered.index >= split_date]

        if test.empty:
            print(f"  Skipping {feature_set_name} due to empty test set.")
            continue

        # Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feature_set])
        X_test = scaler.transform(test[feature_set])
        y_train, y_test = train["ATM_WITHDRWLS"], test["ATM_WITHDRWLS"]

        # Train Ridge Regression with Cross-Validation
        model = RidgeCV(alphas=[0.1, 1, 10, 50, 100, 200, 500])
        model.fit(X_train, y_train)

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Calculate errors
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        # Store results
        results.append({
            'ATM_ID': atm_id,
            'Feature_Set': feature_set_name,
            'Best_Ridge_Alpha': model.alpha_,
            'MAE (Train/Test)': f"{train_mae:.2f} / {test_mae:.2f}",
            'RMSE (Train/Test)': f"{train_rmse:.2f} / {test_rmse:.2f}"
        })

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv('ridge_results.csv', index=False)
print("Processing complete. Results saved to CSV and graphs saved.")
