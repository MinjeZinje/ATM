import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
file_path = "data.csv"  # Ensure correct path
data = pd.read_csv(file_path)
data['DATE'] = pd.to_datetime(data['DATE'], format="%d/%m/%Y", errors='coerce')
data.sort_values(by=['ATM_ID', 'DATE'], inplace=True)

# Fill missing branch data with 0
data[['BRA_WITHDRWLS', 'BRA_DEPOSITS', 'BRA_WITHDRWL_RATIO']] = data[
    ['BRA_WITHDRWLS', 'BRA_DEPOSITS', 'BRA_WITHDRWL_RATIO']].fillna(0)

# Define Feature Set for Rolling Window Analysis (F5)
feature_set = ["ATM_WITHDRWLS_1WEEKAGO", "HOLIDAY", "AVG_ATM_WITHDRW_PREVMONTH", "ATM_WITHDRWLS_2WEEKAGO",
               "DAY_OF_WEEK", "MONTH", "FIRSTWORKDAY", "LASTWORKDAY", "BRA_WITHDRWL_RATIO",
               "AVG_BRA_WITHDRWL_PREVMONTH", "BRA_WITHDRWLS", "BRA_DEPOSITS", "IsSaturday", "IsSunday"]

# RidgeCV alpha values
tuned_alphas = [0.1, 1, 10, 50, 100, 200, 500]

# Rolling Window Size
rolling_window_size = 90

# Define SMAPE function
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid division by zero by setting zero denominators to 1
    denominator = np.where(denominator == 0, 1, denominator)
    return np.mean(np.abs(y_pred - y_true) / denominator) * 100

# Process each ATM with Rolling Window Forecasting
for atm_id in data["ATM_ID"].unique():
    atm_data = data[data['ATM_ID'] == atm_id].set_index('DATE')
    selected_columns = ["ATM_WITHDRWLS"] + [col for col in feature_set if col in atm_data.columns]
    atm_data_filtered = atm_data[selected_columns].dropna()

    if len(atm_data_filtered) < rolling_window_size + 1:
        print(f"Skipping ATM {atm_id}: Not enough data for rolling window analysis.")
        continue

    errors = []
    coef_values = []

    for i in range(rolling_window_size, len(atm_data_filtered)):
        train = atm_data_filtered.iloc[i - rolling_window_size:i]
        test = atm_data_filtered.iloc[i:i + 1]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feature_set])
        X_test = scaler.transform(test[feature_set])
        y_train = train["ATM_WITHDRWLS"].values
        y_test = test["ATM_WITHDRWLS"].values

        ridge = RidgeCV(alphas=tuned_alphas)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        smape_value = smape(y_test, y_pred)
        residual = y_test[0] - y_pred[0]
        errors.append((test.index[0], mae, rmse, smape_value, residual))
        coef_values.append(ridge.coef_)

    results_df = pd.DataFrame(errors, columns=["Date", "MAE", "RMSE", "SMAPE", "Residual"])
    results_df.to_csv(f"ridge_results_{atm_id}_F5.csv", index=False)
    coef_df = pd.DataFrame(coef_values, columns=feature_set, index=results_df["Date"])
    coef_df.to_csv(f"ridge_coefficients_{atm_id}_F5.csv")
    print(f"Processed ATM {atm_id} with rolling window for F5.")

print("Rolling window stability analysis for F5 completed.")
