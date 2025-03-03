import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load Processed Data
data = pd.read_csv(r'C:\Users\Minjin\PycharmProjects\pythonProject1\.venv\processed_5ATM_features.csv')

# Step 2: Replace Empty Cells with NaN
data.replace(['', ' '], pd.NA, inplace=True)

# Step 3: Drop Rows with Missing Critical Features
data.dropna(subset=['ATM_WITHDRWLS_1WEEKAGO', 'AVG_ATM_WITHDRW_PREVMONTH'], inplace=True)

# Step 4: Add Derived Features (IsSaturday and IsSunday)
data['IsSaturday'] = (data['DAY_OF_WEEK'] == 6).astype(int)
data['IsSunday'] = (data['DAY_OF_WEEK'] == 7).astype(int)

# Step 5: Define Feature Sets
feature_sets = {
    'F0': ['ATM_WITHDRWLS_1WEEKAGO'],
    'F1': ['ATM_WITHDRWLS_1WEEKAGO', 'HOLIDAY', 'AVG_ATM_WITHDRW_PREVMONTH', 'DAY_OF_WEEK'],
    'F2': ['ATM_WITHDRWLS_1WEEKAGO', 'HOLIDAY', 'AVG_ATM_WITHDRW_PREVMONTH',
           'DAY_OF_WEEK', 'MONTH', 'FIRSTWORKDAY', 'LASTWORKDAY'],
    'F3': ['ATM_WITHDRWLS_1WEEKAGO', 'ATM_WITHDRWLS_2WEEKAGO', 'HOLIDAY',
           'AVG_ATM_WITHDRW_PREVMONTH', 'DAY_OF_WEEK', 'IsSaturday', 'IsSunday'],
    'F4': ['ATM_WITHDRWLS_1WEEKAGO', 'HOLIDAY', 'AVG_ATM_WITHDRW_PREVMONTH',
           'DAY_OF_WEEK', 'MONTH', 'FIRSTWORKDAY', 'LASTWORKDAY', 'ATM_WITHDRWLS_2WEEKAGO',
           'BRA_WITHDRWLS', 'BRA_DEPOSITS', 'IsSaturday', 'IsSunday'],
    'F5': ['ATM_WITHDRWLS_1WEEKAGO', 'HOLIDAY', 'AVG_ATM_WITHDRW_PREVMONTH',
           'ATM_WITHDRWLS_2WEEKAGO', 'DAY_OF_WEEK', 'MONTH', 'FIRSTWORKDAY', 'LASTWORKDAY',
           'BRA_WITHDRWL_RATIO', 'AVG_BRA_WITHDRWL_PREVMONTH', 'BRA_WITHDRWLS', 'BRA_DEPOSITS',
           'IsSaturday', 'IsSunday']
}

# Step 6: Train and Evaluate Linear Regression
results = []
tscv = TimeSeriesSplit(n_splits=10)

for feature_set_name, features in feature_sets.items():
    # Filter features from data
    X = data[features]
    y = data['ATM_WITHDRWLS']  # Predict ATM withdrawals

    # Check for and handle missing values in X
    if X.isnull().values.any():
        X = X.dropna()
        y = y[X.index]

    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = Lasso(alpha=80)  # Using Lasso Regression
    mae_list = []
    rmse_list = []

    # Perform Time-Series Cross-Validation
    for train_index, test_index in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        mae_list.append(mae)
        rmse_list.append(rmse)

    # Record results for this feature set
    results.append({
        'Feature Set': feature_set_name,
        'MAE': np.mean(mae_list),
        'RMSE': np.mean(rmse_list)
    })

# Step 7: Display Results
results_df = pd.DataFrame(results)
print(results_df)
