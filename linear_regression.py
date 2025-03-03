import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression

# Load the dataset with a relative path
file_path = 'data.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}. Ensure it is in the correct directory.")

data = pd.read_csv(file_path)

# Replace empty cells with NaN and handle missing values
data.replace(['', ' '], np.nan, inplace=True)
data.ffill(inplace=True)  # Forward-fill missing values

# Define Feature Sets
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

# Hyperparameter tuning for Ridge
alphas = [0.1, 1, 10, 50, 100, 200, 500]

# Train and Evaluate Ridge Regression with Polynomial Features
results = []
ts_cv = TimeSeriesSplit(n_splits=10)

for feature_set_name, features in feature_sets.items():
    X = data[features]
    y = data['ATM_WITHDRWLS']  # Target Variable

    # Handle missing values
    if X.isnull().values.any():
        X = X.dropna()
        y = y.loc[X.index]

    # Scale the features (use StandardScaler instead of MinMaxScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Generate Polynomial Features (degree=2)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    # Feature Selection (SelectKBest with f_regression)
    selector = SelectKBest(score_func=f_regression, k=min(15, X_poly.shape[1]))
    X_selected = selector.fit_transform(X_poly, y)

    # Tune Ridge using GridSearchCV
    ridge_grid = GridSearchCV(Ridge(), param_grid={'alpha': alphas}, scoring='neg_mean_absolute_error', cv=3)
    ridge_grid.fit(X_selected, y)
    best_ridge = ridge_grid.best_estimator_

    mae_ridge_list = []
    rmse_ridge_list = []

    # Perform Time-Series Cross-Validation
    for train_index, test_index in ts_cv.split(X_selected):
        X_train, X_test = X_selected[train_index], X_selected[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        best_ridge.fit(X_train, y_train)
        ridge_pred = best_ridge.predict(X_test)

        mae_ridge = mean_absolute_error(y_test, ridge_pred)
        rmse_ridge = np.sqrt(mean_squared_error(y_test, ridge_pred))

        mae_ridge_list.append(mae_ridge)
        rmse_ridge_list.append(rmse_ridge)

    # Print feature importance (coefficients)
    print(f"Feature Importance for {feature_set_name} - Ridge:")
    print(dict(zip(poly.get_feature_names_out(), best_ridge.coef_)))

    # Record results
    results.append({
        'Feature Set': feature_set_name,
        'Best Ridge Alpha': best_ridge.alpha,
        'MAE (Ridge)': np.mean(mae_ridge_list),
        'RMSE (Ridge)': np.mean(rmse_ridge_list)
    })

# Display Results
results_df = pd.DataFrame(results)
print(results_df)
