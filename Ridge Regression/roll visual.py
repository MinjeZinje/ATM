import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Create output directory
output_dir = "monthly_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# Find all ridge_results files
result_files = glob.glob("ridge_results_*_F5.csv")

# Dictionary to store rolling mean results
rolling_means = {}

for file in result_files:
    # Load results
    df_results = pd.read_csv(file)

    # Convert Date column to datetime
    df_results["Date"] = pd.to_datetime(df_results["Date"])
    df_results.sort_values("Date", inplace=True)

    # Compute 30-day rolling means
    df_results["MAE_Rolling"] = df_results["MAE"].rolling(window=90, min_periods=1).mean()
    df_results["SMAPE_Rolling"] = df_results["SMAPE"].rolling(window=90, min_periods=1).mean()

    # Extract ATM ID from file name
    atm_id = file.split("_")[2]
    rolling_means[atm_id] = df_results

# Plot Rolling Mean of MAE and SMAPE Across ATMs
plt.figure(figsize=(14, 10))

# Subplot 1: Rolling Mean of MAE
plt.subplot(2, 1, 1)
for atm_id, df in rolling_means.items():
    plt.plot(df["Date"], df["MAE_Rolling"], label=atm_id)
plt.title("90-Day Rolling Mean of MAE Across ATMs")
plt.ylabel("MAE (Rolling Avg)")
plt.legend()
plt.grid()

# Subplot 2: Rolling Mean of SMAPE
plt.subplot(2, 1, 2)
for atm_id, df in rolling_means.items():
    plt.plot(df["Date"], df["SMAPE_Rolling"], label=atm_id)
plt.title("90-Day Rolling Mean of SMAPE Across ATMs")
plt.ylabel("SMAPE (%)")
plt.xlabel("Date")
plt.legend()
plt.grid()

# Save the combined plot
plt.tight_layout()
plt.savefig(f"{output_dir}/rolling_error_trends.png")
plt.close()

print("90-day rolling mean analysis completed. Plots saved.")
