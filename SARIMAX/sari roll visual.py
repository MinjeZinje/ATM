import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Create output directory
output_dir = "monthly_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# Find all SARIMAX_results files
result_files = glob.glob("SARIMAX_results_*_F5.csv")
rolling_means = {}

for file in result_files:
    df_results = pd.read_csv(file)
    df_results["Date"] = pd.to_datetime(df_results["Date"])
    df_results.sort_values("Date", inplace=True)
    df_results["MAE_Rolling"] = df_results["MAE"].rolling(window=90, min_periods=1).mean()
    df_results["SMAPE_Rolling"] = df_results["SMAPE"].rolling(window=90, min_periods=1).mean()
    atm_id = file.split("_")[2]
    rolling_means[atm_id] = df_results

plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)
for atm_id, df in rolling_means.items():
    plt.plot(df["Date"], df["MAE_Rolling"], label=atm_id)
plt.title("90-Day Rolling Mean of MAE Across ATMs")
plt.ylabel("MAE (Rolling Avg)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
for atm_id, df in rolling_means.items():
    plt.plot(df["Date"], df["SMAPE_Rolling"], label=atm_id)
plt.title("90-Day Rolling Mean of SMAPE Across ATMs")
plt.ylabel("SMAPE (%)")
plt.xlabel("Date")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rolling_error_trends.png"))
plt.close()
print("90-day rolling mean analysis completed. Plots saved.")
