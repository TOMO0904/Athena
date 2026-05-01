import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    base_data_dir = "processed_data"
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    
    all_results = []
    
    for ds_name in datasets:
        data_dir = os.path.join(base_data_dir, ds_name)
        if not os.path.exists(data_dir):
            continue
            
        print(f"\nEvaluating {ds_name}...")
        
        # Load data
        y_test_scaled = np.load(os.path.join(data_dir, "y_test.npy")).ravel()
        y_pred_naive_scaled = np.load(os.path.join(data_dir, "y_pred_naive.npy")).ravel()
        y_pred_ridge_scaled = np.load(os.path.join(data_dir, "y_pred_ridge.npy")).ravel()
        y_pred_lgbm_scaled = np.load(os.path.join(data_dir, "y_pred_lgbm.npy")).ravel()
        
        with open(os.path.join(data_dir, "test_timestamps.pkl"), "rb") as f:
            timestamps = pickle.load(f)
            
        with open(os.path.join(data_dir, "target_scaler.pkl"), "rb") as f:
            target_scaler = pickle.load(f)
            
        # Inverse Transform
        y_test = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
        y_pred_naive = target_scaler.inverse_transform(y_pred_naive_scaled.reshape(-1, 1)).ravel()
        y_pred_ridge = target_scaler.inverse_transform(y_pred_ridge_scaled.reshape(-1, 1)).ravel()
        y_pred_lgbm = target_scaler.inverse_transform(y_pred_lgbm_scaled.reshape(-1, 1)).ravel()
        
        # Calculate Metrics
        metrics = {
            "Naive": {
                "MSE": mean_squared_error(y_test, y_pred_naive),
                "MAE": mean_absolute_error(y_test, y_pred_naive)
            },
            "Ridge": {
                "MSE": mean_squared_error(y_test, y_pred_ridge),
                "MAE": mean_absolute_error(y_test, y_pred_ridge)
            },
            "LightGBM": {
                "MSE": mean_squared_error(y_test, y_pred_lgbm),
                "MAE": mean_absolute_error(y_test, y_pred_lgbm)
            }
        }
        
        for model_name, m in metrics.items():
            all_results.append({
                "Dataset": ds_name,
                "Model": model_name,
                "MSE": m["MSE"],
                "MAE": m["MAE"]
            })
            
        # Plotting: 1-Week Subset (Zoom in for business insight)
        # 1 week = 168 hours. 
        # For ETTh: 168 points. For ETTm: 168 * 4 = 672 points.
        subset_idx = 672 if "ETTm" in ds_name else 168
        
        plt.figure(figsize=(15, 6))
        plt.plot(timestamps[:subset_idx], y_test[:subset_idx], label="Actual OT", color='black', marker='o', markersize=2)
        plt.plot(timestamps[:subset_idx], y_pred_lgbm[:subset_idx], label="LightGBM Prediction", color='red', marker='x', markersize=2)
        plt.plot(timestamps[:subset_idx], y_pred_naive[:subset_idx], label="Naive Prediction", color='blue', alpha=0.5, linestyle='--')
        plt.title(f"{ds_name}: Model Comparison (1-Week Zoom)")
        plt.ylabel("Temperature (°C)")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{ds_name}_Model_Comparison_1Week.png"))
        plt.close()
        
    print("\n" + "="*50)
    print("FINAL SUMMARY TABLE (1 Hour Ahead Prediction)")
    print("="*50)
    df_results = pd.DataFrame(all_results)
    print(df_results.to_string(index=False))
    
    # Save as CSV for reporting
    df_results.to_csv(os.path.join(output_dir, "all_results_summary.csv"), index=False)
    print("\nSummary saved to evaluation_results/all_results_summary.csv")

if __name__ == "__main__":
    main()
