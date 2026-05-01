import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

def create_sliding_windows(data, seq_len, pred_len, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i : i + seq_len, :])
        y.append(data[i + seq_len + pred_len - 1, target_idx])
    return np.array(X), np.array(y)

def calculate_risk_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    under_mask = y_pred < y_true
    misses = y_true[under_mask] - y_pred[under_mask]
    
    max_miss = np.max(misses) if len(misses) > 0 else 0.0
    p95_miss = np.percentile(misses, 95) if len(misses) > 0 else 0.0
    
    print(f"  [{model_name}] MSE: {mse:.4f}, MAE: {mae:.4f} ℃")
    print(f"  [{model_name}] Max Miss (Under-prediction): {max_miss:.4f} ℃")
    print(f"  [{model_name}] 95th Percentile Miss       : {p95_miss:.4f} ℃")
    
    return {"Model": model_name, "MSE": mse, "MAE": mae, "Max_Miss": max_miss, "P95_Miss": p95_miss}

def main():
    data_dir = "data"
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]
    all_results = []
    
    for filename in datasets:
        ds_name = filename.replace(".csv", "")
        filepath = os.path.join(data_dir, filename)
        
        print(f"\n{'='*50}")
        print(f"Running 6-Hour Prediction Experiment for {ds_name}...")
        print(f"{'='*50}")
        
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['month_sin'] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df.index.month - 1) / 12)
        
        target_col = 'OT'
        features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT', 
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        
        df = df[features]
        target_idx = features.index(target_col)
        
        num_train = int(len(df) * 0.7)
        num_test = int(len(df) * 0.2)
        num_val = len(df) - num_train - num_test
        
        train_df = df.iloc[0:num_train]
        val_df = df.iloc[num_train:num_train+num_val]
        test_df = df.iloc[num_train+num_val:]
        
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_df.values)
        val_data = scaler.transform(val_df.values)
        test_data = scaler.transform(test_df.values)
        
        target_scaler = StandardScaler()
        target_scaler.fit(train_df[[target_col]].values)
        
        # 6 Hours Ahead
        if "ETTm" in ds_name:
            seq_len = 96     # 24 hours
            pred_len = 24    # 6 hours ahead
        else:
            seq_len = 24     # 24 hours
            pred_len = 6     # 6 hours ahead
            
        print(f"Creating windows (seq_len={seq_len}, pred_len={pred_len})...")
        X_train_3d, y_train_s = create_sliding_windows(train_data, seq_len, pred_len, target_idx)
        X_val_3d, y_val_s = create_sliding_windows(val_data, seq_len, pred_len, target_idx)
        X_test_3d, y_test_s = create_sliding_windows(test_data, seq_len, pred_len, target_idx)
        
        X_train = X_train_3d.reshape(X_train_3d.shape[0], -1)
        X_val = X_val_3d.reshape(X_val_3d.shape[0], -1)
        X_test = X_test_3d.reshape(X_test_3d.shape[0], -1)
        
        y_train = y_train_s.ravel()
        y_val = y_val_s.ravel()
        test_timestamps = test_df.index[seq_len + pred_len - 1:]
        
        # 1. Naive Model
        y_pred_naive_s = X_test_3d[:, -1, target_idx]
        
        # 2. Ridge Regression
        print("Training Ridge...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge_s = ridge.predict(X_test)
        
        # 3. LightGBM (Standard)
        print("Training LightGBM...")
        params = {
            'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
            'n_estimators': 100, 'random_state': 42, 'n_jobs': -1, 'objective': 'regression'
        }
        lgbm = lgb.LGBMRegressor(**params)
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
        y_pred_lgbm_s = lgbm.predict(X_test)
        
        # Evaluation (Inverse Transform)
        y_test_real = target_scaler.inverse_transform(y_test_s.reshape(-1, 1)).ravel()
        y_pred_naive_real = target_scaler.inverse_transform(y_pred_naive_s.reshape(-1, 1)).ravel()
        y_pred_ridge_real = target_scaler.inverse_transform(y_pred_ridge_s.reshape(-1, 1)).ravel()
        y_pred_lgbm_real = target_scaler.inverse_transform(y_pred_lgbm_s.reshape(-1, 1)).ravel()
        
        print("\n--- Results ---")
        res_naive = calculate_risk_metrics(y_test_real, y_pred_naive_real, "Naive")
        res_ridge = calculate_risk_metrics(y_test_real, y_pred_ridge_real, "Ridge")
        res_lgbm = calculate_risk_metrics(y_test_real, y_pred_lgbm_real, "LightGBM")
        
        for res in [res_naive, res_ridge, res_lgbm]:
            res["Dataset"] = ds_name
            all_results.append(res)
            
        # Plotting (only for the highly volatile ETTh2 to visualize 6h vs 12h)
        if ds_name == "ETTh2":
            subset_idx = 168
            plt.figure(figsize=(15, 6))
            plt.plot(test_timestamps[:subset_idx], y_test_real[:subset_idx], label="Actual OT", color='black', marker='o', markersize=3)
            plt.plot(test_timestamps[:subset_idx], y_pred_ridge_real[:subset_idx], label="Ridge 6h Prediction", color='red', linewidth=2)
            plt.title(f"{ds_name}: 6-Hour Ahead Prediction (Ridge)")
            plt.ylabel("Temperature (°C)")
            plt.xlabel("Date")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{ds_name}_6h_Model_Comparison.png"))
            plt.close()

    print("\n" + "="*50)
    print("FINAL SUMMARY TABLE (6 Hours Ahead Prediction)")
    print("="*50)
    df_results = pd.DataFrame(all_results)[["Dataset", "Model", "MSE", "MAE", "Max_Miss", "P95_Miss"]]
    print(df_results.to_string(index=False))
    
    df_results.to_csv(os.path.join(output_dir, "6h_results_summary.csv"), index=False)
    print(f"\nResults saved to {output_dir}/6h_results_summary.csv")

if __name__ == "__main__":
    main()
