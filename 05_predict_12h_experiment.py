import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

def create_sliding_windows(data, seq_len, pred_len, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x_i = data[i : i + seq_len, :]
        # Predict the value exactly 'pred_len' steps ahead
        y_i = data[i + seq_len + pred_len - 1, target_idx]
        X.append(x_i)
        y.append(y_i)
    return np.array(X), np.array(y)

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
        print(f"Running 12-Hour Prediction Experiment for {ds_name}...")
        print(f"{'='*50}")
        
        # 1. Load Data
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['month_sin'] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df.index.month - 1) / 12)
        df['weekday_sin'] = np.sin(2 * np.pi * df.index.weekday / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df.index.weekday / 7)
        
        target_col = 'OT'
        features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT', 
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos']
        
        df = df[features]
        target_idx = features.index(target_col)
        
        # 2. Split Data
        num_train = int(len(df) * 0.7)
        num_test = int(len(df) * 0.2)
        num_val = len(df) - num_train - num_test
        
        train_df = df.iloc[0:num_train]
        val_df = df.iloc[num_train:num_train+num_val]
        test_df = df.iloc[num_train+num_val:]
        
        # 3. Scaling
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_df.values)
        val_data = scaler.transform(val_df.values)
        test_data = scaler.transform(test_df.values)
        
        target_scaler = StandardScaler()
        target_scaler.fit(train_df[[target_col]].values)
        
        # 4. Windows for 12 Hours Ahead
        if "ETTm" in ds_name:
            seq_len = 96     # Past 24 hours (15-min intervals)
            pred_len = 48    # Predict 12 hours ahead
        else:
            seq_len = 24     # Past 24 hours (1-hour intervals)
            pred_len = 12    # Predict 12 hours ahead
            
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
        
        # 5. Modeling
        # 5.1 Naive Model (Predict the same OT as the LAST step in the window)
        # Note: the last step in the window is 12 hours BEFORE the target
        y_pred_naive_s = X_test_3d[:, -1, target_idx]
        
        # 5.2 Ridge Regression
        print("Training Ridge...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge_s = ridge.predict(X_test)
        
        # 5.3 LightGBM (Fast Optuna)
        print("Running fast Optuna for LightGBM (5 trials)...")
        def objective(trial):
            param = {
                'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 80),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0)
            }
            gbm = lgb.LGBMRegressor(**param, n_estimators=50, random_state=42, n_jobs=-1)
            gbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])
            return mean_squared_error(y_val, gbm.predict(X_val))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=5)
        
        best_params = study.best_params
        best_params.update({'objective': 'regression', 'metric': 'mse', 'verbosity': -1})
        
        print("Training final LightGBM...")
        final_lgbm = lgb.LGBMRegressor(**best_params, n_estimators=200, random_state=42, n_jobs=-1)
        final_lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                       callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])
        
        y_pred_lgbm_s = final_lgbm.predict(X_test)
        
        # 6. Evaluation (Inverse Transform)
        y_test_real = target_scaler.inverse_transform(y_test_s.reshape(-1, 1)).ravel()
        y_pred_naive_real = target_scaler.inverse_transform(y_pred_naive_s.reshape(-1, 1)).ravel()
        y_pred_ridge_real = target_scaler.inverse_transform(y_pred_ridge_s.reshape(-1, 1)).ravel()
        y_pred_lgbm_real = target_scaler.inverse_transform(y_pred_lgbm_s.reshape(-1, 1)).ravel()
        
        metrics = {
            "Naive": {"MSE": mean_squared_error(y_test_real, y_pred_naive_real), 
                      "MAE": mean_absolute_error(y_test_real, y_pred_naive_real)},
            "Ridge": {"MSE": mean_squared_error(y_test_real, y_pred_ridge_real), 
                      "MAE": mean_absolute_error(y_test_real, y_pred_ridge_real)},
            "LightGBM": {"MSE": mean_squared_error(y_test_real, y_pred_lgbm_real), 
                         "MAE": mean_absolute_error(y_test_real, y_pred_lgbm_real)}
        }
        
        for m_name, m_vals in metrics.items():
            all_results.append({
                "Dataset": ds_name, "Model": m_name, 
                "MSE": m_vals["MSE"], "MAE": m_vals["MAE"]
            })
            
        # 7. Plotting (1 week zoom)
        subset_idx = 672 if "ETTm" in ds_name else 168
        plt.figure(figsize=(15, 6))
        plt.plot(test_timestamps[:subset_idx], y_test_real[:subset_idx], label="Actual OT", color='black', marker='o', markersize=2)
        plt.plot(test_timestamps[:subset_idx], y_pred_lgbm_real[:subset_idx], label="LightGBM 12h Prediction", color='red', marker='x', markersize=2)
        plt.plot(test_timestamps[:subset_idx], y_pred_naive_real[:subset_idx], label="Naive 12h Prediction", color='blue', alpha=0.5, linestyle='--')
        plt.title(f"{ds_name}: 12-Hour Ahead Prediction Comparison (1-Week Zoom)")
        plt.ylabel("Temperature (°C)")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{ds_name}_12h_Model_Comparison.png"))
        plt.close()

    # 8. Summary Table
    print("\n" + "="*50)
    print("FINAL SUMMARY TABLE (12 Hours Ahead Prediction)")
    print("="*50)
    df_results = pd.DataFrame(all_results)
    print(df_results.to_string(index=False))
    
    df_results.to_csv(os.path.join(output_dir, "12h_results_summary.csv"), index=False)
    print(f"\nResults saved to {output_dir}/12h_results_summary.csv")

if __name__ == "__main__":
    main()
