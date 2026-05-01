import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# --- Custom Objective and Metric ---
ALPHA = 5.0  # Penalty multiplier for under-prediction

def asymmetric_objective(y_true, y_pred):
    """
    Custom objective function for LightGBM.
    Penalizes under-predictions (y_pred < y_true) by ALPHA times.
    residual = y_pred - y_true
    If residual < 0, we want a stronger gradient to push y_pred up.
    """
    residual = y_pred - y_true
    grad = np.where(residual < 0, ALPHA * residual, residual)
    hess = np.where(residual < 0, ALPHA, 1.0)
    return grad, hess

def asymmetric_eval(y_true, y_pred):
    """
    Custom evaluation metric for Early Stopping.
    """
    residual = y_pred - y_true
    loss = np.where(residual < 0, ALPHA * (residual**2), residual**2)
    return 'asym_mse', np.mean(loss), False

def create_sliding_windows(data, seq_len, pred_len, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x_i = data[i : i + seq_len, :]
        y_i = data[i + seq_len + pred_len - 1, target_idx]
        X.append(x_i)
        y.append(y_i)
    return np.array(X), np.array(y)

def evaluate_business_risk(y_true, y_pred, model_name):
    # Under-prediction mask (actual is hotter than predicted)
    under_mask = y_pred < y_true
    
    under_rate = np.mean(under_mask) * 100
    if np.sum(under_mask) > 0:
        under_mae = mean_absolute_error(y_true[under_mask], y_pred[under_mask])
    else:
        under_mae = 0.0
        
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"\n--- {model_name} ---")
    print(f"Overall MSE: {mse:.4f}")
    print(f"Overall MAE: {mae:.4f} °C")
    print(f"Under-prediction Rate: {under_rate:.1f}% (Danger cases)")
    print(f"Under-prediction MAE : {under_mae:.4f} °C (Average miss when under-predicting)")
    return mse, mae, under_rate, under_mae

def main():
    data_dir = "data"
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = "ETTh2.csv"
    filepath = os.path.join(data_dir, filename)
    
    print(f"Loading {filename} for Asymmetric Loss Experiment...")
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
    
    seq_len = 24
    pred_len = 12
    
    X_train_3d, y_train_s = create_sliding_windows(train_data, seq_len, pred_len, target_idx)
    X_val_3d, y_val_s = create_sliding_windows(val_data, seq_len, pred_len, target_idx)
    X_test_3d, y_test_s = create_sliding_windows(test_data, seq_len, pred_len, target_idx)
    
    X_train = X_train_3d.reshape(X_train_3d.shape[0], -1)
    X_val = X_val_3d.reshape(X_val_3d.shape[0], -1)
    X_test = X_test_3d.reshape(X_test_3d.shape[0], -1)
    
    y_train = y_train_s.ravel()
    y_val = y_val_s.ravel()
    test_timestamps = test_df.index[seq_len + pred_len - 1:]
    
    common_params = {
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'n_estimators': 200,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 1. Standard LightGBM
    print("\nTraining Standard LightGBM (MSE)...")
    std_lgbm = lgb.LGBMRegressor(**common_params, objective='regression')
    std_lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                 callbacks=[lgb.early_stopping(10, verbose=False)])
    y_pred_std_s = std_lgbm.predict(X_test)
    
    # 2. Safe (Asymmetric) LightGBM
    print("\nTraining Safe LightGBM (Asymmetric Penalty x5)...")
    # By passing the custom objective and custom eval
    safe_lgbm = lgb.LGBMRegressor(**common_params)
    safe_lgbm.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)],
                  eval_metric=asymmetric_eval,
                  callbacks=[lgb.early_stopping(10, verbose=False)])
    
    # Actually, LightGBM scikit-learn API accepts custom objective in `set_params`
    safe_lgbm.set_params(objective=asymmetric_objective)
    safe_lgbm.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)],
                  eval_metric=asymmetric_eval,
                  callbacks=[lgb.early_stopping(10, verbose=False)])
                  
    y_pred_safe_s = safe_lgbm.predict(X_test)
    
    # Inverse Transform
    y_test_real = target_scaler.inverse_transform(y_test_s.reshape(-1, 1)).ravel()
    y_pred_std_real = target_scaler.inverse_transform(y_pred_std_s.reshape(-1, 1)).ravel()
    y_pred_safe_real = target_scaler.inverse_transform(y_pred_safe_s.reshape(-1, 1)).ravel()
    
    print("\n" + "="*50)
    print("BUSINESS RISK EVALUATION (12h Ahead, ETTh2)")
    print("="*50)
    evaluate_business_risk(y_test_real, y_pred_std_real, "Standard AI")
    evaluate_business_risk(y_test_real, y_pred_safe_real, "Safe AI (Asymmetric Loss)")
    
    # Plotting
    subset_idx = 168 # 1 week
    plt.figure(figsize=(15, 6))
    plt.plot(test_timestamps[:subset_idx], y_test_real[:subset_idx], label="Actual OT", color='black', marker='o', markersize=3)
    plt.plot(test_timestamps[:subset_idx], y_pred_std_real[:subset_idx], label="Standard AI", color='blue', alpha=0.6, linestyle='--')
    plt.plot(test_timestamps[:subset_idx], y_pred_safe_real[:subset_idx], label="Safe AI (Asymmetric)", color='red', linewidth=2)
    plt.title("ETTh2: Standard AI vs Safe AI (12h Prediction, Penalty x5)")
    plt.ylabel("Temperature (°C)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ETTh2_Safe_AI_Comparison.png"))
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/ETTh2_Safe_AI_Comparison.png")

if __name__ == "__main__":
    main()
