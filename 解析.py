import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os

# 設定
DATA_DIR = "./data"
TRAIN_RATIO = 0.8
TARGET = 'OT'

def load_data(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"File not found: {filename}")
        return None
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    print(f"Loaded {filename}: {df.shape}")
    return df

def feature_engineering(df, is_m_data=True):
    df = df.copy()
    steps_per_day = 96 if is_m_data else 24
    
    # --- 1. 負荷のラグ変数 (予防保全：短すぎないラグを使用) ---
    lags = [4, 8, 16, 48, steps_per_day, steps_per_day * 7]
    for lag in lags:
        if lag < len(df):
            df[f'HUFL_lag_{lag}'] = df['HUFL'].shift(lag)
            df[f'LUFL_lag_{lag}'] = df['LUFL'].shift(lag)
            
    # --- 2. 負荷の変化率 (1時間〜4時間のトレンド) ---
    for lag in [4, 16]: 
        df[f'HUFL_diff_{lag}'] = df['HUFL'].diff(lag)
        df[f'LUFL_diff_{lag}'] = df['LUFL'].diff(lag)

    # --- 3. 負荷の移動平均 ---
    windows = [12, 24, steps_per_day]
    for w in windows:
        df[f'HUFL_roll_mean_{w}'] = df['HUFL'].shift(1).rolling(window=w).mean()
        df[f'LUFL_roll_mean_{w}'] = df['LUFL'].shift(1).rolling(window=w).mean()

    # --- 4. 交互作用特徴量 ---
    df['load_interaction'] = df['HUFL'] * df['LUFL']

    # --- 5. 季節性・ベースライン (OTの長期傾向) ---
    df['OT_baseline_24h'] = df['OT'].shift(steps_per_day)

    # --- 6. 周期的な時間情報のエンコーディング (sin/cos) ---
    # 24時間周期
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    # 12ヶ月周期
    df['month_sin'] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df.index.month - 1) / 12)
    
    df = df.dropna()
    return df

def train_and_evaluate_final(df, name="Model"):
    # 特徴量とターゲットの分離 (絶対温度を直接予測)
    features = [c for c in df.columns if c != TARGET]
    X = df[features]
    y = df[TARGET]
    
    split_idx = int(len(df) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # --- Model 1: Ridge ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    
    # --- Model 2: RandomForest ---
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    # --- Stacking (Ensemble) ---
    y_pred_stack = (y_pred_ridge + y_pred_rf) / 2
    
    # 滑らかにする (45分平均)
    y_pred_final_smooth = pd.Series(y_pred_stack).rolling(window=3, min_periods=1).mean().values
    
    r2 = r2_score(y_test, y_pred_final_smooth)
    mae = mean_absolute_error(y_test, y_pred_final_smooth)
    print(f"  {name} Final Test R2: {r2:.4f}, MAE: {mae:.4f}")
    
    return y_test, y_pred_final_smooth

def plot_final_results(y_test, y_pred, title):
    plt.figure(figsize=(20, 6))
    plt.plot(y_test.index, y_test.values, label='Actual OT', alpha=0.4, color='blue')
    plt.plot(y_test.index, y_pred, label='Final Stacked Prediction (Absolute)', alpha=0.8, color='purple', linewidth=1)
    plt.title(title)
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = title.replace(" ", "_") + ".png"
    plt.savefig(filename)
    print(f"  Saved plot: {filename}")

def main():
    datasets = ["ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv"]
    
    for filename in datasets:
        print(f"\n=== Analyzing {filename} (Final Absolute Mode) ===")
        df = load_data(filename)
        if df is None: continue
        
        is_m_data = "ETTm" in filename
        df_features = feature_engineering(df, is_m_data=is_m_data)
        
        y_test, y_pred = train_and_evaluate_final(df_features, filename.replace(".csv", ""))
        plot_final_results(y_test, y_pred, f"{filename.replace('.csv', '')} Final Absolute Prediction")

if __name__ == "__main__":
    main()