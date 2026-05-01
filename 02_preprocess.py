import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def create_sliding_windows(data, seq_len, pred_len, target_idx):
    """
    data: np.array of shape (num_samples, num_features)
    seq_len: int, length of the input window
    pred_len: int, number of steps ahead to predict (1 hour ahead)
    target_idx: int, index of the target feature
    """
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x_i = data[i : i + seq_len, :]
        # Predict the value exactly 'pred_len' steps ahead (1 hour ahead)
        y_i = data[i + seq_len + pred_len - 1, target_idx]
        X.append(x_i)
        y.append(y_i)
    return np.array(X), np.array(y)

def main():
    data_dir = "data"
    base_output_dir = "processed_data"
    os.makedirs(base_output_dir, exist_ok=True)
    
    datasets = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]
    
    for filename in datasets:
        ds_name = filename.replace(".csv", "")
        output_dir = os.path.join(base_output_dir, ds_name)
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(data_dir, filename)
        print(f"\n======================================")
        print(f"Preprocessing {ds_name}...")
        print(f"======================================")
        
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        
        # 1. Feature Engineering (Time features)
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
        
        # 2. Train / Val / Test Split (70%, 10%, 20%)
        num_train = int(len(df) * 0.7)
        num_test = int(len(df) * 0.2)
        num_val = len(df) - num_train - num_test
        
        train_df = df.iloc[0:num_train]
        val_df = df.iloc[num_train:num_train+num_val]
        test_df = df.iloc[num_train+num_val:]
        
        print(f"Train samples: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # 3. Scaling (fit only on train)
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_df.values)
        val_data = scaler.transform(val_df.values)
        test_data = scaler.transform(test_df.values)
        
        # 4. Sliding Windows Configuration
        if "ETTm" in ds_name:
            # 15-minute intervals
            seq_len = 96  # Past 24 hours (4 * 24)
            pred_len = 4  # Predict exactly 1 hour ahead (4 steps)
        else:
            # 1-hour intervals
            seq_len = 24  # Past 24 hours
            pred_len = 1  # Predict exactly 1 hour ahead (1 step)
            
        print(f"Creating sliding windows (seq_len={seq_len}, pred_len={pred_len})...")
        X_train, y_train = create_sliding_windows(train_data, seq_len, pred_len, target_idx)
        X_val, y_val = create_sliding_windows(val_data, seq_len, pred_len, target_idx)
        X_test, y_test = create_sliding_windows(test_data, seq_len, pred_len, target_idx)
        
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        target_scaler = StandardScaler()
        target_scaler.fit(train_df[[target_col]].values)
        
        # Save objects
        np.save(os.path.join(output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(output_dir, "X_val.npy"), X_val)
        np.save(os.path.join(output_dir, "y_val.npy"), y_val)
        np.save(os.path.join(output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(output_dir, "y_test.npy"), y_test)
        
        test_timestamps = test_df.index[seq_len + pred_len - 1:]
        with open(os.path.join(output_dir, "test_timestamps.pkl"), "wb") as f:
            pickle.dump(test_timestamps, f)
            
        with open(os.path.join(output_dir, "target_scaler.pkl"), "wb") as f:
            pickle.dump(target_scaler, f)
            
        print(f"Preprocessing completed for {ds_name}.")

    print("\nAll datasets processed and saved.")

if __name__ == "__main__":
    main()
