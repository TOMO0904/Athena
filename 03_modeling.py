import os
import numpy as np
import lightgbm as lgb
import optuna
import pickle
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def main():
    base_data_dir = "processed_data"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    
    for ds_name in datasets:
        data_dir = os.path.join(base_data_dir, ds_name)
        if not os.path.exists(data_dir):
            continue
            
        print(f"\n======================================")
        print(f"Modeling {ds_name}...")
        print(f"======================================")
        
        # Load data
        X_train_3d = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy")).ravel()
        X_val_3d = np.load(os.path.join(data_dir, "X_val.npy"))
        y_val = np.load(os.path.join(data_dir, "y_val.npy")).ravel()
        X_test_3d = np.load(os.path.join(data_dir, "X_test.npy"))
        y_test = np.load(os.path.join(data_dir, "y_test.npy")).ravel()
        
        # Flatten features
        X_train = X_train_3d.reshape(X_train_3d.shape[0], -1)
        X_val = X_val_3d.reshape(X_val_3d.shape[0], -1)
        X_test = X_test_3d.reshape(X_test_3d.shape[0], -1)
        
        print(f"Flattened X_train shape: {X_train.shape}")
        
        # 1. Baseline 1: Naive (Predict same as the last observed OT)
        # Target index is 6 in the window
        y_pred_naive = X_test_3d[:, -1, 6] 
        
        # 2. Baseline 2: Ridge Regression
        print("Training Ridge Regression...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        
        # 3. LightGBM with Optuna
        print("Running Optuna for LightGBM...")
        
        def objective(trial):
            param = {
                'objective': 'regression',
                'metric': 'mse',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True)
            }
            
            gbm = lgb.LGBMRegressor(**param, n_estimators=100, random_state=42, n_jobs=-1)
            gbm.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
            
            preds = gbm.predict(X_val)
            mse = mean_squared_error(y_val, preds)
            return mse

        # Use 15 trials for fast execution over 4 datasets
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=15)
        
        print(f"Best LightGBM Params for {ds_name}: {study.best_params}")
        
        # Train final LightGBM model
        best_params = study.best_params
        best_params['objective'] = 'regression'
        best_params['metric'] = 'mse'
        best_params['verbosity'] = -1
        
        final_lgbm = lgb.LGBMRegressor(**best_params, n_estimators=500, random_state=42, n_jobs=-1)
        final_lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        y_pred_lgbm = final_lgbm.predict(X_test)
        
        # Save predictions
        np.save(os.path.join(data_dir, "y_pred_naive.npy"), y_pred_naive)
        np.save(os.path.join(data_dir, "y_pred_ridge.npy"), y_pred_ridge)
        np.save(os.path.join(data_dir, "y_pred_lgbm.npy"), y_pred_lgbm)
        
        # Save models
        with open(os.path.join(model_dir, f"{ds_name}_ridge_model.pkl"), "wb") as f:
            pickle.dump(ridge, f)
        with open(os.path.join(model_dir, f"{ds_name}_lgbm_model.pkl"), "wb") as f:
            pickle.dump(final_lgbm, f)
            
    print("\nAll models trained and predictions saved.")

if __name__ == "__main__":
    main()
