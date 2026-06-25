import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

def extract_tfidf_features(X_train, X_test, text_col='description', max_features=100):
    if text_col not in X_train.columns:
        return X_train, X_test
    
    train_text = X_train[text_col].fillna('')
    test_text = X_test[text_col].fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    train_tfidf_sparse = tfidf.fit_transform(train_text)
    test_tfidf_sparse = tfidf.transform(test_text)
    
    feature_names = tfidf.get_feature_names_out()
    tfidf_cols = [f"tfidf_{word}" for word in feature_names]
    
    train_tfidf_df = pd.DataFrame(train_tfidf_sparse.toarray(), columns=tfidf_cols, index=X_train.index)
    test_tfidf_df = pd.DataFrame(test_tfidf_sparse.toarray(), columns=tfidf_cols, index=X_test.index)
    
    X_train_enhanced = pd.concat([X_train.drop(columns=[text_col]), train_tfidf_df], axis=1)
    X_test_enhanced = pd.concat([X_test.drop(columns=[text_col]), test_tfidf_df], axis=1)
    return X_train_enhanced, X_test_enhanced

def tune_and_train_model(df, model_name, n_trials=30):
    print(f"\n--- Tuning & Training {model_name} (N={len(df)}) ---")
    X = df.drop(columns=['price'])
    y = df['price']
    y_log = np.log1p(y)
    
    # 1. Final Train/Test split
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
    y_test_actual = np.expm1(y_test_log)
    
    # NLP Features
    X_train, X_test = extract_tfidf_features(X_train, X_test, text_col='description', max_features=100)
    
    # 2. Objective Function for Optuna
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),
            'random_state': 42
        }
        
        # Build monotonic constraints array
        cst = []
        for col in X_train.columns:
            if col in ['has_ac', 'has_pool', 'has_gym', 'has_elevator', 'has_parking', 'has_wifi', 'has_tv', 'has_jacuzzi', 'has_balcony', 'has_kitchen', 'has_washer']:
                cst.append(1)
            elif col in ['accommodates', 'bedrooms', 'beds', 'bathrooms_num']:
                cst.append(1)
            elif col in ['distance_to_center_km', 'distance_to_bosphorus_km']:
                cst.append(-1)
            else:
                cst.append(0)
        
        # Fast internal train/val split inside X_train to avoid KFold overhead
        X_t, X_v, y_t, y_v = train_test_split(X_train, y_train_log, test_size=0.2, random_state=42)
        
        model = HistGradientBoostingRegressor(**params, monotonic_cst=cst)
        model.fit(X_t, y_t)
        
        preds_log = model.predict(X_v)
        rmse = root_mean_squared_error(y_v, preds_log)
        return rmse

    print(f"Running Optuna Optimization for {n_trials} trials...")
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Keep logs clean
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials) # Progress bar removed to avoid stdout clutter
    
    best_params = study.best_params
    print(f"Best Params found: {best_params}")
    
    # 3. Train final model on FULL training set with best params
    print("Retraining on full training set with best hyperparameters...")
    
    # Build monotonic constraints array
    cst = []
    for col in X_train.columns:
        if col in ['has_ac', 'has_pool', 'has_gym', 'has_elevator', 'has_parking', 'has_wifi', 'has_tv', 'has_jacuzzi', 'has_balcony', 'has_kitchen', 'has_washer']:
            cst.append(1)
        elif col in ['accommodates', 'bedrooms', 'beds', 'bathrooms_num']:
            cst.append(1)
        elif col in ['distance_to_center_km', 'distance_to_bosphorus_km']:
            cst.append(-1)
        else:
            cst.append(0)
            
    best_model = HistGradientBoostingRegressor(**best_params, random_state=42, monotonic_cst=cst)
    best_model.fit(X_train, y_train_log)
    
    # 4. Predict on the totally unseen Test Set
    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    
    # 5. Evaluate
    mae = mean_absolute_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)
    
    print(f"  Final Tuned R2  : {r2:.4f}")
    print(f"  Final Tuned MAE : {mae:.2f} TRY")
    
    joblib.dump(best_model, f'{model_name}_tuned.pkl')
    return y_test_actual, y_pred

def run_optuna_pipeline(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    room_type_col = 'room_type_Entire home/apt'
    if room_type_col not in df.columns:
        print(f"Error: {room_type_col} not found!")
        return

    # --- 1. SMART CURRENCY CONVERSION ---
    usd_to_try_rate = 41.4777
    
    mask_entire = (df[room_type_col] == 1) & (df['price'] < 600)
    df.loc[mask_entire, 'price'] = df.loc[mask_entire, 'price'] * usd_to_try_rate
    
    mask_rooms = (df[room_type_col] == 0) & (df['price'] < 100)
    df.loc[mask_rooms, 'price'] = df.loc[mask_rooms, 'price'] * usd_to_try_rate

    # --- 2. OUTLIER REMOVAL ---
    print("\nSegmenting the dataset and removing outliers locally...")
    df_entire = df[df[room_type_col] == 1].copy()
    df_rooms = df[df[room_type_col] == 0].copy()
    
    u_entire = df_entire['price'].quantile(0.99)
    l_entire = df_entire['price'].quantile(0.01)
    df_entire = df_entire[(df_entire['price'] >= l_entire) & (df_entire['price'] <= u_entire)]
    
    u_rooms = df_rooms['price'].quantile(0.99)
    l_rooms = df_rooms['price'].quantile(0.01)
    df_rooms = df_rooms[(df_rooms['price'] >= l_rooms) & (df_rooms['price'] <= u_rooms)]
    
    # --- 3. TUNE AND TRAIN ---
    # Using 30 trials per model to keep execution time reasonable (~1-2 mins total)
    y_actual_A, y_pred_A = tune_and_train_model(df_entire, 'model_entire_home', n_trials=30)
    y_actual_B, y_pred_B = tune_and_train_model(df_rooms, 'model_rooms', n_trials=30)
    
    # --- 4. COMBINED EVALUATION ---
    print("\n" + "="*40)
    print("COMBINED TUNED MODEL RESULTS")
    print("="*40)
    y_actual_combined = np.concatenate([y_actual_A, y_actual_B])
    y_pred_combined = np.concatenate([y_pred_A, y_pred_B])
    
    combined_mae = mean_absolute_error(y_actual_combined, y_pred_combined)
    combined_r2 = r2_score(y_actual_combined, y_pred_combined)
    
    print(f"Combined Tuned R-Squared (R2)  : {combined_r2:.4f}")
    print(f"Combined Tuned Mean Abs Error  : {combined_mae:.2f} TRY")
    print("="*40)

if __name__ == "__main__":
    run_optuna_pipeline('listings_with_location.csv')
