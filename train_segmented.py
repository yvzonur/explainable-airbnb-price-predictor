import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def extract_tfidf_features(X_train, X_test, text_col='description', max_features=100):
    """
    Fits TF-IDF on X_train to prevent data leakage, and transforms both X_train and X_test.
    Returns X_train and X_test with dense TF-IDF features appended, and the text_col removed.
    """
    if text_col not in X_train.columns:
        return X_train, X_test
        
    print(f"Applying TF-IDF on '{text_col}' (preventing Data Leakage)...")
    
    # Fill NAs
    train_text = X_train[text_col].fillna('')
    test_text = X_test[text_col].fillna('')
    
    # Initialize and fit ONLY on training data
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    
    # Fit & Transform train, only Transform test
    train_tfidf_sparse = tfidf.fit_transform(train_text)
    test_tfidf_sparse = tfidf.transform(test_text)
    
    # Get feature names
    feature_names = tfidf.get_feature_names_out()
    tfidf_cols = [f"tfidf_{word}" for word in feature_names]
    
    # Convert sparse matrices to DataFrames
    train_tfidf_df = pd.DataFrame(train_tfidf_sparse.toarray(), columns=tfidf_cols, index=X_train.index)
    test_tfidf_df = pd.DataFrame(test_tfidf_sparse.toarray(), columns=tfidf_cols, index=X_test.index)
    
    # Concatenate back to original features and drop the raw text column
    X_train_enhanced = pd.concat([X_train.drop(columns=[text_col]), train_tfidf_df], axis=1)
    X_test_enhanced = pd.concat([X_test.drop(columns=[text_col]), test_tfidf_df], axis=1)
    
    return X_train_enhanced, X_test_enhanced

def train_and_eval_model(df, model_name):
    print(f"\n--- Training {model_name} (N={len(df)}) ---")
    X = df.drop(columns=['price'])
    y = df['price']
    
    y_log = np.log1p(y)
    
    # 1. Split Data FIRST (Crucial for preventing data leakage)
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    # 2. Apply TF-IDF safely (fitted only on train)
    X_train, X_test = extract_tfidf_features(X_train, X_test, text_col='description', max_features=100)
    
    y_test_actual = np.expm1(y_test_log)
    
    # 3. Train Model
    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train_log)
    
    # 4. Predict
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    
    # 5. Evaluate
    mae = mean_absolute_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)
    
    print(f"  R2  : {r2:.4f}")
    print(f"  MAE : {mae:.2f} TRY")
    
    joblib.dump(model, f'{model_name}.pkl')
    
    return y_test_actual, y_pred

def train_segmented_models(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Make sure we have the column
    room_type_col = 'room_type_Entire home/apt'
    if room_type_col not in df.columns:
        print(f"Error: {room_type_col} not found!")
        return

    # --- 1. SMART CURRENCY CONVERSION (HEURISTIC FIX) ---
    usd_to_try_rate = 41.4777
    
    # Rule 1: Entire homes under 600 are definitely USD
    mask_entire = (df[room_type_col] == 1) & (df['price'] < 600)
    num_entire = mask_entire.sum()
    df.loc[mask_entire, 'price'] = df.loc[mask_entire, 'price'] * usd_to_try_rate
    
    # Rule 2: Rooms under 100 are probably USD (e.g. $10, $20 rooms). 
    # But a 400 TRY room ($10) might actually be TRY. So we lower the threshold for rooms!
    mask_rooms = (df[room_type_col] == 0) & (df['price'] < 100)
    num_rooms = mask_rooms.sum()
    df.loc[mask_rooms, 'price'] = df.loc[mask_rooms, 'price'] * usd_to_try_rate
    
    print(f"Smart Heuristic: Converted {num_entire} Entire Homes (<600) and {num_rooms} Rooms (<100) from USD to TRY.")

    # --- 2. OUTLIER REMOVAL (Per Segment) ---
    print("\nSegmenting the dataset and removing outliers locally...")
    df_entire = df[df[room_type_col] == 1].copy()
    df_rooms = df[df[room_type_col] == 0].copy()
    
    # It's better to remove outliers PER SEGMENT because rooms and homes have vastly different price distributions!
    u_entire = df_entire['price'].quantile(0.99)
    l_entire = df_entire['price'].quantile(0.01)
    df_entire = df_entire[(df_entire['price'] >= l_entire) & (df_entire['price'] <= u_entire)]
    
    u_rooms = df_rooms['price'].quantile(0.99)
    l_rooms = df_rooms['price'].quantile(0.01)
    df_rooms = df_rooms[(df_rooms['price'] >= l_rooms) & (df_rooms['price'] <= u_rooms)]
    
    print(f"Entire Homes size after outlier removal: {len(df_entire)}")
    print(f"Rooms size after outlier removal: {len(df_rooms)}")
    
    # --- 3. TRAIN EXPERT MODELS ---
    y_test_actual_A, y_pred_A = train_and_eval_model(df_entire, 'model_entire_home')
    y_test_actual_B, y_pred_B = train_and_eval_model(df_rooms, 'model_rooms')
    
    # --- 4. COMBINED EVALUATION ---
    print("\n" + "="*40)
    print("COMBINED SEGMENTED MODEL RESULTS")
    print("="*40)
    
    y_test_actual_combined = np.concatenate([y_test_actual_A, y_test_actual_B])
    y_pred_combined = np.concatenate([y_pred_A, y_pred_B])
    
    combined_mae = mean_absolute_error(y_test_actual_combined, y_pred_combined)
    combined_r2 = r2_score(y_test_actual_combined, y_pred_combined)
    
    print(f"Combined R-Squared (R2)  : {combined_r2:.4f}")
    print(f"Combined Mean Abs Error  : {combined_mae:.2f} TRY")
    print("="*40)

if __name__ == "__main__":
    train_segmented_models('listings_with_location.csv')
