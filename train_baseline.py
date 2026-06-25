import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib

def train_baseline_model(data_path):
    """
    Trains a baseline LightGBM model to predict Airbnb prices.
    Applies log transformation to the target variable to handle skewness.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Original dataset size: {len(df)}")
    
    # --- 1. CURRENCY CONVERSION (HEURISTIC) ---
    usd_to_try_rate = 41.4777
    threshold = 600
    
    # Count how many we are going to convert
    num_converted = len(df[df['price'] < threshold])
    print(f"Currency heuristic: Converting {num_converted} listings from USD to TRY (price < {threshold})")
    
    # Apply the conversion
    df.loc[df['price'] < threshold, 'price'] = df.loc[df['price'] < threshold, 'price'] * usd_to_try_rate
    # ------------------------------------------

    # --- 2. OUTLIER REMOVAL ---
    # We remove the top 1% and bottom 1% AFTER currency conversion
    upper_limit = df['price'].quantile(0.99)
    lower_limit = df['price'].quantile(0.01)
    df = df[(df['price'] >= lower_limit) & (df['price'] <= upper_limit)]
    print(f"Dataset size after outlier removal: {len(df)}")
    print(f"Price range is now: {df['price'].min():.2f} to {df['price'].max():.2f}")
    # -----------------------
    
    # 1. Define Features (X) and Target (y)
    print("Preparing features and target...")
    X = df.drop(columns=['price'])
    y = df['price']
    
    # 2. Handle Skewed Target (Log Transformation)
    # Price is heavily right-skewed. Models perform better when predicting log(price).
    # log1p is log(1 + x), which is safer if there are any 0 values.
    y_log = np.log1p(y)
    
    # 3. Train-Test Split (80% Train, 20% Test)
    # random_state=42 ensures we get the exact same split every time we run the script
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    # Keep track of actual test prices (in dollars) for fair evaluation later
    y_test_actual = np.expm1(y_test_log)
    
    # 4. Initialize and Train Baseline Model (Using Scikit-Learn's LightGBM equivalent)
    print("Training Baseline Model...")
    model = HistGradientBoostingRegressor(
        max_iter=500,         # Number of trees
        learning_rate=0.05,   # How fast the model learns
        max_depth=8,          # Maximum depth of each tree
        random_state=42
    )
    
    model.fit(X_train, y_train_log)
    
    # 5. Predict and Evaluate
    print("Evaluating Model on Test Set...")
    y_pred_log = model.predict(X_test)
    
    # Reverse the log transformation to get actual dollar predictions
    y_pred = np.expm1(y_pred_log)
    
    # 6. Calculate Metrics
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = root_mean_squared_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)
    
    print("\n" + "="*30)
    print("BASELINE MODEL RESULTS")
    print("="*30)
    print(f"R-Squared (R2)  : {r2:.4f} (Closer to 1 is better)")
    print(f"Mean Abs Error  : ${mae:.2f}")
    print(f"Root Mean Sq Err: ${rmse:.2f}")
    print("="*30)
    
    # Save the model
    joblib.dump(model, 'baseline_model.pkl')
    print("\nModel saved to 'baseline_model.pkl'")
    
    # You could return the test data and predictions here for residual analysis later
    return X_test, y_test_actual, y_pred

if __name__ == "__main__":
    train_baseline_model('listings_with_location.csv')
