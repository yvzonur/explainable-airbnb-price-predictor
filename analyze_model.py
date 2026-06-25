import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from train_baseline import train_baseline_model

def run_analysis():
    print("1. Running model to get test data and predictions...")
    # This retrains the model quickly to get the exact X_test, y_test, y_pred
    X_test, y_test, y_pred = train_baseline_model('listings_with_location.csv')
    
    # Reload model object
    model = joblib.load('baseline_model.pkl')
    
    residuals = y_test - y_pred
    
    print("\n2. Generating Residual Plots...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color='blue', edgecolor='k')
    
    # Draw perfect prediction line
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Price (TRY)')
    plt.ylabel('Predicted Price (TRY)')
    plt.title('Residual Analysis: Actual vs Predicted Prices')
    plt.legend()
    plt.tight_layout()
    plt.savefig('residual_scatter.png')
    plt.close()
    
    print("3. Segmenting Errors by Room Type...")
    # Find room type columns in X_test
    room_cols = [c for c in X_test.columns if 'room_type_' in c]
    
    analysis_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': np.abs(residuals)
    })
    
    for col in room_cols:
        mask = X_test[col] == 1
        if mask.sum() > 0:
            mae = analysis_df.loc[mask, 'Error'].mean()
            count = mask.sum()
            print(f"  - MAE for {col.replace('room_type_', '')} (N={count}): {mae:.2f} TRY")
            
    print("\n4. Running SHAP Analysis...")
    # We take a sample of 1000 for SHAP to avoid memory/time issues
    sample_size = min(1000, len(X_test))
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test.iloc[sample_indices]
    
    try:
        print("   Trying shap.Explainer (Fast mode)...")
        # For HistGradientBoosting, shap.Explainer can sometimes fallback automatically to Exact
        # We limit sample to 200 just in case to avoid freezing
        X_sample_small = X_sample.head(200)
        explainer = shap.Explainer(model.predict, X_sample_small)
        shap_values = explainer(X_sample_small)
    except Exception as e:
        print(f"   Explainer failed: {e}. Falling back to basic feature importances or skipping SHAP.")
        return
        
    print("5. Generating SHAP Plots...")
    # Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample_small, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
    
    # Dependence Plot for distance_to_center_km
    if 'distance_to_center_km' in X_sample_small.columns:
        plt.figure(figsize=(10, 6))
        # Note: Depending on shap version, the API for dependence_plot varies.
        # This works for old SHAP versions with matrix, or new SHAP with Explanation objects.
        try:
            shap.dependence_plot('distance_to_center_km', shap_values.values, X_sample_small, show=False)
        except AttributeError:
            shap.dependence_plot('distance_to_center_km', shap_values, X_sample_small, show=False)
        
        plt.tight_layout()
        plt.savefig('shap_dependence.png')
        plt.close()
        
    print("Analysis complete! Plots saved as PNG files.")

if __name__ == "__main__":
    run_analysis()
