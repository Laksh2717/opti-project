import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# STEP 1: LOAD AND EXPLORE DATA

def load_data(filepath):
    """Load the bike sharing dataset"""
    df = pd.read_csv(filepath)
    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    return df

# STEP 2: PREPROCESSING

def preprocess_data(df):
    """
    Preprocess the bike sharing data
    Extract features from datetime and prepare for modeling
    """
    df = df.copy()
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Extract time-based features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    
    # Select features for modeling
    # Using numerical features that are available
    feature_columns = ['season', 'holiday', 'workingday', 'weather',  'temp', 'atemp', 'humidity', 'windspeed', 'year', 'month', 'hour', 'dayofweek']
    
    X = df[feature_columns].values
    y = df['count'].values  # Target variable
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features used: {feature_columns}")
    
    return X, y, feature_columns

# STEP 3: TRAIN/TEST SPLIT

def train_test_split(X, y, test_size=0.3, random_state=42):
    """
    Split data into train and test sets
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

# STEP 4: FEATURE SCALING (Important: Fit on train, transform on both)

def standardize_features(X_train, X_test):
    """
    Standardize features to have mean=0 and std=1
    """
    # Calculate mean and std from TRAINING data only
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1
    
    # Apply same transformation to both train and test
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled, mean, std

# STEP 5: POLYNOMIAL FEATURE GENERATION

def create_polynomial_features(X, degree, include_interactions=False):
    """
    Create polynomial features up to specified degree
    
    Parameters:
    - X: input features (n_samples, n_features)
    - degree: polynomial degree
    - include_interactions: if True, include x_i * x_j terms at degree 2
    
    Returns polynomial feature matrix
    """
    n_samples, n_features = X.shape
    
    if degree == 1:
        # Linear: just add intercept
        return np.column_stack([np.ones(n_samples), X])
    
    # Start with intercept and original features
    poly_features = [np.ones(n_samples), X]
    
    if degree >= 2:
        # Add squared terms (x_i^2)
        squared = X ** 2
        poly_features.append(squared)
        
        # Add interaction terms if requested (only for degree 2)
        if include_interactions and degree == 2:
            interactions = []
            for i in range(n_features):
                for j in range(i+1, n_features):
                    interactions.append(X[:, i] * X[:, j])
            if interactions:
                poly_features.append(np.column_stack(interactions))
    
    if degree >= 3:
        # Add cubic terms (x_i^3)
        cubed = X ** 3
        poly_features.append(cubed)
    
    if degree >= 4:
        # Add quartic terms (x_i^4)
        quartic = X ** 4
        poly_features.append(quartic)
    
    # Concatenate all features
    result = np.column_stack([f if f.ndim == 2 else f.reshape(-1, 1) 
                              for f in poly_features])
    
    print(f"Polynomial degree {degree}: {X.shape[1]} features -> {result.shape[1]} features")
    return result

# STEP 6: LINEAR REGRESSION (FROM SCRATCH - NO SKLEARN)

def fit_linear_regression(X, y):
    """
    Fit linear regression using Normal Equation: w = (X^T X)^(-1) X^T y
    
    Parameters:
    - X: feature matrix (n_samples, n_features) - should include intercept column
    - y: target vector (n_samples,)
    
    Returns: weight vector w
    """
    # Normal equation: w = (X^T X)^(-1) X^T y
    XTX = X.T @ X
    XTy = X.T @ y
    
    # Solve the system (more numerically stable than computing inverse directly)
    try:
        w = np.linalg.solve(XTX, XTy)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        print("Warning: Using pseudo-inverse due to singular matrix")
        w = np.linalg.pinv(XTX) @ XTy
    
    return w

def predict(X, w):
    """Make predictions using learned weights"""
    return X @ w

# STEP 7: EVALUATION METRICS

def compute_mse(y_true, y_pred):
    """Compute Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

def compute_r2(y_true, y_pred):
    """Compute R-squared (coefficient of determination)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def compute_rmse(y_true, y_pred):
    """Compute Root Mean Squared Error"""
    return np.sqrt(compute_mse(y_true, y_pred))

# STEP 8: MAIN TRAINING AND EVALUATION PIPELINE

def train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, 
                             degree, include_interactions=False):
    """
    Train a polynomial regression model and evaluate it
    """
    model_name = f"Polynomial Degree {degree}"
    if include_interactions and degree == 2:
        model_name += " (with interactions)"
    
    print(f"\nTraining: {model_name}\n")
    
    # Create polynomial features
    X_train_poly = create_polynomial_features(X_train_scaled, degree, include_interactions)
    X_test_poly = create_polynomial_features(X_test_scaled, degree, include_interactions)
    
    # Fit model
    print("Fitting model...")
    w = fit_linear_regression(X_train_poly, y_train)
    print(f"Number of parameters: {len(w)}")
    
    # Make predictions
    y_train_pred = predict(X_train_poly, w)
    y_test_pred = predict(X_test_poly, w)
    
    # Compute metrics
    train_mse = compute_mse(y_train, y_train_pred)
    test_mse = compute_mse(y_test, y_test_pred)
    train_r2 = compute_r2(y_train, y_train_pred)
    test_r2 = compute_r2(y_test, y_test_pred)
    train_rmse = compute_rmse(y_train, y_train_pred)
    test_rmse = compute_rmse(y_test, y_test_pred)
    
    # Print results
    print(f"\nTRAIN SET METRICS:")
    print(f"  MSE:  {train_mse:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nTEST SET METRICS:")
    print(f"  MSE:  {test_mse:.2f}")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  R²:   {test_r2:.4f}")
    
    return {
        'model_name': model_name, 'degree': degree, 'weights': w, 'train_mse': train_mse,
        'test_mse': test_mse, 'train_r2': train_r2, 'test_r2': test_r2, 
        'train_rmse': train_rmse, 'test_rmse': test_rmse, 'y_test_pred': y_test_pred
    }

# STEP 9: VISUALIZATION

def plot_results(results_list, y_test):
    """Create visualization of model comparisons"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    model_names = [r['model_name'] for r in results_list]
    test_mse = [r['test_mse'] for r in results_list]
    test_r2 = [r['test_r2'] for r in results_list]
    train_mse = [r['train_mse'] for r in results_list]
    train_r2 = [r['train_r2'] for r in results_list]
    
    # Shorten names for plotting
    short_names = [f"Deg {r['degree']}" + (" +int" if "interaction" in r['model_name'] else "") 
                   for r in results_list]
    
    # Plot 1: Test MSE comparison
    axes[0, 0].bar(short_names, test_mse, color='steelblue')
    axes[0, 0].set_title('Test Set MSE (Lower is Better)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Test R² comparison
    axes[0, 1].bar(short_names, test_r2, color='darkgreen')
    axes[0, 1].set_title('Test Set R² (Higher is Better)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Plot 3: Train vs Test MSE
    x = np.arange(len(model_names))
    width = 0.35
    axes[1, 0].bar(x - width/2, train_mse, width, label='Train MSE', color='lightcoral')
    axes[1, 0].bar(x + width/2, test_mse, width, label='Test MSE', color='steelblue')
    axes[1, 0].set_title('Train vs Test MSE', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(short_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Predictions vs Actual (for best model)
    best_idx = np.argmax(test_r2)
    best_pred = results_list[best_idx]['y_test_pred']
    axes[1, 1].scatter(y_test, best_pred, alpha=0.3, s=10)
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[1, 1].set_xlabel('Actual Count')
    axes[1, 1].set_ylabel('Predicted Count')
    axes[1, 1].set_title(f'Best Model: {results_list[best_idx]["model_name"]}', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'model_comparison.png'")
    plt.show()

# STEP 10: MAIN EXECUTION

def main():
    """Main function to run the complete analysis"""
    
    # 1. Load data
    print("\n[1] LOADING DATA...")
    df = load_data('train.csv')  # Make sure train.csv is in the same folder
    
    # 2. Preprocess
    print("\n[2] PREPROCESSING...")
    X, y, feature_names = preprocess_data(df)
    
    # 3. Split data
    print("\n[3] SPLITTING DATA (70-30)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 4. Standardize (NO LEAKAGE: fit on train only)
    print("\n[4] STANDARDIZING FEATURES (no train-test leakage)...")
    X_train_scaled, X_test_scaled, mean, std = standardize_features(X_train, X_test)
    
    # 5. Train all models
    print("\n[5] TRAINING ALL MODELS...")
    results = []
    
    # Model 1: Linear Regression (degree 1)
    results.append(train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, degree=1))
    
    # Model 2: Polynomial degree 2 (no interactions)
    results.append(train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, degree=2))
    
    # Model 3: Polynomial degree 3
    results.append(train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, degree=3))
    
    # Model 4: Polynomial degree 4
    results.append(train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, degree=4))
    
    # Model 5: Quadratic with interactions
    results.append(train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, degree=2, include_interactions=True))
    
    # 6. Summary and comparison
    print("\nFINAL COMPARISON - TEST SET PERFORMANCE")
    print(f"{'Model':<40} {'Test MSE':<15} {'Test R²':<10}")
    for r in results:
        print(f"{r['model_name']:<40} {r['test_mse']:<15.2f} {r['test_r2']:<10.4f}")
    
    # Find best model
    best_idx = np.argmax([r['test_r2'] for r in results])
    best_model = results[best_idx]
    
    print(f"\nBEST MODEL: {best_model['model_name']}")
    print(f"Test MSE: {best_model['test_mse']:.2f}")
    print(f"Test R²: {best_model['test_r2']:.4f}")
    
    # 7. Visualizations
    print("\n[6] CREATING VISUALIZATIONS...")
    plot_results(results, y_test)
    
    return results

if __name__ == "__main__":
    results = main()