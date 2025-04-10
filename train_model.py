import pandas as pd
from utils.preprocessing import preprocess_data
from pathlib import Path
import joblib
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
from datetime import datetime

def load_data():
    """Load and prepare the training data"""
    data_path = Path('data') / 'retail_store_inventory.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    return df

def train_and_save_model():
    """Main training function"""
    # Setup directories
    os.makedirs('model', exist_ok=True)
    
    # Load and prepare data
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data...")
    X = preprocess_data(df, training=True)
    y = df['units_sold'].values
    
    # Train-test split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train pipeline
    print("Training model...")
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        random_state=42,
        verbose=-1
    )
    
    pipeline = Pipeline([
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    print("\nModel Evaluation:")
    for name, X, y in [("Training", X_train, y_train), ("Test", X_test, y_test)]:
        preds = pipeline.predict(X)
        print(f"{name} MAE: {mean_absolute_error(y, preds):.2f}")
        print(f"{name} RMSE: {np.sqrt(mean_squared_error(y, preds)):.2f}")
    
    # Save pipeline
    model_path = Path('model') / 'best_lightgbm_pipeline.pkl'
    joblib.dump(pipeline, model_path)
    
    print(f"\nPipeline saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()