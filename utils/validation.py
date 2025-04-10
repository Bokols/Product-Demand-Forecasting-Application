import pandas as pd
import joblib
from pathlib import Path

def validate_prediction_input(input_df):
    """Validate prediction input matches training structure"""
    preprocessor_path = Path(__file__).parent.parent / 'model' / 'preprocessor.joblib'
    
    if not preprocessor_path.exists():
        raise FileNotFoundError("No trained preprocessor found")
    
    preprocessor_data = joblib.load(preprocessor_path)
    config = preprocessor_data['config']
    
    # Check required columns
    missing_cols = [col for col in config['required'] if col not in input_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check categorical values
    for cat_col in config['categorical']:
        if cat_col in input_df:
            unique_values = input_df[cat_col].unique()
            if len(unique_values) > 20:  # Sanity check for categoricals
                raise ValueError(f"Column '{cat_col}' has too many unique values "
                               f"({len(unique_values)}). Is it categorical?")
    
    # Check numerical ranges
    for num_col in config['numerical']:
        if num_col in input_df:
            if input_df[num_col].isnull().any():
                raise ValueError(f"Numerical column '{num_col}' contains null values")
    
    return True