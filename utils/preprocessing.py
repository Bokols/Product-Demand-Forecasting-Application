import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

PREPROCESSOR_PATH = Path(__file__).parent.parent / 'model' / 'preprocessor.joblib'

def clean_column_names(df):
    """Clean column names by replacing spaces and special characters"""
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('/', '_')
    return df

def add_product_names(df):
    """Add product names based on product_id"""
    product_name_mapping = {
        'P0001': 'Premium Olive Oil',
        'P0002': 'LEGO Classic Creative Brick Box',
        'P0003': 'Stuffed Teddy Bear',
        'P0004': 'Remote Control Race Car',
        'P0005': 'Wireless Bluetooth Earbuds',
        'P0006': 'Organic Coffee Beans (1kg)',
        'P0007': 'Foldable Side Table',
        'P0008': "Men's Winter Jacket",
        'P0009': 'USB-C Fast Charging Cable',
        'P0010': 'Nerf Blaster Elite Disruptor',
        'P0011': 'Bookshelf (5-Tier)',
        'P0012': "Women's Casual Sneakers",
        'P0013': 'Board Game: Monopoly Classic',
        'P0014': 'Denim Jeans (Slim Fit)',
        'P0015': 'Leather Crossbody Bag',
        'P0016': 'Portable Phone Power Bank',
        'P0017': 'Jigsaw Puzzle (100 Pieces)',
        'P0018': 'Cotton T-Shirt (Unisex)',
        'P0019': 'Silk Scarf (Printed)',
        'P0020': 'Play-Doh 10-Pack Set'
    }
    df['product_name'] = df['product_id'].map(product_name_mapping)
    return df

def get_feature_config():
    """Return feature configuration to ensure consistency"""
    return {
        'categorical': ['store_id', 'product_id', 'category', 'region', 
                       'weather_condition', 'seasonality', 'holiday_promotion'],
        'numerical': ['price', 'discount', 'competitor_pricing', 'year',
                     'month', 'day', 'day_of_week', 'day_of_year',
                     'week_of_year', 'quarter', 'days_since_start',
                     'price_discount_ratio', 'price_competitor_diff',
                     'inventory_turnover'],
        'required': ['date', 'store_id', 'product_id', 'category', 'region',
                    'price', 'discount', 'competitor_pricing', 'weather_condition',
                    'seasonality', 'holiday_promotion', 'units_sold', 'inventory_level']
    }

def create_preprocessor():
    """Create the column transformer with all preprocessing steps"""
    config = get_feature_config()
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, config['numerical']),
            ('cat', categorical_transformer, config['categorical'])
        ])
    
    return preprocessor

def preprocess_data(df, training=True):
    """Preprocess data ensuring feature consistency"""
    # Initial cleaning
    df = clean_column_names(df)
    if 'product_name' not in df.columns:
        df = add_product_names(df)
    
    # Validate required columns
    config = get_feature_config()
    missing_cols = [col for col in config['required'] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert and extract date features
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    # Feature engineering
    min_date = df['date'].min() if training else pd.to_datetime('2022-01-01')
    df['days_since_start'] = (df['date'] - min_date).dt.days
    df['price_discount_ratio'] = df['discount'] / (df['price'] + 1e-10)  # Avoid division by zero
    df['price_competitor_diff'] = df['price'] - df['competitor_pricing']
    df['inventory_turnover'] = df['units_sold'] / (df['inventory_level'] + 1e-10)
    
    if training:
        preprocessor = create_preprocessor()
        processed_data = preprocessor.fit_transform(df)
        
        # Save preprocessor with metadata
        PREPROCESSOR_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({
            'preprocessor': preprocessor,
            'feature_names': preprocessor.get_feature_names_out(),
            'config': get_feature_config(),
            'training_columns': df.columns.tolist(),
            'date_min': min_date
        }, PREPROCESSOR_PATH)
    else:
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError("Preprocessor not found. Train model first.")
            
        preprocessor_data = joblib.load(PREPROCESSOR_PATH)
        preprocessor = preprocessor_data['preprocessor']
        
        # Ensure all expected columns exist
        for col in preprocessor_data['config']['required']:
            if col not in df.columns:
                df[col] = 0  # Add missing columns with default values
        
        processed_data = preprocessor.transform(df)
        
        # Validate feature count
        expected_features = preprocessor_data['feature_names']
        if processed_data.shape[1] != len(expected_features):
            raise ValueError(
                f"Feature mismatch. Expected {len(expected_features)} features, "
                f"got {processed_data.shape[1]}. "
                "Ensure input data matches training structure."
            )
    
    return processed_data