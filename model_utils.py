import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: Union[str, Path]):
    """
    Load the trained pipeline
    Args:
        model_path: Path to the saved pipeline file
    Returns:
        The trained pipeline object
    Raises:
        RuntimeError: If there are issues loading the pipeline
    """
    try:
        pipeline = joblib.load(model_path)
        
        if not hasattr(pipeline, 'predict'):
            raise ValueError("Loaded object is not a valid scikit-learn pipeline")
            
        logger.info(f"Model loaded successfully with {pipeline.n_features_in_} features")
        return pipeline
    except Exception as e:
        raise RuntimeError(f"Error loading model pipeline: {str(e)}")

def make_predictions(pipeline, input_data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using the loaded pipeline
    Args:
        pipeline: The trained pipeline object
        input_data: DataFrame containing input features
    Returns:
        Array of predictions
    Raises:
        RuntimeError: For prediction errors
    """
    try:
        from utils.preprocessing import preprocess_data
        
        logger.info(f"Input data columns: {input_data.columns.tolist()}")
        logger.info(f"Model expects {pipeline.n_features_in_} features")
        
        processed_data = preprocess_data(input_data, training=False)
        logger.info(f"Processed data shape: {processed_data.shape}")
        
        if processed_data.shape[1] != pipeline.n_features_in_:
            raise ValueError(
                f"Feature mismatch. Model expects {pipeline.n_features_in_} features, "
                f"got {processed_data.shape[1]}. "
                f"Input columns: {input_data.columns.tolist()}"
            )
        
        return pipeline.predict(processed_data)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise RuntimeError(f"Prediction failed: {str(e)}")

def calculate_business_impact(
    predictions: np.ndarray,
    price_data: Optional[np.ndarray] = None,
    inventory_data: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate business impact metrics
    Args:
        predictions: Predicted demand values
        price_data: Price data for cost calculations
        inventory_data: Inventory data for stock calculations
    Returns:
        Dictionary of business metrics
    """
    results = {
        'mean_demand': float(np.mean(predictions)),
        'median_demand': float(np.median(predictions)),
        'min_demand': float(np.min(predictions)),
        'max_demand': float(np.max(predictions)),
        'total_predicted': float(np.sum(predictions))
    }
    
    if price_data is not None:
        results['projected_revenue'] = float(np.sum(predictions * price_data))
        
    if inventory_data is not None:
        overstock = np.maximum(inventory_data - predictions, 0)
        understock = np.maximum(predictions - inventory_data, 0)
        
        results.update({
            'overstock_units': float(np.sum(overstock)),
            'understock_units': float(np.sum(understock))
        })
        
        if price_data is not None:
            results.update({
                'overstock_cost': float(np.sum(overstock * price_data * 0.3)),  # 30% holding cost
                'understock_cost': float(np.sum(understock * price_data * 0.5)),  # 50% opportunity cost
                'total_cost': float(np.sum(overstock * price_data * 0.3)) + 
                             float(np.sum(understock * price_data * 0.5))
            })
    
    return results

def generate_recommendations(
    predictions: np.ndarray,
    data: pd.DataFrame,
    threshold: float = 0.2
) -> List[str]:
    """
    Generate inventory recommendations
    Args:
        predictions: Predicted demand values
        data: DataFrame containing product info
        threshold: Change threshold for recommendations
    Returns:
        List of recommendation strings
    """
    if 'product_name' not in data.columns:
        return ["⚠️ Product information missing for recommendations"]
    
    # Calculate average demand per product
    data['predicted_demand'] = predictions
    product_stats = data.groupby('product_name').agg({
        'predicted_demand': 'mean',
        'inventory_level': 'mean',
        'price': 'mean'
    }).reset_index()
    
    product_stats['demand_inventory_ratio'] = (
        product_stats['predicted_demand'] / 
        (product_stats['inventory_level'] + 1e-10))
    
    recommendations = []
    
    for _, row in product_stats.iterrows():
        ratio = row['demand_inventory_ratio']
        if ratio > (1 + threshold):
            rec = (f"📈 Increase inventory for {row['product_name']} "
                  f"(demand {row['predicted_demand']:.0f} vs inventory {row['inventory_level']:.0f})")
            recommendations.append(rec)
        elif ratio < (1 - threshold):
            rec = (f"📉 Decrease inventory for {row['product_name']} "
                  f"(demand {row['predicted_demand']:.0f} vs inventory {row['inventory_level']:.0f})")
            recommendations.append(rec)
    
    if not recommendations:
        return ["🟢 Inventory levels balanced - no changes recommended"]
    
    # Add summary recommendation
    net_effect = sum(1 for r in recommendations if "Increase" in r) - \
                 sum(1 for r in recommendations if "Decrease" in r)
    summary = (
        "🔴 Consider increasing overall inventory" if net_effect > 3 else
        "🔵 Consider decreasing overall inventory" if net_effect < -3 else
        "🟡 Moderate inventory adjustments needed"
    )
    recommendations.insert(0, summary)
    
    return recommendations