import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model_utils import load_model, make_predictions, calculate_business_impact, generate_recommendations
from utils.preprocessing import clean_column_names, add_product_names
from datetime import datetime, timedelta
from pathlib import Path

# Page configuration
st.set_page_config(page_title="Demand Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Demand Predictor")
st.markdown("""
Generate demand forecasts and analyze business impact.  
Adjust parameters in the sidebar and click **Generate Forecast** to see predictions.
""")

# Add introductory explanation
with st.expander("ℹ️ About this tool", expanded=False):
    st.markdown("""
    This demand predictor uses machine learning to forecast product demand based on:
    - Historical sales patterns
    - Current inventory levels
    - Pricing and promotion data
    - External factors like seasonality and weather
    
    **Key features**:
    - Scenario planning with adjustable business parameters
    - Inventory optimization recommendations
    - Business impact projections
    - Exportable forecast data
    """)

@st.cache_resource
def load_forecast_model():
    """Load the trained pipeline"""
    model_path = Path('model') / 'best_lightgbm_pipeline.pkl'
    try:
        pipeline = load_model(model_path)
        st.session_state['model_features'] = pipeline.n_features_in_
        return pipeline
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()

@st.cache_data
def load_historical_data():
    """Load historical data"""
    data_path = Path('data') / 'retail_store_inventory.csv'
    try:
        df = pd.read_csv(data_path)
        df = clean_column_names(df)
        df = add_product_names(df)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"❌ Failed to load data: {str(e)}")
        st.stop()

def generate_prediction_input(historical_df, future_dates):
    """Generate prediction dataframe with all required features"""
    try:
        # Use most recent data as template
        template = historical_df[historical_df['date'] == historical_df['date'].max()].copy()
        
        # Create future dates
        prediction_data = []
        for date in future_dates:
            temp = template.copy()
            temp['date'] = pd.to_datetime(date)
            prediction_data.append(temp)
        
        prediction_df = pd.concat(prediction_data)
        
        # Ensure all required features exist
        required_features = [
            'store_id', 'product_id', 'category', 'region',
            'price', 'discount', 'competitor_pricing',
            'weather_condition', 'seasonality', 'holiday_promotion',
            'units_sold', 'inventory_level'
        ]
        
        for feat in required_features:
            if feat not in prediction_df.columns:
                prediction_df[feat] = template[feat].mode()[0] if feat in template else 0
        
        prediction_df['date'] = pd.to_datetime(prediction_df['date'])
        
        # Add date features
        date_features = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'quarter']
        for feat in date_features:
            prediction_df[feat] = getattr(prediction_df['date'].dt, feat.lower())
        
        prediction_df['week_of_year'] = prediction_df['date'].dt.isocalendar().week
        
        # Add derived features
        min_date = historical_df['date'].min()
        prediction_df['days_since_start'] = (prediction_df['date'] - min_date).dt.days
        prediction_df['price_discount_ratio'] = prediction_df['discount'] / (prediction_df['price'] + 1e-10)
        prediction_df['price_competitor_diff'] = prediction_df['price'] - prediction_df['competitor_pricing']
        prediction_df['inventory_turnover'] = prediction_df['units_sold'] / (prediction_df['inventory_level'] + 1e-10)
        
        return prediction_df
    except Exception as e:
        st.error(f"❌ Error generating input: {str(e)}")
        st.stop()

# Load model and data
model = load_forecast_model()
historical_df = load_historical_data()

# Sidebar controls with tooltips
st.sidebar.header("Prediction Parameters")

# Forecast horizon with explanation
horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    options=["Next 7 days", "Next 14 days", "Next 30 days"],
    index=0,
    help="Select how many days into the future you want to predict. The model generates daily demand predictions."
)

n_days = 7 if "7" in horizon else 14 if "14" in horizon else 30

# Generate future dates starting from today
today = datetime.now().date()
future_dates = [today + timedelta(days=i) for i in range(1, n_days+1)]
prediction_input = generate_prediction_input(historical_df, future_dates)

# Product selection with explanation
selected_products = st.sidebar.multiselect(
    "Products to Forecast",
    options=historical_df['product_name'].unique(),
    default=historical_df['product_name'].unique()[:3],
    help="Select specific products to analyze. Leave blank to include all products."
)

if selected_products:
    prediction_input = prediction_input[prediction_input['product_name'].isin(selected_products)]

# Business parameters section with explanations
st.sidebar.subheader("Business Scenarios")
st.sidebar.markdown("Adjust these parameters to simulate different business conditions:")

price_adjustment = st.sidebar.slider(
    "Price Adjustment (%)", 
    -50, 50, 0, 5,
    help="Simulate price changes. Positive values = price increases, negative = decreases. Affects demand elasticity."
)

discount_adjustment = st.sidebar.slider(
    "Discount Adjustment (%)", 
    -100, 100, 0, 5,
    help="Change discount levels. Positive values = more promotions, negative = fewer. Max 100% discount."
)

comp_adjustment = st.sidebar.slider(
    "Competitor Price Adjustment (%)", 
    -50, 50, 0, 5,
    help="Simulate competitor price changes. Positive values = they increase prices, negative = they discount."
)

holiday_promotion = st.sidebar.checkbox(
    "Holiday/Promotion Period", 
    False,
    help="Check this to account for seasonal demand surges during holidays or special promotions."
)

# Apply adjustments
prediction_input['price'] *= (1 + price_adjustment/100)
prediction_input['discount'] = np.clip(prediction_input['discount'] * (1 + discount_adjustment/100), 0, 100)
prediction_input['competitor_pricing'] *= (1 + comp_adjustment/100)
prediction_input['holiday_promotion'] = 'yes' if holiday_promotion else 'no'

# Generate predictions button with status explanation
if st.sidebar.button("Generate Forecast", type="primary"):
    with st.spinner("Generating predictions... This may take a moment for large datasets."):
        try:
            # Validate all required columns exist
            required_columns = [
                'store_id', 'product_id', 'category', 'region',
                'price', 'discount', 'competitor_pricing',
                'weather_condition', 'seasonality', 'holiday_promotion',
                'units_sold', 'inventory_level',
                'year', 'month', 'day', 'day_of_week', 'day_of_year',
                'week_of_year', 'quarter', 'days_since_start',
                'price_discount_ratio', 'price_competitor_diff',
                'inventory_turnover'
            ]
            
            missing_columns = [col for col in required_columns if col not in prediction_input.columns]
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
                st.stop()
            
            predictions = make_predictions(model, prediction_input)
            prediction_input['predicted_demand'] = predictions
            
            # Calculate business impact
            business_impact = calculate_business_impact(
                predictions,
                price_data=prediction_input['price'].values,
                inventory_data=prediction_input['inventory_level'].values
            )
            
            # Generate recommendations
            recommendations = generate_recommendations(predictions, prediction_input)
            
            # Store results
            st.session_state['predictions'] = prediction_input
            st.session_state['business_impact'] = business_impact
            st.session_state['recommendations'] = recommendations
            
            st.success("✅ Forecast generated successfully!")
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

# Display results with enhanced explanations
if 'predictions' in st.session_state:
    predictions_df = st.session_state['predictions']
    business_impact = st.session_state['business_impact']
    recommendations = st.session_state['recommendations']
    
    # Metrics with explanations
    st.header("Forecast Summary")
    with st.expander("Understanding these metrics"):
        st.markdown("""
        - **Total Predicted Demand**: Sum of all units expected to be sold
        - **Average Daily Demand**: Mean units sold per day
        - **Projected Revenue**: Estimated income (price × predicted demand)
        - **Total Inventory Cost**: Holding costs for overstock + lost sales from understock
        """)
    
    cols = st.columns(4)
    cols[0].metric("Total Predicted Demand", f"{business_impact['total_predicted']:,.0f} units")
    cols[1].metric("Average Daily Demand", f"{business_impact['mean_demand']:,.1f} units")
    cols[2].metric("Projected Revenue", f"${business_impact.get('projected_revenue', 0):,.0f}")
    cols[3].metric("Total Inventory Cost", f"${business_impact.get('total_cost', 0):,.0f}")
    
    # Visualization with explanation
    st.header("Demand Forecast")
    st.markdown(f"""
    *Daily predicted demand for selected products over {horizon.lower()}*
    """)
    fig = px.line(
        predictions_df.groupby(['date', 'product_name'])['predicted_demand'].sum().reset_index(),
        x='date', y='predicted_demand', color='product_name',
        title=f"{horizon} Demand Forecast",
        labels={'predicted_demand': 'Predicted Demand (units)', 'date': 'Date'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Business Impact with detailed explanations
    st.header("Inventory Analysis")
    with st.expander("How to interpret inventory analysis"):
        st.markdown("""
        - **Overstock Units**: Items likely to remain unsold (inventory > demand)
        - **Understock Units**: Potential lost sales (demand > inventory)
        - Optimal inventory matches predicted demand exactly
        """)
    
    col1, col2 = st.columns(2)
    col1.metric("Potential Overstock Units", 
               f"{business_impact.get('overstock_units', 0):,.0f}",
               help="Units that may remain unsold based on current inventory")
    col2.metric("Potential Understock Units", 
               f"{business_impact.get('understock_units', 0):,.0f}",
               help="Potential lost sales due to insufficient inventory")
    
    # Recommendations with priority indicators
    st.header("Inventory Recommendations")
    st.markdown("""
    *Color indicators show recommendation priority*  
    🟥 **High priority** | 🟨 **Medium priority** | 🟩 **Low priority**
    """)
    
    for rec in recommendations:
        if "Increase" in rec and "!" in rec:
            st.error(rec)
        elif "Increase" in rec:
            st.warning(rec)
        elif "Decrease" in rec and "!" in rec:
            st.error(rec)
        elif "Decrease" in rec:
            st.warning(rec)
        else:
            st.success(rec)
    
    # Data Export with format options
    st.header("Export Results")
    with st.expander("Export options"):
        st.markdown("""
        Download forecast data in your preferred format:
        - CSV for spreadsheets
        - JSON for APIs and developers
        - Excel for detailed analysis
        """)
    
    export_format = st.selectbox("Export format", ["CSV", "Excel", "JSON"])
    
    if st.button(f"Download Forecast Data as {export_format}"):
        if export_format == "CSV":
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"demand_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        elif export_format == "Excel":
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                predictions_df.to_excel(writer, index=False)
            st.download_button(
                label="Download Excel",
                data=output.getvalue(),
                file_name=f"demand_forecast_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime='application/vnd.ms-excel'
            )
        elif export_format == "JSON":
            json = predictions_df.to_json(orient='records')
            st.download_button(
                label="Download JSON",
                data=json,
                file_name=f"demand_forecast_{datetime.now().strftime('%Y%m%d')}.json",
                mime='application/json'
            )
else:
    st.info("""
    ℹ️ No forecast generated yet.  
    Adjust parameters in the sidebar and click **Generate Forecast** to see predictions.
    """)

# Add footer with model information
st.markdown("---")
st.caption(f"""
Model last trained: {datetime.fromtimestamp(Path('model/best_lightgbm_pipeline.pkl').stat().st_mtime).strftime('%Y-%m-%d')}  
Using LightGBM algorithm with {model.n_features_in_} features
""")