import streamlit as st
import pandas as pd
import numpy as np
from utils.visualization import (
    plot_demand_trend, 
    plot_demand_distribution,
    plot_feature_importance
)
from utils.preprocessing import preprocess_data, clean_column_names, add_product_names
import plotly.express as px
import os
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
st.title("📊 Data Explorer")
st.markdown("""
Explore the retail demand forecasting dataset and analyze trends.  
Use the sidebar filters to focus on specific products, time periods, or regions.
""")

# Add introductory guide
with st.expander("ℹ️ How to Use This Explorer", expanded=False):
    st.markdown("""
    **Data Explorer Guide**:
    
    1. **Filter data** using the sidebar controls
    2. **View metrics** in the summary section
    3. **Analyze patterns** in the interactive charts
    4. **Export insights** using the download options
    
    **Key Features**:
    - Demand trend analysis
    - Price elasticity visualization
    - Inventory optimization insights
    - Business impact calculations
    """)

@st.cache_data
def load_data():
    """Load and preprocess the retail data"""
    data_path = os.path.join('data', 'retail_store_inventory.csv')
    df = pd.read_csv(data_path)
    df = clean_column_names(df)
    df = add_product_names(df)
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract date components with explanations
    date_features = {
        'year': 'Year extracted from date',
        'month': 'Month (1-12)',
        'day': 'Day of month',
        'day_of_week': 'Day of week (0=Monday)',
        'day_of_year': 'Day of year (1-365)',
        'week_of_year': 'ISO week number',
        'quarter': 'Quarter (1-4)'
    }
    
    for feat, desc in date_features.items():
        if feat == 'week_of_year':
            df['week_of_year'] = df['date'].dt.isocalendar().week
        elif feat == 'quarter':
            df['quarter'] = df['date'].dt.quarter
        else:
            df[feat] = getattr(df['date'].dt, feat)
    
    return df

# Load data with status
with st.spinner("Loading data..."):
    df = load_data()

# Sidebar filters with tooltips
st.sidebar.header("Data Filters")
st.sidebar.markdown("Use these controls to focus on specific data subsets:")

# Date range filter with explanation
min_date = df['date'].min().date()
max_date = df['date'].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Select start and end dates to analyze a specific time period"
)

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    st.sidebar.caption(f"Showing data from {start_date} to {end_date}")
else:
    st.sidebar.warning("Please select both start and end dates")

# Product filter with search capability
products = st.sidebar.multiselect(
    "Products",
    options=df['product_name'].unique(),
    default=df['product_name'].unique()[:3],
    help="Select specific products or leave blank for all products"
)
if products:
    df = df[df['product_name'].isin(products)]

# Category filter with explanation
categories = st.sidebar.multiselect(
    "Categories",
    options=df['category'].unique(),
    default=df['category'].unique(),
    help="Filter by product category"
)
if categories:
    df = df[df['category'].isin(categories)]

# Region filter with explanation
regions = st.sidebar.multiselect(
    "Regions",
    options=df['region'].unique(),
    default=df['region'].unique(),
    help="Focus on specific geographic regions"
)
if regions:
    df = df[df['region'].isin(regions)]

# Store filter with search help
stores = st.sidebar.multiselect(
    "Stores",
    options=df['store_id'].unique(),
    default=df['store_id'].unique()[:3],
    help="Select specific store locations"
)
if stores:
    df = df[df['store_id'].isin(stores)]

# Main content
st.header("Dataset Overview")
st.write(f"Displaying {len(df):,} rows from the filtered dataset")

# Raw data toggle with warning
if st.checkbox("Show raw data", help="View the underlying dataset (may be large)"):
    st.dataframe(df)
    st.download_button(
        label="Download Filtered Data",
        data=df.to_csv(index=False),
        file_name="filtered_retail_data.csv",
        mime="text/csv"
    )

# Key metrics with explanations
st.subheader("Key Metrics")
with st.expander("About these metrics"):
    st.markdown("""
    - **Total Units Sold**: Sum of all products sold in filtered dataset
    - **Average Daily Demand**: Mean units sold per day
    - **Total Inventory Value**: Current stock × price (at time of recording)
    - **Average Discount**: Mean promotional discount percentage
    """)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Units Sold", f"{df['units_sold'].sum():,.0f}")
col2.metric("Average Daily Demand", f"{df['units_sold'].mean():,.1f}")
col3.metric("Total Inventory Value", f"${(df['inventory_level'] * df['price']).sum():,.0f}")
col4.metric("Average Discount", f"{df['discount'].mean():.1f}%")

# Demand analysis section
st.header("Demand Analysis")
st.markdown("Explore demand patterns using the interactive tabs below:")

tab1, tab2, tab3 = st.tabs(["Trends", "Distribution", "Seasonality"])

with tab1:
    st.subheader("Demand Trends Over Time")
    st.markdown("*How demand changes across your selected time period*")
    
    groupby_trend = st.selectbox(
        "Group by",
        options=[None, 'product_name', 'category', 'region', 'store_id'],
        index=0,
        help="Break down trends by specific dimensions"
    )
    fig_trend = plot_demand_trend(df, groupby=groupby_trend)
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.subheader("Demand Distribution")
    st.markdown("*How demand is distributed across different factors*")
    
    groupby_dist = st.selectbox(
        "Group distribution by",
        options=[None, 'product_name', 'category', 'region', 'store_id', 'seasonality', 'weather_condition'],
        index=0,
        help="Compare demand distributions across categories"
    )
    fig_dist = plot_demand_distribution(df, groupby=groupby_dist)
    st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    st.subheader("Seasonal Patterns")
    st.markdown("*Recurring demand patterns by time periods*")
    
    # Monthly seasonality
    if 'month' in df.columns:
        st.markdown("**Monthly Patterns**")
        monthly_agg = df.groupby(['month', 'product_name'])['units_sold'].sum().reset_index()
        fig_monthly = px.line(monthly_agg, x='month', y='units_sold', color='product_name',
                             title='Monthly Demand by Product',
                             labels={'month': 'Month', 'units_sold': 'Units Sold'})
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Day of week seasonality
    if 'day_of_week' in df.columns:
        st.markdown("**Day-of-Week Patterns**")
        dow_agg = df.groupby(['day_of_week', 'product_name'])['units_sold'].sum().reset_index()
        fig_dow = px.line(dow_agg, x='day_of_week', y='units_sold', color='product_name',
                         title='Weekly Demand Patterns',
                         labels={'day_of_week': 'Day of Week (0=Monday)', 'units_sold': 'Units Sold'})
        st.plotly_chart(fig_dow, use_container_width=True)

# Price and discount analysis with explanations
st.header("Price Sensitivity Analysis")
st.markdown("""
Examine how pricing and promotions affect demand:  
- **Price Elasticity**: How demand changes with price
- **Discount Impact**: Effectiveness of promotions
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price vs Demand")
    st.markdown("*Relationship between price and units sold*")
    fig_price = px.scatter(df, x='price', y='units_sold', color='category',
                          trendline="lowess",
                          title="Price Elasticity of Demand",
                          labels={'price': 'Price ($)', 'units_sold': 'Units Sold'})
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    st.subheader("Discount Impact")
    st.markdown("*How discounts affect sales volume*")
    fig_discount = px.box(df, x='discount', y='units_sold', color='category',
                         title="Demand Distribution by Discount Level",
                         labels={'discount': 'Discount (%)', 'units_sold': 'Units Sold'})
    st.plotly_chart(fig_discount, use_container_width=True)

# Inventory analysis with business context
st.header("Inventory Optimization")
st.markdown("""
Key metrics for inventory management:  
- **Turnover**: How quickly inventory sells  
- **Stockout Risk**: Probability of running out  
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Inventory Turnover")
    st.markdown("*Sales relative to inventory levels*")
    df['inventory_turnover'] = df['units_sold'] / df['inventory_level']
    fig_turnover = px.box(df, x='category', y='inventory_turnover',
                         title="Inventory Turnover by Category",
                         labels={'category': 'Category', 'inventory_turnover': 'Turnover Ratio'})
    st.plotly_chart(fig_turnover, use_container_width=True)

with col2:
    st.subheader("Stockout Risk")
    st.markdown("*Likelihood of inventory depletion*")
    df['stockout_risk'] = (df['units_sold'] / df['inventory_level']).clip(upper=1)
    fig_stockout = px.box(df, x='category', y='stockout_risk',
                         title="Stockout Risk by Category",
                         labels={'category': 'Category', 'stockout_risk': 'Risk Probability'})
    st.plotly_chart(fig_stockout, use_container_width=True)

# Economic impact analysis with explanations
st.header("Business Impact Analysis")
st.markdown("""
Financial implications of inventory decisions:  
- **Revenue Loss**: From missed sales opportunities  
- **Overstock Costs**: From excess inventory holding  
""")

# Calculate potential revenue loss from stockouts
df['potential_revenue_loss'] = np.where(
    df['units_sold'] > df['inventory_level'],
    (df['units_sold'] - df['inventory_level']) * df['price'],
    0
)

# Calculate overstock cost
df['overstock_cost'] = np.where(
    df['inventory_level'] > df['units_sold'],
    (df['inventory_level'] - df['units_sold']) * df['price'] * 0.3,  # 30% holding cost
    0
)

total_revenue_loss = df['potential_revenue_loss'].sum()
total_overstock_cost = df['overstock_cost'].sum()

col1, col2 = st.columns(2)
col1.metric("Total Potential Revenue Loss from Stockouts", 
           f"${total_revenue_loss:,.0f}",
           help="Lost revenue from insufficient inventory")
col2.metric("Total Overstock Holding Costs", 
           f"${total_overstock_cost:,.0f}",
           help="Costs from excess inventory (30% holding cost)")

# Performance products table with context
st.subheader("Top Products Needing Inventory Adjustment")
st.markdown("""
Products with the largest inventory mismatches:  
🔴 = High priority | 🟡 = Medium priority | 🔵 = Low priority  
""")

worst_products = df.groupby('product_name').agg({
    'potential_revenue_loss': 'sum',
    'overstock_cost': 'sum'
}).sort_values('potential_revenue_loss', ascending=False).head(10)

# Add priority indicators
def highlight_priority(row):
    if row['potential_revenue_loss'] > 10000:
        return ['background-color: #ffcccc'] * len(row)
    elif row['overstock_cost'] > 5000:
        return ['background-color: #ffffcc'] * len(row)
    return [''] * len(row)

st.dataframe(
    worst_products.style.format("${:,.0f}").apply(highlight_priority, axis=1)
)

# Model training section with warning
st.sidebar.header("Model Training")
st.sidebar.markdown("""
⚠️ Advanced Feature  
Only use if you've updated the dataset significantly
""")

if st.sidebar.button("Fit and Save Preprocessor"):
    with st.spinner("Fitting and saving preprocessor..."):
        try:
            from utils.preprocessing import preprocess_data
            preprocess_data(df, training=True)
            st.sidebar.success("Preprocessor fitted and saved successfully!")
        except Exception as e:
            st.sidebar.error(f"Error fitting preprocessor: {str(e)}")

# Add data freshness indicator
st.markdown("---")
st.caption(f"Data last loaded: {datetime.now().strftime('%Y-%m-%d %H:%M')}")