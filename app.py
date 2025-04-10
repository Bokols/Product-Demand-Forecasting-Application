import streamlit as st
from PIL import Image
import os
from datetime import datetime
import requests
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom sidebar
with st.sidebar:
    # Profile section
    st.header("Bo Kolstrup")
    st.caption("Data Scientist | Digital Marketing Specialist | Python")
    
    # Load local profile image
    try:
        profile_img = Image.open("assets/Bo-Kolstrup.png")
        st.image(profile_img, width=200, use_column_width='auto')
    except Exception as e:
        st.warning(f"Couldn't load profile image: {str(e)}")
        profile_img = None

    # About section
    st.markdown("""
    ### About Me
    I am passionate about applying data science and machine learning to solve real-world business challenges.
    """)
    
    # Social links with icons
    st.markdown("""
    ### Connect With Me
    <div style="display: flex; gap: 10px; margin-bottom: 20px;">
        <a href="https://www.linkedin.com/in/bo-kolstrup/" target="_blank">
            <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width="28">
        </a>
        <a href="https://github.com/Bokols" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="28">
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Updated navigation with your requested names
    st.subheader("Navigation")
    
    # Home button - stays on current page when already on home
    if st.button("🏠 Home", 
                help="Return to main dashboard",
                disabled=st.session_state.get('current_page') == 'home',
                use_container_width=True):
        st.session_state.current_page = 'home'
        st.switch_page("app.py")
    
    # Data Analysis Hub button
    if st.button("📊 Data Analysis Hub", 
                help="Explore historical data and trends",
                disabled=st.session_state.get('current_page') == 'explorer',
                use_container_width=True):
        st.session_state.current_page = 'explorer'
        st.switch_page("pages/1_📊_Explorer.py")
    
    # AI Forecasting Lab button
    if st.button("🔮 AI Forecasting Lab", 
                help="Generate demand predictions and recommendations",
                disabled=st.session_state.get('current_page') == 'predict',
                use_container_width=True):
        st.session_state.current_page = 'predict'
        st.switch_page("pages/2_🔮_Predict.py")

# Initialize session state for page tracking
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Main content
st.title("🛒 Retail Demand Forecasting")
st.markdown("*AI-powered inventory optimization for modern retailers*")

# Project overview
with st.expander("📌 About This Project", expanded=True):
    st.markdown("""
    ## 🔍 What is this tool?
    
    A **machine learning-powered dashboard** that helps retailers:
    - 📈 Forecast product demand with 85%+ accuracy
    - 📊 Analyze historical sales patterns
    - 🛍️ Optimize inventory levels
    - 💰 Reduce overstock and stockout costs

    ## 🧠 How It Works
    
    1. **Data Integration**:  
       - Ingests historical sales, inventory, and pricing data
       - Incorporates external factors (weather, promotions, etc.)
    
    2. **AI Forecasting**:  
       - Uses LightGBM machine learning model
       - Trained on 2+ years of retail data
       - Updates predictions daily
    
    3. **Actionable Insights**:  
       - Automated inventory recommendations
       - Scenario planning tools
       - Financial impact projections
    """)

# Features overview
st.markdown("---")
st.header("Key Features")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Data Analysis Hub")
    st.markdown("""
    - Interactive data visualization
    - Demand trend analysis
    - Price elasticity modeling
    - Seasonal pattern detection
    - Inventory health metrics
    """)

with col2:
    st.subheader("🔮 AI Forecasting Lab")
    st.markdown("""
    - 30-day demand forecasting
    - What-if scenario testing
    - Automated recommendations
    - Business impact analysis
    - Exportable reports
    """)

# Footer
st.markdown("---")
st.caption(f"""
🚀 Developed by Bo Kolstrup | 📅 Last updated: {datetime.now().strftime("%Y-%m-%d")}  
[GitHub Repository](https://github.com/Bokols) | [Contact](mailto:bokolstrup@gmail.com)
""")