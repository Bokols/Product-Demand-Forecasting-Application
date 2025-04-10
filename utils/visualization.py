import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_demand_trend(df, time_col='date', demand_col='units_sold', groupby=None):
    """Plot demand trend over time"""
    if groupby:
        df_agg = df.groupby([time_col, groupby])[demand_col].sum().reset_index()
        fig = px.line(df_agg, x=time_col, y=demand_col, color=groupby,
                     title=f'Demand Trend by {groupby}')
    else:
        df_agg = df.groupby(time_col)[demand_col].sum().reset_index()
        fig = px.line(df_agg, x=time_col, y=demand_col, 
                     title='Overall Demand Trend')
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Units Sold',
        hovermode='x unified'
    )
    return fig

def plot_demand_distribution(df, demand_col='units_sold', groupby=None):
    """Plot distribution of demand"""
    if groupby:
        fig = px.box(df, x=groupby, y=demand_col, 
                    title=f'Demand Distribution by {groupby}')
    else:
        fig = px.histogram(df, x=demand_col, nbins=50,
                          title='Demand Distribution')
    
    fig.update_layout(
        yaxis_title='Count' if not groupby else 'Units Sold',
        xaxis_title='Units Sold' if not groupby else groupby
    )
    return fig

def plot_feature_importance(feature_importance, top_n=20):
    """Plot feature importance from the model"""
    importance_df = pd.DataFrame({
        'feature': feature_importance['feature'],
        'importance': feature_importance['importance']
    }).sort_values('importance', ascending=False).head(top_n)
    
    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                title='Top Feature Importance')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_business_impact(results):
    """Plot business impact metrics"""
    metrics = ['overstock_cost', 'understock_cost']
    values = [results['overstock_cost'], results['understock_cost']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics,
        y=values,
        text=[f'${x:,.0f}' for x in values],
        textposition='auto',
        marker_color=['#EF553B', '#636EFA']
    ))
    
    fig.update_layout(
        title='Inventory Cost Impact',
        yaxis_title='Cost ($)',
        xaxis_title='Metric'
    )
    return fig

def plot_prediction_vs_actual(predictions, actuals):
    """Plot predictions vs actual values"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actuals,
        y=predictions,
        mode='markers',
        name='Predictions vs Actual'
    ))
    
    # Add perfect prediction line
    max_val = max(max(predictions), max(actuals))
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Predictions vs Actual Demand',
        xaxis_title='Actual Demand',
        yaxis_title='Predicted Demand',
        showlegend=True
    )
    return fig