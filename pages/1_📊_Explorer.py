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

# Configuración de la página
st.set_page_config(page_title="Explorador de Datos", page_icon="📊", layout="wide")
st.title("📊 Explorador de Datos")
st.markdown("""
Explora el conjunto de datos de pronóstico de demanda minorista y analiza tendencias.  
Utiliza los filtros de la barra lateral para enfocarte en productos, períodos o regiones específicas.
""")

# Guía introductoria
with st.expander("ℹ️ Cómo Usar Este Explorador", expanded=False):
    st.markdown("""
    **Guía del Explorador de Datos**:
    
    1. **Filtra datos** usando los controles de la barra lateral
    2. **Visualiza métricas** en la sección de resumen
    3. **Analiza patrones** en los gráficos interactivos
    4. **Exporta insights** usando las opciones de descarga
    
    **Características Principales**:
    - Análisis de tendencias de demanda
    - Visualización de elasticidad de precios
    - Insights para optimización de inventario
    - Cálculos de impacto empresarial
    """)

@st.cache_data
def load_data():
    """Carga y preprocesa los datos minoristas"""
    data_path = os.path.join('data', 'retail_store_inventory.csv')
    df = pd.read_csv(data_path)
    df = clean_column_names(df)
    df = add_product_names(df)
    df['date'] = pd.to_datetime(df['date'])
    
    # Extrae componentes de fecha con explicaciones
    date_features = {
        'year': 'Año extraído de la fecha',
        'month': 'Mes (1-12)',
        'day': 'Día del mes',
        'day_of_week': 'Día de la semana (0=Lunes)',
        'day_of_year': 'Día del año (1-365)',
        'week_of_year': 'Número de semana ISO',
        'quarter': 'Trimestre (1-4)'
    }
    
    for feat, desc in date_features.items():
        if feat == 'week_of_year':
            df['week_of_year'] = df['date'].dt.isocalendar().week
        elif feat == 'quarter':
            df['quarter'] = df['date'].dt.quarter
        else:
            df[feat] = getattr(df['date'].dt, feat)
    
    return df

# Carga datos con estado
with st.spinner("Cargando datos..."):
    df = load_data()

# Filtros de la barra lateral con tooltips
st.sidebar.header("Filtros de Datos")
st.sidebar.markdown("Usa estos controles para enfocarte en subconjuntos específicos:")

# Filtro de rango de fechas con explicación
min_date = df['date'].min().date()
max_date = df['date'].max().date()

date_range = st.sidebar.date_input(
    "Rango de Fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Selecciona fechas de inicio y fin para analizar un período específico"
)

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    st.sidebar.caption(f"Mostrando datos desde {start_date} hasta {end_date}")
else:
    st.sidebar.warning("Por favor selecciona ambas fechas de inicio y fin")

# Filtro de productos con capacidad de búsqueda
products = st.sidebar.multiselect(
    "Productos",
    options=df['product_name'].unique(),
    default=df['product_name'].unique()[:3],
    help="Selecciona productos específicos o deja en blanco para todos"
)
if products:
    df = df[df['product_name'].isin(products)]

# Filtro de categorías con explicación
categories = st.sidebar.multiselect(
    "Categorías",
    options=df['category'].unique(),
    default=df['category'].unique(),
    help="Filtrar por categoría de producto"
)
if categories:
    df = df[df['category'].isin(categories)]

# Filtro de regiones con explicación
regions = st.sidebar.multiselect(
    "Regiones",
    options=df['region'].unique(),
    default=df['region'].unique(),
    help="Enfócate en regiones geográficas específicas"
)
if regions:
    df = df[df['region'].isin(regions)]

# Filtro de tiendas con ayuda de búsqueda
stores = st.sidebar.multiselect(
    "Tiendas",
    options=df['store_id'].unique(),
    default=df['store_id'].unique()[:3],
    help="Selecciona ubicaciones de tiendas específicas"
)
if stores:
    df = df[df['store_id'].isin(stores)]

# Contenido principal
st.header("Resumen del Conjunto de Datos")
st.write(f"Mostrando {len(df):,} filas del conjunto de datos filtrado")

# Alternar datos crudos con advertencia
if st.checkbox("Mostrar datos crudos", help="Ver el conjunto de datos subyacente (puede ser grande)"):
    st.dataframe(df)
    st.download_button(
        label="Descargar Datos Filtrados",
        data=df.to_csv(index=False),
        file_name="datos_minoristas_filtrados.csv",
        mime="text/csv"
    )

# Métricas clave con explicaciones
st.subheader("Métricas Clave")
with st.expander("Acerca de estas métricas"):
    st.markdown("""
    - **Unidades Totales Vendidas**: Suma de todos los productos vendidos en el conjunto filtrado
    - **Demanda Diaria Promedio**: Media de unidades vendidas por día
    - **Valor Total de Inventario**: Stock actual × precio (al momento de registro)
    - **Descuento Promedio**: Porcentaje medio de descuento promocional
    """)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Unidades Totales Vendidas", f"{df['units_sold'].sum():,.0f}")
col2.metric("Demanda Diaria Promedio", f"{df['units_sold'].mean():,.1f}")
col3.metric("Valor Total de Inventario", f"${(df['inventory_level'] * df['price']).sum():,.0f}")
col4.metric("Descuento Promedio", f"{df['discount'].mean():.1f}%")

# Sección de análisis de demanda
st.header("Análisis de Demanda")
st.markdown("Explora patrones de demanda usando las pestañas interactivas:")

tab1, tab2, tab3 = st.tabs(["Tendencias", "Distribución", "Estacionalidad"])

with tab1:
    st.subheader("Tendencias de Demanda en el Tiempo")
    st.markdown("*Cómo cambia la demanda en el período seleccionado*")
    
    groupby_trend = st.selectbox(
        "Agrupar por",
        options=[None, 'product_name', 'category', 'region', 'store_id'],
        index=0,
        help="Desglosa tendencias por dimensiones específicas"
    )
    fig_trend = plot_demand_trend(df, groupby=groupby_trend)
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.subheader("Distribución de Demanda")
    st.markdown("*Cómo se distribuye la demanda entre diferentes factores*")
    
    groupby_dist = st.selectbox(
        "Agrupar distribución por",
        options=[None, 'product_name', 'category', 'region', 'store_id', 'seasonality', 'weather_condition'],
        index=0,
        help="Compara distribuciones de demanda entre categorías"
    )
    fig_dist = plot_demand_distribution(df, groupby=groupby_dist)
    st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    st.subheader("Patrones Estacionales")
    st.markdown("*Patrones recurrentes de demanda por períodos*")
    
    # Estacionalidad mensual
    if 'month' in df.columns:
        st.markdown("**Patrones Mensuales**")
        monthly_agg = df.groupby(['month', 'product_name'])['units_sold'].sum().reset_index()
        fig_monthly = px.line(monthly_agg, x='month', y='units_sold', color='product_name',
                             title='Demanda Mensual por Producto',
                             labels={'month': 'Mes', 'units_sold': 'Unidades Vendidas'})
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Estacionalidad por día de semana
    if 'day_of_week' in df.columns:
        st.markdown("**Patrones por Día de Semana**")
        dow_agg = df.groupby(['day_of_week', 'product_name'])['units_sold'].sum().reset_index()
        fig_dow = px.line(dow_agg, x='day_of_week', y='units_sold', color='product_name',
                         title='Patrones de Demanda Semanal',
                         labels={'day_of_week': 'Día de Semana (0=Lunes)', 'units_sold': 'Unidades Vendidas'})
        st.plotly_chart(fig_dow, use_container_width=True)

# Análisis de precios y descuentos con explicaciones
st.header("Análisis de Sensibilidad a Precios")
st.markdown("""
Examina cómo los precios y promociones afectan la demanda:  
- **Elasticidad de Precio**: Cómo cambia la demanda con el precio
- **Impacto de Descuentos**: Efectividad de las promociones
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Precio vs Demanda")
    st.markdown("*Relación entre precio y unidades vendidas*")
    fig_price = px.scatter(df, x='price', y='units_sold', color='category',
                          trendline="lowess",
                          title="Elasticidad-Precio de la Demanda",
                          labels={'price': 'Precio ($)', 'units_sold': 'Unidades Vendidas'})
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    st.subheader("Impacto de Descuentos")
    st.markdown("*Cómo los descuentos afectan el volumen de ventas*")
    fig_discount = px.box(df, x='discount', y='units_sold', color='category',
                         title="Distribución de Demanda por Nivel de Descuento",
                         labels={'discount': 'Descuento (%)', 'units_sold': 'Unidades Vendidas'})
    st.plotly_chart(fig_discount, use_container_width=True)

# Análisis de inventario con contexto empresarial
st.header("Optimización de Inventario")
st.markdown("""
Métricas clave para gestión de inventario:  
- **Rotación**: Qué tan rápido se vende el inventario  
- **Riesgo de Agotamiento**: Probabilidad de quedarse sin stock  
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Rotación de Inventario")
    st.markdown("*Ventas relativas a los niveles de inventario*")
    df['inventory_turnover'] = df['units_sold'] / df['inventory_level']
    fig_turnover = px.box(df, x='category', y='inventory_turnover',
                         title="Rotación de Inventario por Categoría",
                         labels={'category': 'Categoría', 'inventory_turnover': 'Ratio de Rotación'})
    st.plotly_chart(fig_turnover, use_container_width=True)

with col2:
    st.subheader("Riesgo de Agotamiento")
    st.markdown("*Probabilidad de agotar el inventario*")
    df['stockout_risk'] = (df['units_sold'] / df['inventory_level']).clip(upper=1)
    fig_stockout = px.box(df, x='category', y='stockout_risk',
                         title="Riesgo de Agotamiento por Categoría",
                         labels={'category': 'Categoría', 'stockout_risk': 'Probabilidad de Riesgo'})
    st.plotly_chart(fig_stockout, use_container_width=True)

# Análisis de impacto económico con explicaciones
st.header("Análisis de Impacto Empresarial")
st.markdown("""
Implicaciones financieras de decisiones de inventario:  
- **Pérdida de Ingresos**: Por oportunidades de venta perdidas  
- **Costos de Exceso**: Por mantener inventario excesivo  
""")

# Calcula pérdida potencial por agotamientos
df['potential_revenue_loss'] = np.where(
    df['units_sold'] > df['inventory_level'],
    (df['units_sold'] - df['inventory_level']) * df['price'],
    0
)

# Calcula costo de exceso
df['overstock_cost'] = np.where(
    df['inventory_level'] > df['units_sold'],
    (df['inventory_level'] - df['units_sold']) * df['price'] * 0.3,  # 30% costo de mantenimiento
    0
)

total_revenue_loss = df['potential_revenue_loss'].sum()
total_overstock_cost = df['overstock_cost'].sum()

col1, col2 = st.columns(2)
col1.metric("Pérdida Total Potencial por Agotamientos", 
           f"${total_revenue_loss:,.0f}",
           help="Ingresos perdidos por inventario insuficiente")
col2.metric("Costos Totales de Exceso", 
           f"${total_overstock_cost:,.0f}",
           help="Costos por inventario excesivo (30% costo de mantenimiento)")

# Tabla de productos problemáticos con contexto
st.subheader("Productos Prioritarios que Necesitan Ajuste")
st.markdown("""
Productos con mayores desajustes de inventario:  
🔴 = Alta prioridad | 🟡 = Prioridad media | 🔵 = Baja prioridad  
""")

worst_products = df.groupby('product_name').agg({
    'potential_revenue_loss': 'sum',
    'overstock_cost': 'sum'
}).sort_values('potential_revenue_loss', ascending=False).head(10)

# Añade indicadores de prioridad
def highlight_priority(row):
    if row['potential_revenue_loss'] > 10000:
        return ['background-color: #ffcccc'] * len(row)
    elif row['overstock_cost'] > 5000:
        return ['background-color: #ffffcc'] * len(row)
    return [''] * len(row)

st.dataframe(
    worst_products.style.format("${:,.0f}").apply(highlight_priority, axis=1)
)

# Sección de entrenamiento de modelo con advertencia
st.sidebar.header("Entrenamiento de Modelo")
st.sidebar.markdown("""
⚠️ Función Avanzada  
Usar solo si has actualizado significativamente el conjunto de datos
""")

if st.sidebar.button("Ajustar y Guardar Preprocesador"):
    with st.spinner("Ajustando y guardando preprocesador..."):
        try:
            from utils.preprocessing import preprocess_data
            preprocess_data(df, training=True)
            st.sidebar.success("¡Preprocesador ajustado y guardado exitosamente!")
        except Exception as e:
            st.sidebar.error(f"Error ajustando preprocesador: {str(e)}")

# Añade indicador de actualización de datos
st.markdown("---")
st.caption(f"Datos cargados por última vez: {datetime.now().strftime('%Y-%m-%d %H:%M')}")