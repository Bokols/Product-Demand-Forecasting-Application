import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model_utils import load_model, make_predictions, calculate_business_impact, generate_recommendations
from utils.preprocessing import clean_column_names, add_product_names
from datetime import datetime, timedelta
from pathlib import Path

# Configuración de página
st.set_page_config(page_title="Pronosticador de Demanda", page_icon="🔮", layout="wide")
st.title("🔮 Pronosticador de Demanda")
st.markdown("""
Genera pronósticos de demanda y analiza el impacto empresarial.  
Ajusta los parámetros en la barra lateral y haz clic en **Generar Pronóstico** para ver predicciones.
""")

# Explicación introductoria
with st.expander("ℹ️ Acerca de esta herramienta", expanded=False):
    st.markdown("""
    Este pronosticador usa aprendizaje automático para predecir la demanda basado en:
    - Patrones históricos de ventas
    - Niveles actuales de inventario
    - Datos de precios y promociones
    - Factores externos como estacionalidad y clima
    
    **Características principales**:
    - Planificación de escenarios con parámetros ajustables
    - Recomendaciones para optimizar inventario
    - Proyecciones de impacto empresarial
    - Datos exportables del pronóstico
    """)

@st.cache_resource
def load_forecast_model():
    """Carga el modelo entrenado"""
    model_path = Path('model') / 'best_lightgbm_pipeline.pkl'
    try:
        pipeline = load_model(model_path)
        st.session_state['model_features'] = pipeline.n_features_in_
        return pipeline
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.stop()

@st.cache_data
def load_historical_data():
    """Carga datos históricos"""
    data_path = Path('data') / 'retail_store_inventory.csv'
    try:
        df = pd.read_csv(data_path)
        df = clean_column_names(df)
        df = add_product_names(df)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {str(e)}")
        st.stop()

def generate_prediction_input(historical_df, future_dates):
    """Genera dataframe de entrada con todas las características requeridas"""
    try:
        # Usa los datos más recientes como plantilla
        template = historical_df[historical_df['date'] == historical_df['date'].max()].copy()
        
        # Crea fechas futuras
        prediction_data = []
        for date in future_dates:
            temp = template.copy()
            temp['date'] = pd.to_datetime(date)
            prediction_data.append(temp)
        
        prediction_df = pd.concat(prediction_data)
        
        # Asegura que existan todas las características requeridas
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
        
        # Añade características de fecha
        date_features = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'quarter']
        for feat in date_features:
            prediction_df[feat] = getattr(prediction_df['date'].dt, feat.lower())
        
        prediction_df['week_of_year'] = prediction_df['date'].dt.isocalendar().week
        
        # Añade características derivadas
        min_date = historical_df['date'].min()
        prediction_df['days_since_start'] = (prediction_df['date'] - min_date).dt.days
        prediction_df['price_discount_ratio'] = prediction_df['discount'] / (prediction_df['price'] + 1e-10)
        prediction_df['price_competitor_diff'] = prediction_df['price'] - prediction_df['competitor_pricing']
        prediction_df['inventory_turnover'] = prediction_df['units_sold'] / (prediction_df['inventory_level'] + 1e-10)
        
        return prediction_df
    except Exception as e:
        st.error(f"❌ Error generando datos de entrada: {str(e)}")
        st.stop()

# Carga modelo y datos
model = load_forecast_model()
historical_df = load_historical_data()

# Controles de la barra lateral con tooltips
st.sidebar.header("Parámetros de Predicción")

# Horizonte de pronóstico con explicación
horizon = st.sidebar.selectbox(
    "Horizonte de Pronóstico",
    options=["Próximos 7 días", "Próximos 14 días", "Próximos 30 días"],
    index=0,
    help="Selecciona cuántos días en el futuro quieres predecir. El modelo genera predicciones diarias de demanda."
)

n_days = 7 if "7" in horizon else 14 if "14" in horizon else 30

# Genera fechas futuras desde hoy
today = datetime.now().date()
future_dates = [today + timedelta(days=i) for i in range(1, n_days+1)]
prediction_input = generate_prediction_input(historical_df, future_dates)

# Selección de productos con explicación
selected_products = st.sidebar.multiselect(
    "Productos a Pronosticar",
    options=historical_df['product_name'].unique(),
    default=historical_df['product_name'].unique()[:3],
    help="Selecciona productos específicos para analizar. Deja en blanco para incluir todos."
)

if selected_products:
    prediction_input = prediction_input[prediction_input['product_name'].isin(selected_products)]

# Sección de parámetros empresariales con explicaciones
st.sidebar.subheader("Escenarios Empresariales")
st.sidebar.markdown("Ajusta estos parámetros para simular diferentes condiciones:")

price_adjustment = st.sidebar.slider(
    "Ajuste de Precio (%)", 
    -50, 50, 0, 5,
    help="Simula cambios de precio. Valores positivos = aumentos, negativos = descuentos. Afecta la elasticidad de la demanda."
)

discount_adjustment = st.sidebar.slider(
    "Ajuste de Descuento (%)", 
    -100, 100, 0, 5,
    help="Cambia niveles de descuento. Valores positivos = más promociones, negativos = menos. Máximo 100% de descuento."
)

comp_adjustment = st.sidebar.slider(
    "Ajuste de Precio Competencia (%)", 
    -50, 50, 0, 5,
    help="Simula cambios de precios de competidores. Valores positivos = aumentan precios, negativos = descuentos."
)

holiday_promotion = st.sidebar.checkbox(
    "Período de Promoción/Festivo", 
    False,
    help="Marca esto para considerar aumentos estacionales de demanda durante festivos o promociones."
)

# Aplica ajustes
prediction_input['price'] *= (1 + price_adjustment/100)
prediction_input['discount'] = np.clip(prediction_input['discount'] * (1 + discount_adjustment/100), 0, 100)
prediction_input['competitor_pricing'] *= (1 + comp_adjustment/100)
prediction_input['holiday_promotion'] = 'si' if holiday_promotion else 'no'

# Botón para generar predicciones con explicación de estado
if st.sidebar.button("Generar Pronóstico", type="primary"):
    with st.spinner("Generando predicciones... Esto puede tomar un momento para conjuntos grandes."):
        try:
            # Valida que existan todas las columnas requeridas
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
                st.error(f"❌ Columnas requeridas faltantes: {missing_columns}")
                st.stop()
            
            predictions = make_predictions(model, prediction_input)
            prediction_input['predicted_demand'] = predictions
            
            # Calcula impacto empresarial
            business_impact = calculate_business_impact(
                predictions,
                price_data=prediction_input['price'].values,
                inventory_data=prediction_input['inventory_level'].values
            )
            
            # Genera recomendaciones
            recommendations = generate_recommendations(predictions, prediction_input)
            
            # Almacena resultados
            st.session_state['predictions'] = prediction_input
            st.session_state['business_impact'] = business_impact
            st.session_state['recommendations'] = recommendations
            
            st.success("✅ ¡Pronóstico generado exitosamente!")
        except Exception as e:
            st.error(f"❌ Error en la predicción: {str(e)}")

# Muestra resultados con explicaciones mejoradas
if 'predictions' in st.session_state:
    predictions_df = st.session_state['predictions']
    business_impact = st.session_state['business_impact']
    recommendations = st.session_state['recommendations']
    
    # Métricas con explicaciones
    st.header("Resumen del Pronóstico")
    with st.expander("Entendiendo estas métricas"):
        st.markdown("""
        - **Demanda Total Pronosticada**: Suma de todas las unidades esperadas a vender
        - **Demanda Diaria Promedio**: Media de unidades vendidas por día
        - **Ingresos Proyectados**: Ingresos estimados (precio × demanda pronosticada)
        - **Costo Total de Inventario**: Costos por exceso + ventas perdidas por faltante
        """)
    
    cols = st.columns(4)
    cols[0].metric("Demanda Total Pronosticada", f"{business_impact['total_predicted']:,.0f} unidades")
    cols[1].metric("Demanda Diaria Promedio", f"{business_impact['mean_demand']:,.1f} unidades")
    cols[2].metric("Ingresos Proyectados", f"${business_impact.get('projected_revenue', 0):,.0f}")
    cols[3].metric("Costo Total de Inventario", f"${business_impact.get('total_cost', 0):,.0f}")
    
    # Visualización con explicación
    st.header("Pronóstico de Demanda")
    st.markdown(f"""
    *Demanda diaria pronosticada para los productos seleccionados en {horizon.lower()}*
    """)
    fig = px.line(
        predictions_df.groupby(['date', 'product_name'])['predicted_demand'].sum().reset_index(),
        x='date', y='predicted_demand', color='product_name',
        title=f"Pronóstico de Demanda {horizon}",
        labels={'predicted_demand': 'Demanda Pronosticada (unidades)', 'date': 'Fecha'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Impacto Empresarial con explicaciones detalladas
    st.header("Análisis de Inventario")
    with st.expander("Cómo interpretar el análisis de inventario"):
        st.markdown("""
        - **Unidades en Exceso**: Ítems que probablemente no se venderán (inventario > demanda)
        - **Unidades en Faltante**: Ventas potencialmente perdidas (demanda > inventario)
        - El inventario óptimo coincide exactamente con la demanda pronosticada
        """)
    
    col1, col2 = st.columns(2)
    col1.metric("Unidades Potenciales en Exceso", 
               f"{business_impact.get('overstock_units', 0):,.0f}",
               help="Unidades que podrían no venderse según el inventario actual")
    col2.metric("Unidades Potenciales en Faltante", 
               f"{business_impact.get('understock_units', 0):,.0f}",
               help="Ventas potencialmente perdidas por inventario insuficiente")
    
    # Recomendaciones con indicadores de prioridad
    st.header("Recomendaciones de Inventario")
    st.markdown("""
    *Los colores indican la prioridad de la recomendación*  
    🟥 **Alta prioridad** | 🟨 **Prioridad media** | 🟩 **Baja prioridad**
    """)
    
    for rec in recommendations:
        if "Aumentar" in rec and "!" in rec:
            st.error(rec)
        elif "Aumentar" in rec:
            st.warning(rec)
        elif "Reducir" in rec and "!" in rec:
            st.error(rec)
        elif "Reducir" in rec:
            st.warning(rec)
        else:
            st.success(rec)
    
    # Exportación de datos con opciones de formato
    st.header("Exportar Resultados")
    with st.expander("Opciones de exportación"):
        st.markdown("""
        Descarga los datos del pronóstico en tu formato preferido:
        - CSV para hojas de cálculo
        - JSON para APIs y desarrolladores
        - Excel para análisis detallados
        """)
    
    export_format = st.selectbox("Formato de exportación", ["CSV", "Excel", "JSON"])
    
    if st.button(f"Descargar Datos como {export_format}"):
        if export_format == "CSV":
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name=f"pronostico_demanda_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        elif export_format == "Excel":
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                predictions_df.to_excel(writer, index=False)
            st.download_button(
                label="Descargar Excel",
                data=output.getvalue(),
                file_name=f"pronostico_demanda_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime='application/vnd.ms-excel'
            )
        elif export_format == "JSON":
            json = predictions_df.to_json(orient='records')
            st.download_button(
                label="Descargar JSON",
                data=json,
                file_name=f"pronostico_demanda_{datetime.now().strftime('%Y%m%d')}.json",
                mime='application/json'
            )
else:
    st.info("""
    ℹ️ Aún no se ha generado un pronóstico.  
    Ajusta los parámetros en la barra lateral y haz clic en **Generar Pronóstico** para ver predicciones.
    """)

# Pie de página con información del modelo
st.markdown("---")
st.caption(f"""
Modelo entrenado por última vez: {datetime.fromtimestamp(Path('model/best_lightgbm_pipeline.pkl').stat().st_mtime).strftime('%Y-%m-%d')}  
Usando algoritmo LightGBM con {model.n_features_in_} características
""")