import streamlit as st
from PIL import Image
import os
from datetime import datetime
import requests
from io import BytesIO

# Configuración de página
st.set_page_config(
    page_title="Pronóstico de Demanda Minorista",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Barra lateral personalizada
with st.sidebar:
    # Sección de perfil
    st.header("Bo Kolstrup")
    st.caption("Científico de Datos | Especialista en Marketing Digital | Python")
    
    # Cargar imagen de perfil local
    try:
        profile_img = Image.open("assets/Bo-Kolstrup.png")
        st.image(profile_img, width=200, use_column_width='auto')
    except Exception as e:
        st.warning(f"No se pudo cargar la imagen de perfil: {str(e)}")
        profile_img = None

    # Sección "Acerca de"
    st.markdown("""
    ### Sobre Mí
    Apasionado por aplicar ciencia de datos y aprendizaje automático para resolver desafíos empresariales reales.
    """)
    
    # Enlaces sociales con íconos
    st.markdown("""
    ### Conéctate Conmigo
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
    
    # Navegación actualizada con los nombres solicitados
    st.subheader("Navegación")
    
    # Botón de Inicio - permanece en la página actual cuando ya está en inicio
    if st.button("🏠 Inicio", 
                help="Volver al panel principal",
                disabled=st.session_state.get('current_page') == 'home',
                use_container_width=True):
        st.session_state.current_page = 'home'
        st.switch_page("app.py")
    
    # Botón de Centro de Análisis
    if st.button("📊 Centro de Análisis", 
                help="Explora datos históricos y tendencias",
                disabled=st.session_state.get('current_page') == 'explorer',
                use_container_width=True):
        st.session_state.current_page = 'explorer'
        st.switch_page("pages/1_📊_Explorer.py")
    
    # Botón de Laboratorio de Pronósticos
    if st.button("🔮 Laboratorio de IA", 
                help="Genera predicciones y recomendaciones",
                disabled=st.session_state.get('current_page') == 'predict',
                use_container_width=True):
        st.session_state.current_page = 'predict'
        st.switch_page("pages/2_🔮_Predict.py")

# Inicializar estado de sesión para seguimiento de página
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Contenido principal
st.title("🛒 Pronóstico de Demanda Minorista")
st.markdown("*Optimización de inventario con IA para minoristas modernos*")

# Resumen del proyecto
with st.expander("📌 Acerca de Este Proyecto", expanded=True):
    st.markdown("""
    ## 🔍 ¿Qué es esta herramienta?
    
    Un **panel impulsado por aprendizaje automático** que ayuda a minoristas:
    - 📈 Predecir demanda de productos con 85%+ de precisión
    - 📊 Analizar patrones históricos de ventas
    - 🛍️ Optimizar niveles de inventario
    - 💰 Reducir costos por exceso o faltante de stock

    ## 🧠 Cómo Funciona
    
    1. **Integración de Datos**:  
       - Procesa datos históricos de ventas, inventario y precios
       - Incorpora factores externos (clima, promociones, etc.)
    
    2. **Pronóstico con IA**:  
       - Utiliza modelo LightGBM de aprendizaje automático
       - Entrenado con 2+ años de datos minoristas
       - Actualiza predicciones diariamente
    
    3. **Información Accionable**:  
       - Recomendaciones automáticas de inventario
       - Herramientas de planificación de escenarios
       - Proyecciones de impacto financiero
    """)

# Resumen de características
st.markdown("---")
st.header("Características Principales")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Centro de Análisis")
    st.markdown("""
    - Visualización interactiva de datos
    - Análisis de tendencias de demanda
    - Modelado de elasticidad de precios
    - Detección de patrones estacionales
    - Métricas de salud de inventario
    """)

with col2:
    st.subheader("🔮 Laboratorio de IA")
    st.markdown("""
    - Pronóstico de demanda a 30 días
    - Pruebas de escenarios hipotéticos
    - Recomendaciones automatizadas
    - Análisis de impacto empresarial
    - Reportes exportables
    """)

# Pie de página
st.markdown("---")
st.caption(f"""
🚀 Desarrollado por Bo Kolstrup | 📅 Última actualización: {datetime.now().strftime("%Y-%m-%d")}  
[Repositorio GitHub](https://github.com/Bokols) | [Contacto](mailto:bokolstrup@gmail.com)
""")