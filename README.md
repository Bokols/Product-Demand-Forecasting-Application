# Aplicación de Pronóstico de Demanda de Productos

**Prueba el modelo [aquí](https://appuct-demand-forecasting-application-sr9xuniuxywt3wz5txuzrq.streamlit.app/):**

## Introducción

En la industria del retail, anticipar la demanda de productos es esencial para mantener niveles óptimos de inventario, reducir costos operacionales y mejorar la satisfacción del cliente. Este proyecto presenta una **Aplicación de Pronóstico de Demanda de Productos**, una solución de machine learning desarrollada para predecir la demanda diaria a corto plazo utilizando datos históricos de ventas y variables comerciales específicas.

La aplicación integra **preprocesamiento de datos, análisis exploratorio, ingeniería de características, entrenamiento de modelos** y **despliegue interactivo** mediante Streamlit. Su objetivo es ofrecer no solo pronósticos precisos, sino también herramientas de simulación y análisis para la toma de decisiones estratégicas.

## Descripción General del Proyecto

El proyecto se compone de tres partes principales:

1. **Notebook de Desarrollo del Modelo**  
   - Procesamiento de datos, análisis de demanda, ingeniería de variables y entrenamiento de un modelo LightGBM.

2. **Explorador de Datos Interactivo**
   - Aplicación en Streamlit para visualizar tendencias de ventas, analizar sensibilidad al precio y rendimiento del inventario.

3. **Herramienta de Pronóstico y Simulación**
   - Aplicación en Streamlit que permite simular escenarios comerciales, generar pronósticos diarios y recibir recomendaciones de inventario.
  


## Desarrollo del Modelo (Jupyter Notebook)

### 1. Preparación y Limpieza de Datos

- Carga de un dataset con ~100.000 transacciones de ventas.
- Estandarización de nombres de columnas y conversión de fechas.
- Extracción de variables temporales: año, mes, día, día de la semana, semana del año, trimestre.
- Mapeo de IDs de productos a nombres legibles.
- Tratamiento de valores faltantes.

### 2. Análisis Exploratorio (EDA)

Se analizaron patrones de comportamiento en los datos históricos de ventas.

**Hallazgos principales:**

- **Estacionalidad:** Se observaron ciclos claros de demanda mensual y semanal. La demanda aumenta significativamente en noviembre y diciembre, y también durante los fines de semana.
- **Elasticidad de precio:** Existe una correlación negativa entre el precio y las unidades vendidas, especialmente en productos no esenciales.
- **Promociones:** Descuentos superiores al 20% incrementan fuertemente la demanda, aunque se detectó un umbral de saturación cerca del 50%.
- **Variabilidad regional:** Algunas regiones presentaron baja rotación de inventario, lo que sugiere sobrestock o baja demanda local.
- **Desequilibrios de inventario:** Se identificaron faltantes frecuentes durante semanas pico y exceso de stock en períodos de baja rotación.

### 3. Ingeniería de Características

Se crearon variables nuevas para mejorar el rendimiento del modelo:

- `days_since_start`: Tendencia temporal.
- `price_discount_ratio`: Intensidad promocional.
- `price_competitor_diff`: Diferencia frente a precios de la competencia.
- `inventory_turnover`: Eficiencia de ventas respecto al inventario.

### 4. Entrenamiento del Modelo

- Se entrenó un modelo **LightGBM** con división temporal (entrenamiento = histórico, prueba = últimos 30 días).
- Métricas de rendimiento:
  - **MAE**: ~7,3 unidades/día
  - **RMSE**: ~12,6 unidades/día
- El modelo fue exportado como `best_lightgbm_pipeline.pkl` para su uso en producción.

## Explorador de Datos (Streamlit)

La aplicación permite explorar datos históricos con filtros dinámicos por producto, región, tienda y fecha.

**Características clave:**

- Visualización de tendencias por producto y categoría.
- Análisis de sensibilidad a precios y promociones.
- Indicadores de rotación de inventario y riesgo de quiebre de stock.
- Cálculo del impacto económico: pérdida por faltantes y costos por sobrestock.
- Tabla de productos críticos con prioridades visuales.

## Herramienta de Pronóstico y Simulación

La segunda aplicación permite simular condiciones de negocio y predecir la demanda futura.

**Funciones principales:**

- Pronóstico de 7, 14 o 30 días.
- Simulación de cambios en precio, descuentos y competencia.
- Ajuste de parámetros para promociones u ofertas especiales.
- Generación automática de recomendaciones de inventario.
- Exportación de resultados en CSV, Excel o JSON.

**Métricas generadas:**

- Demanda total proyectada
- Ingresos esperados
- Riesgos de sobrestock y faltantes
- Costos totales de inventario

## Resultados y Hallazgos

### Modelado y Predicción

- El modelo capturó patrones estacionales y efectos de negocio de forma precisa.
- Los errores de pronóstico se mantuvieron dentro del ±10% en datos no vistos.
- Las simulaciones permitieron ajustar decisiones sin necesidad de reentrenar el modelo.

### Conclusiones de Negocio

- Los **faltantes durante promociones** generaron pérdidas estimadas de más de **$30.000 USD** en un solo mes.
- El **sobrestock en productos de baja rotación** tuvo costos mensuales superiores a **$12.000 USD** en algunas regiones.
- Pequeños descuentos (10–20%) fueron efectivos sin afectar demasiado el margen.
- La **rotación de inventario** fue notablemente superior en ciertas categorías, lo que ofrece oportunidades de mejora logística.

## Conclusión

Este proyecto logró desarrollar una solución integral y escalable para la predicción de demanda en el sector retail. La combinación de un modelo robusto, simulación de escenarios y visualización de datos brinda a los usuarios herramientas valiosas para:

- Anticipar la demanda futura
- Optimizar los niveles de inventario
- Minimizar pérdidas por quiebres o sobrestock
- Tomar decisiones informadas basadas en datos

**Líneas futuras de desarrollo** incluyen:

- Integración con APIs en tiempo real (clima, ventas)
- Reentrenamiento automatizado del modelo
- Visualización de explicaciones del modelo (SHAP)
- Conexión con sistemas ERP o de gestión de inventario

**Prueba el modelo [aquí](https://appuct-demand-forecasting-application-sr9xuniuxywt3wz5txuzrq.streamlit.app/):**
