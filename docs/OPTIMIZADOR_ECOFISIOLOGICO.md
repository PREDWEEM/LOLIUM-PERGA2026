# Optimizador ecofisiológico PREDWEEM

## Objetivo

Calibrar los filtros ecofisiológicos del modelo de emergencia con un conjunto de campo y medir su capacidad de generalización con un segundo conjunto independiente. El archivo independiente no interviene en la selección de parámetros.

## Integración

La página `pages/05_Optimizador_Ecofisiologico.py` aparece automáticamente en la navegación multipágina cuando se ejecuta `app_emergenciacombinado.py` con Streamlit.

El motor reutilizable se encuentra en el paquete `predweem_optimizer/` y no depende de Streamlit.

## Datos requeridos

- Meteorología diaria: `Fecha`, `TMAX`, `TMIN`, `Prec` o sinónimos reconocibles.
- Campo de calibración: fecha y una variable numérica de emergencia.
- Campo de validación independiente: otro experimento, campaña o localidad no usado para calibrar.
- Opcionalmente, una columna `Grupo`, `Sitio`, `Localidad`, `Campaña` o `Año` permite evaluar múltiples unidades y penalizar soluciones inestables.

El campo puede cargarse como flujo por intervalo o como conteo acumulado.

## Método

1. Muestreo global estratificado del espacio paramétrico.
2. Refinamiento local alrededor de las mejores soluciones.
3. Score multicriterio con KGE, NSE, CCC, F1, RMSE acumulado y desfase T50.
4. Penalización por variabilidad entre grupos de calibración.
5. Selección final únicamente con calibración.
6. Evaluación posterior en el conjunto independiente.

## Variables disponibles

Incluye capacidad hídrica superficial, Ke, centro y pendiente del filtro hídrico, corte por humedad, recarga, modulador térmico, latencia, ventana y umbral de termoinhibición, choque hídrico, primer pico y lag temporal.

## Correcciones respecto del calibrador 2D anterior

- El lag temporal se incluye dentro de cada simulación candidata.
- El umbral y la ventana térmica dejan de estar fijados en valores distintos a la interfaz principal.
- El modulador térmico y el balance hídrico se evalúan en el mismo motor.
- El denominador de la curva simulada se trunca en la última fecha de campo.
- El primer conteo de campo se conserva como primer intervalo, en lugar de descartarse.
- Se impide usar el mismo archivo para calibración y validación.
- La recarga hídrica se habilita por el estado de humedad alcanzado, permitiendo que lluvias sucesivas moderadas recarguen el estrato superficial.

## Salidas

- Parámetros óptimos en tabla y JSON.
- Ranking de candidatos.
- Métricas por grupo para calibración y validación independiente.
- Curvas acumuladas y gráfico 1:1.
- Sensibilidad aproximada por correlación de Spearman.
- Informe Excel completo.
