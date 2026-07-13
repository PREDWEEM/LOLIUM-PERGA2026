# Optimizador ecofisiológico PREDWEEM

## Situación actual: un solo conjunto de campo

El repositorio dispone actualmente de `VALIDA.xlsx` como único conjunto de
observaciones de emergencia. Por ese motivo, la página del optimizador utiliza
por defecto **validación cruzada temporal interna por bloques contiguos**.

Este diseño:

1. transforma cada conteo en un intervalo explícito entre dos fechas;
2. divide la secuencia cronológica en bloques;
3. evalúa cada combinación paramétrica en cada bloque retenido;
4. selecciona la combinación con mejor score medio, penalizando la variabilidad
   entre bloques;
5. vuelve a calcular un ajuste descriptivo sobre toda la serie.

El score de validación cruzada es una estimación interna. No debe denominarse
“validación independiente”, porque todos los bloques pertenecen al mismo
experimento, localidad y campaña.

## Uso

Ejecutar la aplicación Streamlit principal. La página:

`pages/05_Optimizador_Ecofisiologico.py`

aparece automáticamente en el menú multipágina.

En el modo predeterminado:

- meteorología: `meteo_daily.csv`;
- campo: `VALIDA.xlsx`;
- diseño: validación cruzada temporal interna;
- bloques solicitados: 3;
- mínimo: 2 intervalos por bloque.

El motor de validación independiente permanece disponible en el paquete para
cuando se incorpore otra campaña, localidad o experimento.

## Variables explorables

- capacidad superficial `w_max`;
- coeficiente de evaporación `ke_suelo`;
- umbral y pendiente hídrica;
- corte y recarga de humedad;
- modulador térmico del suelo;
- duración de latencia;
- ventana y umbral de termoinhibición;
- ventana, umbral, final e intensidad del choque hídrico;
- umbral y persistencia del primer pico;
- lag de emergencia.

## Métricas

- KGE y NSE de flujos por intervalo;
- CCC y RMSE acumulados;
- F1 de detección de eventos;
- desfase T50;
- score robusto medio menos penalización por variabilidad entre bloques.

El T50 no participa en la selección por bloques, porque el T50 calculado dentro
de un bloque no representa el T50 global de la campaña. Sí se informa en el
ajuste descriptivo completo.

## Interpretación científica

Con `VALIDA.xlsx` pueden obtenerse parámetros provisionales y analizarse
sensibilidad e inestabilidad temporal. La validación externa definitiva requiere
al menos otra campaña, localidad o experimento que no participe en la selección
de parámetros.
