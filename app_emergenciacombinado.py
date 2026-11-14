# -*- coding: utf-8 -*-
"""
PREDWEEM Lolium — Pergamino.

La página muestra la alerta de grandes flujos de Lolium del paquete ``lolium-comun`` (probabilidad de gran flujo en
los próximos 7 días, qué hacer esta semana, ventana del graminicida de barbecho, residual a la siembra y flujos tardíos)
con el clima diario del repositorio (``meteo_daily.csv``).

El motor PREDWEEM anterior (red neuronal y balance hídrico, ``app_emergenciacombinado_core.py`` y sus modelos) ya no se
ejecuta: su inicio estimado de la emergencia erró 12–14 días en promedio y no mejoraba la alerta. Sus archivos quedan
en el repositorio solo como referencia.
"""
import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

# Control de acceso (ingreso con Google y usuarios aprobados). Inactivo hasta configurar los secretos [acceso] y [auth].
from acceso_predweem import mostrar_sesion, requerir_acceso
from lolium_comun import pagina

BASE = Path(__file__).resolve().parent

requerir_acceso("Pergamino")
pagina.encabezado("🌾 PREDWEEM LOLIUM — PERGAMINO (BA) lat=-33.9443 lon=-60.5745", "PREDWEEM PERGAMINO")
parametros = pagina.barra_lateral(BASE, "Pergamino")
meteo = pagina.cargar_meteo(BASE, "https://raw.githubusercontent.com/PREDWEEM/LOLIUM-PERGA2026/main/meteo_daily.csv")

NOTA_PERGAMINO = (
    "**Inicio tardío y variable.** En Pergamino la emergencia empezó a fines de febrero en 2024, a principios de abril en "
    "2026 y a mediados de abril en 2023 (curvas del gemelo digital). Las lluvias de febrero y marzo no distinguen esos años "
    "(2024 y 2026 fueron lluviosos; 2023, muy seco), y el primer pico del motor PREDWEEM anterior adelantaba el inicio (05/03/2026). "
    "Por eso en Pergamino: (1) el reloj térmico y la ventana del graminicida de barbecho se calculan solo desde la fecha de primeras "
    "plántulas informada (o el primer conteo con plántulas); (2) mientras no se informen plántulas, la alerta llega como "
    "máximo a *Atención* («recorrer el lote para detectar el inicio»); (3) las semanas que un conteo posterior en cero "
    "(sin plántulas previas) confirma sin emergencia quedan en 0 %, y la curva acumulada arranca después "
    "de ese conteo; la semana que sigue al último conteo en cero se evalúa normalmente, porque el primer gran flujo suele venir "
    "justo después (Pergamino 2026: cero el 01/04, gran flujo al 13/04). **Informar la fecha de las primeras plántulas es lo "
    "que más mejora la alerta en Pergamino.**\n\n"
    "**Validación en Pergamino** (2024 y 2026, con clima SIGA de INTA Pergamino; 30 intervalos —incluidas las semanas de "
    "febrero y marzo de 2026, sin emergencia según el responsable del ensayo—, 11 con emergencia ≥ 5 % de la campaña por "
    "semana). AUC 0,67 solo con clima, 0,74 con la fecha de primeras plántulas y 0,81 con conteos: la alerta acierta una vez "
    "iniciada la emergencia, pero antes del inicio responde a las lluvias. Sin el tope daba *Alerta* en 9 de 19 intervalos "
    "sin gran flujo (7 de ellos en febrero–marzo de 2026); con el tope, 2 de 19, y los 11 grandes flujos quedan en *Atención* o "
    "*Alerta* (7 en *Alerta*: los primeros de cada año quedan en *Atención* porque todavía no había plántulas informadas). "
    "Con las mismas semanas, la red neuronal del gemelo dio AUC 0,66.\n\n"
    "**Curva acumulada.** Es el calendario de referencia regional (error de 8 puntos en 2024 y 27 en 2026). Un "
    "calendario propio armado con las otras campañas de Pergamino empeoró la alerta, porque los años difieren mucho.\n\n"
    "**Límites.** Dos campañas evaluables; 2023 y 2024 son curvas relativas digitalizadas de boletines de la Red DIMA "
    "(quincenales); el clima SIGA de 2023 tiene huecos en abril–agosto y no permite evaluarla."
)


_resultados_alerta = None
if meteo is not None and not meteo.empty:
    try:
        import alerta_lolmu as _alerta_lolmu
        import visualizacion_alerta_lolmu as _vis_alerta

        # Streamlit conserva en memoria los módulos ya importados entre ejecuciones: tras una actualización del
        # repositorio, recargarlos evita usar una versión vieja
        importlib.reload(_alerta_lolmu)
        importlib.reload(_vis_alerta)
        _resultados_alerta = _vis_alerta.mostrar_alerta_lolmu(
            meteo,
            **parametros,
    localidad="Pergamino",
    nota_sitio=NOTA_PERGAMINO,
    tope_sin_inicio=True,  # sin plántulas informadas, como máximo Atención (inicio tardío y variable)
    tope_hasta_ultimo_cero=True,  # semanas ya cerradas por un conteo posterior en cero: probabilidad 0
    curva_desde_ultimo_cero=True,  # la curva acumulada arranca a mitad de camino entre el último cero y el primer conteo con plántulas
    curva_tiempo_termico=True,  # con primeras plántulas, curva acumulada en tiempo térmico desde esa fecha
        )
    except Exception as _error_alerta:  # noqa: BLE001
        st.warning(f"No se pudo calcular la alerta: {_error_alerta}")

pagina.descarga_excel(_resultados_alerta, "Pergamino", "descarga_excel_resultados_final_pergamino")
mostrar_sesion()
