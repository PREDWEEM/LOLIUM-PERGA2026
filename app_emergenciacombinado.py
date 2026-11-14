# -*- coding: utf-8 -*-
"""
Punto de entrada privado de PREDWEEM Pergamino.

La aplicación científica se conserva en ``app_emergenciacombinado_core.py``. Antes de ejecutarla:

* ``build_private_core_source`` la prepara para el checkout privado (recursos locales y activos reales);
* se retiran de la página los elementos que reemplaza la alerta de flushes del final: mapa de riesgo, las pestañas
  (Monitor de decisión —con la dinámica principal, los mensajes de inicio y control y el indicador de tiempo
  térmico—, Precipitaciones y suelo, Análisis estratégico y Bio-calibración), la leyenda de versión, la sección
  "Datos del lote" (se usa el clima SIGA del repositorio y la cobertura de rastrojo por defecto, 80 %), la imagen de
  fondo (reemplazada por un fondo liso claro) y, en la barra lateral, los parámetros del motor (quedan en sus valores
  por defecto: termoinhibición 26 °C), sus avisos, el calibrador 2D y el botón del reporte del motor; la barra lateral
  conserva los parámetros de la ventana de control.

La alerta (``alerta_lolmu.py``) es la del paquete lolium-comun, validada con Pergamino 2024 y 2026.
En Pergamino el inicio de la emergencia es tardío y variable (fines de febrero en 2024; abril en 2026; mayo en 2023)
y el primer pico del motor lo adelanta (05/03/2026, cuando no hubo emergencia hasta abril). Por eso, a diferencia de
las otras localidades, el reloj térmico y la ventana de control **no** se anclan al primer pico del motor: se
calculan desde la fecha de las primeras plántulas que informa el usuario (o desde el primer conteo con plántulas), y,
mientras no se informen plántulas, la alerta llega como máximo a *Atención*.
"""
from pathlib import Path

from private_runtime import build_private_core_source
from visualizacion_horizonte_pronostico import mostrar_horizonte_pronostico

# Control de acceso (ingreso con Google y usuarios aprobados). Inactivo hasta configurar los secretos [acceso] y [auth].
import streamlit as st
from acceso_predweem import mostrar_sesion, requerir_acceso

requerir_acceso("Pergamino")

_CORE_APP = Path(__file__).with_name("app_emergenciacombinado_core.py")
_core_source = build_private_core_source(_CORE_APP)


def _reemplazar(origen: str, viejo: str, nuevo: str) -> str:
    if viejo not in origen:
        raise RuntimeError(f"No se encontró el bloque a reemplazar en el motor: {viejo[:70]!r}")
    return origen.replace(viejo, nuevo, 1)


def _cortar(origen: str, desde: str, hasta: str, nuevo: str) -> str:
    """Reemplaza el tramo [desde, hasta) del motor."""
    ini = origen.find(desde)
    fin = origen.find(hasta, ini) if ini >= 0 else -1
    if ini < 0 or fin < ini:
        raise RuntimeError(f"No se encontró el tramo a retirar del motor: {desde[:70]!r}")
    return origen[:ini] + nuevo + origen[fin:]


_FONDO_CSS = """<style>
.stApp { background: #f4f6f3; }
[data-testid="stAppViewContainer"] > .main, [data-testid="stMain"] { background: transparent; }
[data-testid="stSidebar"] { background-color: #e8f3ea; border-right: 1px solid #cfe3d3; }
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #14532d !important; }
[data-testid="stCaptionContainer"], .stCaption { color: #4b5563 !important; }
h1, h2, h3, h4 { color: #1f2937; }
</style>"""

# mapa de riesgo del motor (el cálculo no cambia)
_core_source = _reemplazar(_core_source, "st.plotly_chart(go.Figure(data=go.Heatmap(", "_omitir_grafico(go.Figure(data=go.Heatmap(")
# la regla "header {visibility: hidden;}" ocultaba también el encabezado de los calendarios desplegables
_core_source = _reemplazar(_core_source, "    header {visibility: hidden;}", '    header[data-testid="stHeader"] {visibility: hidden;}')
# fondo liso claro en lugar de la imagen de plantas (mejor contraste)
_core_source = _reemplazar(_core_source, 'set_bg_hack(str(BASE / "fondo_predweem_v3.png"))', "st.markdown(_FONDO_CSS, unsafe_allow_html=True)")
# leyenda de versión del motor
_core_source = _reemplazar(
    _core_source,
    'st.caption("Pergamino · VISUAL V3 · ventana eficiente verde · exportación PNG de alta resolución · lag fijo 22 días")',
    "pass",
)

# sección "Datos del lote": clima SIGA del repositorio, sin conteos del formato anterior, cobertura por defecto (80 %)
_core_source = _cortar(
    _core_source,
    'with st.expander("📂 1. Datos del Lote", expanded=True):',
    "# --- SIDEBAR ---",
    "archivo_meteo = None\narchivo_campo = None\nmodo_meteo = \"Automática\"\ncobertura_pct = 80\n"
    "ke_val, mod_termico = parametros_superficie(cobertura_pct)\n"
    'df_meteo_raw = load_data(None, "meteo_daily")\ndf_campo_raw = None\n\n',
)

# barra lateral: solo la ventana de control; los parámetros del motor quedan en sus valores por defecto
_core_source = _cortar(
    _core_source,
    'st.sidebar.markdown("## ⚙️ 2. Fisiología y Logística")',
    "# ---------------------------------------------------------\n# 6. MOTOR DE CÁLCULO",
    '''st.sidebar.markdown("## ⚙️ Ventana de control")
st.sidebar.caption("Tiempo térmico sobre la temperatura media del aire desde las primeras plántulas y objetivos de la ventana post-emergente.")
col_t1, col_t2 = st.sidebar.columns(2)
with col_t1: t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2: t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)
t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)
st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = st.sidebar.number_input("TT Control Post-emergente (°Cd)", value=600, step=10)
dga_critico = st.sidebar.number_input("Límite Ventana (°Cd)", value=800, step=10)

# parámetros del motor PREDWEEM en sus valores por defecto (controles retirados de la página)
umbral_er = 0.0001
umbral_termoinhibicion = 26.0
umbral_choque_hidrico = 45.0
residualidad = 0
w_max_val = 18.81
exponente_kr = 0.0

''',
)

# pestañas del motor (incluyen la dinámica principal, los mensajes de inicio y control y el indicador térmico)
_core_source = _cortar(_core_source, "    tab1, tab2, tab3, tab4 = st.tabs(", "    output = io.BytesIO()", "")
# botón lateral del reporte del motor: la descarga final reúne los resultados de la alerta
_core_source = _reemplazar(_core_source, 'st.sidebar.download_button("📥 Descargar Reporte Pergamino"', '_omitir_grafico("📥 Descargar Reporte Pergamino"')


def _omitir_grafico(*_args, **_kwargs):
    """No dibuja el elemento (se conservan la figura y todos los cálculos del motor)."""
    return None


exec(
    compile(_core_source, str(_CORE_APP), "exec"),
    globals(),
)

if "df" in globals() and isinstance(df, pd.DataFrame) and not df.empty:
    st.divider()
    mostrar_horizonte_pronostico(df, "Pergamino")

NOTA_PERGAMINO = (
    "**Inicio tardío y variable.** En Pergamino la emergencia empezó a fines de febrero en 2024, a principios de abril en "
    "2026 y a mediados de abril en 2023 (curvas del gemelo digital). Las lluvias de febrero y marzo no distinguen esos años "
    "(2024 y 2026 fueron lluviosos; 2023, muy seco), y el primer pico del motor PREDWEEM adelanta el inicio (05/03/2026). "
    "Por eso en Pergamino: (1) el reloj térmico y la ventana de control se calculan solo desde la fecha de primeras "
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

# Alerta de flushes (independiente del motor científico; no modifica sus resultados)
if "df_meteo_raw" in globals() and isinstance(df_meteo_raw, pd.DataFrame) and not df_meteo_raw.empty:
    st.divider()
    try:
        import importlib

        import alerta_lolmu as _alerta_lolmu
        import visualizacion_alerta_lolmu as _vis_alerta

        # Streamlit conserva en memoria los módulos ya importados entre ejecuciones: tras una actualización del
        # repositorio, recargarlos evita usar una versión vieja (p. ej. sin la opción tope_sin_inicio)
        importlib.reload(_alerta_lolmu)
        importlib.reload(_vis_alerta)
        mostrar_alerta_lolmu = _vis_alerta.mostrar_alerta_lolmu

        _resultados_alerta = mostrar_alerta_lolmu(
            df_meteo_raw,
            inicio_predicho=None,  # el primer pico del motor adelanta el inicio en Pergamino: el reloj parte de las primeras plántulas
            t_base=float(globals().get("t_base_val", 2.0)),
            t_opt=float(globals().get("t_opt_max", 20.0)),
            t_crit=float(globals().get("t_critica", 30.0)),
            tt_control=float(globals().get("dga_optimo", 600.0)),
            tt_limite=float(globals().get("dga_critico", 800.0)),
            localidad="Pergamino",
            nota_sitio=NOTA_PERGAMINO,
            tope_sin_inicio=True,  # sin plántulas informadas, como máximo Atención (inicio tardío y variable)
            tope_hasta_ultimo_cero=True,  # semanas ya cerradas por un conteo posterior en cero: probabilidad 0
            curva_desde_ultimo_cero=True,  # la curva acumulada arranca a mitad de camino entre el último cero y el primer conteo con plántulas
            curva_tiempo_termico=True,  # con primeras plántulas, curva acumulada en tiempo térmico desde esa fecha
        )
    except Exception as _error_alerta:  # noqa: BLE001
        st.warning(f"No se pudo calcular la alerta: {_error_alerta}")


def _escribir_hoja(writer, dataframe, nombre):
    """Escribe una hoja y aplica un formato básico de lectura."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return
    dataframe.to_excel(writer, sheet_name=nombre, index=False)
    hoja = writer.sheets[nombre]
    ultima_columna = max(0, len(dataframe.columns) - 1)
    hoja.freeze_panes(1, 0)
    hoja.autofilter(0, 0, len(dataframe), ultima_columna)
    hoja.set_column(0, ultima_columna, 18)


# Descarga final: los resultados que muestra la app (alerta, emergencia acumulada y ventana de control)
_resultados_alerta = globals().get("_resultados_alerta")
if isinstance(_resultados_alerta, dict) and _resultados_alerta:
    reporte_excel_final = io.BytesIO()
    with pd.ExcelWriter(reporte_excel_final, engine="xlsxwriter", datetime_format="dd/mm/yyyy", date_format="dd/mm/yyyy") as writer:
        for _hoja, _tabla in _resultados_alerta.items():
            _escribir_hoja(writer, _tabla, _hoja)
    reporte_excel_final.seek(0)
    _emision = _resultados_alerta["Resumen"].set_index("Indicador")["Valor"].get("Fecha de emisión", "")

    st.divider()
    st.subheader("📥 Descarga de resultados")
    st.caption(
        "Excel con los resultados que muestra la app para la fecha de emisión elegida: resumen (nivel y probabilidad de "
        "gran flujo, inicio y ventana de control 600–800 °Cd, emergencia acumulada esperada), alerta semanal, emergencia "
        "acumulada con su rango, tiempo térmico diario desde el inicio, conteos usados, clima, parámetros y validación."
    )
    st.download_button(
        label="📊 Descargar resultados en Excel",
        data=reporte_excel_final.getvalue(),
        file_name=f"PREDWEEM_Lolium_Pergamino_alerta_{str(_emision).replace('/', '-') or 'resultados'}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        key="descarga_excel_resultados_final_pergamino",
    )

mostrar_sesion()
