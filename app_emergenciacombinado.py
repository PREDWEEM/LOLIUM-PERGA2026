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

La alerta (``alerta_lolmu.py``) es la calibrada con 17 campañas de Bordenave, validada con Pergamino 2024 y 2026.
En Pergamino el inicio de la emergencia es tardío y variable (fines de febrero en 2024; abril en 2026; mayo en 2023)
y el primer pico del motor lo adelanta (05/03/2026, cuando no hubo emergencia hasta abril). Por eso, a diferencia de
las otras localidades, el reloj térmico y la ventana de control **no** se anclan al primer pico del motor: se
calculan desde la fecha de las primeras plántulas que informa el usuario (o desde el primer conteo con plántulas).
"""
from pathlib import Path

from private_runtime import build_private_core_source
from visualizacion_horizonte_pronostico import mostrar_horizonte_pronostico

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
    "2026 y a principios de mayo en 2023 (curvas del gemelo digital). El primer pico del motor PREDWEEM adelanta el inicio "
    "(05/03/2026, cuando no hubo emergencia hasta abril), por eso aquí el reloj térmico y la ventana de control se calculan "
    "solo desde la fecha de las primeras plántulas informada (o el primer conteo con plántulas). **Informar esa fecha es "
    "lo que más mejora la alerta en Pergamino.**\\n\\n"
    "**Validación en Pergamino** (2024 y 2026, cada una con su clima SIGA de INTA Pergamino; 22 intervalos de 7 a 16 días, "
    "11 con emergencia ≥ 5 % de la campaña por semana). Solo con clima: AUC 0,93; 6 de 11 flushes en *Alerta* (en 2026, "
    "con inicio en abril, solo 2 de 6). Con la fecha de primeras plántulas: AUC 0,96; 9 de 11 flushes en *Alerta* y 2 "
    "falsas alarmas sobre 11 intervalos sin flush. Con conteos: 11 de 11 y 3 falsas alarmas.\\n\\n"
    "**Por qué no un calendario propio.** Con un calendario armado con las otras campañas de Pergamino la alerta empeora "
    "(AUC 0,85 con primeras plántulas) porque los años difieren mucho entre sí (2024 temprano; 2023 y 2026 tardíos). Por la "
    "misma razón la curva acumulada sigue siendo la de Bordenave: su error en las fechas de conteo es bajo en 2024 (8 puntos) "
    "y alto en 2026 (27 puntos, por el inicio tardío), y el calendario local no lo mejora en promedio. Tómela como "
    "referencia regional.\\n\\n"
    "**Límites.** Dos campañas evaluables; 2023 y 2024 son curvas relativas digitalizadas de boletines de la Red DIMA "
    "(quincenales); 2023 no tiene clima."
)

# Alerta de flushes (independiente del motor científico; no modifica sus resultados)
if "df_meteo_raw" in globals() and isinstance(df_meteo_raw, pd.DataFrame) and not df_meteo_raw.empty:
    st.divider()
    try:
        from visualizacion_alerta_lolmu import mostrar_alerta_lolmu

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
        "flush, inicio y ventana de control 600–800 °Cd, emergencia acumulada esperada), alerta semanal, emergencia "
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
