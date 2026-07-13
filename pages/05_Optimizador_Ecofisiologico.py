# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import hashlib
import io
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from predweem_optimizer import (
    DEFAULT_OPTIMIZED_PARAMETERS,
    PARAMETER_SPACE,
    load_ann_model,
    optimize_parameters,
    parameter_importance,
    params_to_json,
    prepare_field,
    prepare_weather,
    validate_independently,
)


st.set_page_config(page_title="Optimizador ecofisiológico PREDWEEM", page_icon="🧬", layout="wide")


def read_table(source, sheet_name=0):
    if source is None:
        return None
    name = str(getattr(source, "name", source)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(source, sheet_name=sheet_name)
    return pd.read_csv(source)


def default_file(candidates):
    for name in candidates:
        path = BASE / name
        if path.exists():
            return path
    return None


def file_fingerprint(source) -> str | None:
    if source is None:
        return None
    if isinstance(source, Path):
        return hashlib.sha256(source.read_bytes()).hexdigest()
    position = source.tell()
    source.seek(0)
    digest = hashlib.sha256(source.read()).hexdigest()
    source.seek(position)
    return digest


def metric_cards(title, summary, prefix=""):
    st.markdown(f"### {title}")
    cols = st.columns(6)
    values = [
        ("Score", summary.get(f"Score_{prefix}", summary.get("Score_Calibracion", 0.0)), ".3f"),
        ("KGE", summary.get("KGE_Flujos", summary.get("KGE_Flujos_Media", 0.0)), ".3f"),
        ("NSE", summary.get("NSE_Flujos", summary.get("NSE_Flujos_Media", 0.0)), ".3f"),
        ("CCC", summary.get("CCC_Acumulado", summary.get("CCC_Acumulado_Media", 0.0)), ".3f"),
        ("RMSE", summary.get("RMSE_Acumulado", summary.get("RMSE_Acumulado_Media", 0.0)), ".3f"),
        ("T50", summary.get("Desfase_T50", summary.get("Desfase_T50_Media", 0.0)), "+.0f"),
    ]
    for col, (label, value, fmt) in zip(cols, values):
        try:
            col.metric(label, format(float(value), fmt), "días" if label == "T50" else None)
        except Exception:
            col.metric(label, "N/D")


st.title("🧬 Optimizador de variables ecofisiológicas")
st.caption("PREDWEEM Pergamino 2026 · calibración robusta y validación independiente a campo")
st.info(
    "El conjunto de validación no participa en la selección de parámetros. "
    "Se usa únicamente después de fijar la mejor combinación con los datos de calibración."
)

try:
    ann_model = load_ann_model(BASE)
except Exception as exc:
    st.error(f"No se pudo cargar la red neuronal: {exc}")
    st.stop()

weather_default = default_file(["meteo_daily.csv", "meteo_daily.xlsx"])
calibration_default = default_file([
    "CALIBRACION.xlsx", "CALIBRA.xlsx", "pergamino_campo.xlsx", "pergamino_campo.csv"
])
validation_default = default_file(["VALIDA.xlsx", "VALIDACION.xlsx", "validacion.xlsx"])

with st.expander("1. Datos de calibración y validación", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Calibración")
        weather_cal_upload = st.file_uploader("Meteorología de calibración", type=["csv", "xlsx", "xls"], key="met_cal")
        field_cal_upload = st.file_uploader("Campo de calibración", type=["csv", "xlsx", "xls"], key="field_cal")
        mode_cal = st.selectbox("Formato de la variable de campo", ["interval", "cumulative"], format_func=lambda x: "Flujo/conteo por intervalo" if x == "interval" else "Conteo acumulado", key="mode_cal")
        st.caption(f"Meteo automática: {weather_default.name if weather_default else 'no disponible'}")
        st.caption(f"Campo automático: {calibration_default.name if calibration_default else 'no disponible; debe cargarse'}")
    with c2:
        st.markdown("#### Validación independiente")
        weather_val_upload = st.file_uploader("Meteorología de validación", type=["csv", "xlsx", "xls"], key="met_val")
        field_val_upload = st.file_uploader("Campo independiente de validación", type=["csv", "xlsx", "xls"], key="field_val")
        mode_val = st.selectbox("Formato de la variable de campo independiente", ["interval", "cumulative"], format_func=lambda x: "Flujo/conteo por intervalo" if x == "interval" else "Conteo acumulado", key="mode_val")
        use_same_weather = st.checkbox("Usar la misma meteorología para validación", value=True)
        st.caption(f"Campo automático independiente: {validation_default.name if validation_default else 'no disponible'}")

weather_cal_source = weather_cal_upload or weather_default
field_cal_source = field_cal_upload or calibration_default
weather_val_source = weather_val_upload or (weather_cal_source if use_same_weather else None)
field_val_source = field_val_upload or validation_default

with st.expander("2. Espacio de búsqueda", expanded=True):
    optimized = st.multiselect(
        "Variables a optimizar",
        list(PARAMETER_SPACE),
        default=DEFAULT_OPTIMIZED_PARAMETERS,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    st.caption("Los parámetros no seleccionados permanecen en los valores por defecto del modelo.")
    a, b, c, d = st.columns(4)
    n_global = a.number_input("Iteraciones globales", 50, 5000, 400, 50)
    n_local = b.number_input("Refinamiento local", 0, 3000, 200, 50)
    seed = c.number_input("Semilla", 0, 999999, 42, 1)
    robustness = d.slider("Penalización por inestabilidad", 0.0, 0.5, 0.15, 0.01)
    latitude = st.number_input("Latitud para ET0 Hargreaves", value=-33.9443, format="%.4f")

run = st.button("🚀 Ejecutar optimización y validar", type="primary", width="stretch")

if run:
    if weather_cal_source is None or field_cal_source is None:
        st.error("Se requieren meteorología y observaciones de campo para calibración.")
        st.stop()
    if weather_val_source is None or field_val_source is None:
        st.error("Se requieren meteorología y datos de campo independientes para validación.")
        st.stop()
    if not optimized:
        st.error("Seleccione al menos una variable para optimizar.")
        st.stop()
    if file_fingerprint(field_cal_source) == file_fingerprint(field_val_source):
        st.error("Calibración y validación contienen el mismo archivo. Use datos de campo independientes.")
        st.stop()

    try:
        weather_cal = prepare_weather(read_table(weather_cal_source))
        field_cal = prepare_field(read_table(field_cal_source), value_mode=mode_cal)
        weather_val = prepare_weather(read_table(weather_val_source))
        field_val = prepare_field(read_table(field_val_source), value_mode=mode_val)
    except Exception as exc:
        st.error(f"Error al interpretar los archivos: {exc}")
        st.stop()

    if field_cal["Fecha"].min() == field_val["Fecha"].min() and field_cal["Fecha"].max() == field_val["Fecha"].max():
        st.warning("Los rangos de fechas de calibración y validación son idénticos. Verifique que sean experimentos realmente independientes.")

    progress = st.progress(0, text="Preparando búsqueda global...")
    try:
        optimization = optimize_parameters(
            weather_cal,
            field_cal,
            ann_model,
            optimized_parameters=optimized,
            n_global=int(n_global),
            n_local=int(n_local),
            seed=int(seed),
            latitude=float(latitude),
            robustness_penalty=float(robustness),
        )
        progress.progress(80, text="Evaluando el óptimo sobre datos independientes...")
        validation = validate_independently(
            optimization, weather_val, field_val, ann_model, latitude=float(latitude)
        )
        progress.progress(100, text="Optimización finalizada.")
    except Exception as exc:
        st.exception(exc)
        st.stop()

    st.session_state["eco_optimization"] = optimization
    st.session_state["eco_validation"] = validation
    st.session_state["eco_optimized_names"] = optimized

if "eco_optimization" in st.session_state:
    optimization = st.session_state["eco_optimization"]
    validation = st.session_state["eco_validation"]
    optimized = st.session_state["eco_optimized_names"]

    st.success("Parámetros seleccionados sin utilizar el conjunto independiente de validación.")
    metric_cards("Calibración robusta", optimization["best_summary"], prefix="Calibracion")
    metric_cards("Validación independiente", validation["summary"], prefix="Validacion")

    tabs = st.tabs(["Parámetros óptimos", "Candidatos", "Validación gráfica", "Sensibilidad", "Descargas"])

    with tabs[0]:
        params_df = pd.DataFrame({
            "Parametro": list(optimization["best_params"]),
            "Valor_optimo": list(optimization["best_params"].values()),
            "Optimizado": [name in optimized for name in optimization["best_params"]],
        })
        st.dataframe(params_df, width="stretch", hide_index=True)
        st.markdown("#### Desempeño por grupo de calibración")
        st.dataframe(optimization["calibration_by_group"], width="stretch", hide_index=True)
        st.markdown("#### Desempeño por grupo de validación independiente")
        st.dataframe(validation["by_group"], width="stretch", hide_index=True)

    with tabs[1]:
        columns = ["Score_Calibracion", "Score_Medio", "Score_SD", "Score_Peor_Grupo", "Etapa"] + list(optimized)
        st.dataframe(optimization["results"][columns].head(50), width="stretch", hide_index=True)

    with tabs[2]:
        sync = validation["sync"]
        if sync.empty:
            st.warning("No se pudieron construir intervalos de validación.")
        else:
            fig = go.Figure()
            for group, part in sync.groupby("Grupo"):
                fig.add_trace(go.Scatter(x=part["Fecha"], y=part["Campo_Acumulado"], mode="markers+lines", name=f"Campo {group}"))
                fig.add_trace(go.Scatter(x=part["Fecha"], y=part["Sim_Acumulado"], mode="lines", line=dict(dash="dash"), name=f"Modelo {group}"))
            fig.update_layout(title="Validación independiente: emergencia acumulada", yaxis_title="Proporción acumulada", xaxis_title="Fecha", hovermode="x unified", height=480)
            st.plotly_chart(fig, width="stretch")

            fig11 = go.Figure()
            fig11.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="1:1", line=dict(dash="dash")))
            fig11.add_trace(go.Scatter(x=sync["Campo_Relativo"], y=sync["Sim_Relativo"], mode="markers", text=sync["Grupo"], name="Intervalos"))
            fig11.update_layout(title="Flujos por intervalo: observado vs. simulado", xaxis_title="Observado relativo", yaxis_title="Simulado relativo", height=430)
            st.plotly_chart(fig11, width="stretch")

    with tabs[3]:
        importance = parameter_importance(optimization["results"], optimized)
        st.caption("Correlación de Spearman entre cada parámetro explorado y el score de calibración; no implica causalidad.")
        st.dataframe(importance, width="stretch", hide_index=True)
        if not importance.empty:
            fig_imp = go.Figure(go.Bar(x=importance["Importancia_Abs"], y=importance["Parametro"], orientation="h"))
            fig_imp.update_layout(title="Sensibilidad global aproximada", xaxis_title="|rho de Spearman|", yaxis_title="")
            st.plotly_chart(fig_imp, width="stretch")

    with tabs[4]:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            pd.DataFrame([optimization["best_params"]]).to_excel(writer, "Parametros_Optimos", index=False)
            pd.DataFrame([optimization["best_summary"]]).to_excel(writer, "Resumen_Calibracion", index=False)
            pd.DataFrame([validation["summary"]]).to_excel(writer, "Resumen_Validacion", index=False)
            optimization["results"].to_excel(writer, "Candidatos", index=False)
            optimization["calibration_by_group"].to_excel(writer, "Calibracion_Grupos", index=False)
            validation["by_group"].to_excel(writer, "Validacion_Grupos", index=False)
            optimization["calibration_sync"].to_excel(writer, "Intervalos_Calibracion", index=False)
            validation["sync"].to_excel(writer, "Intervalos_Validacion", index=False)
        st.download_button("📥 Descargar informe Excel", output.getvalue(), "PREDWEEM_optimizacion_ecofisiologica.xlsx", width="stretch")
        st.download_button("📄 Descargar parámetros JSON", params_to_json(optimization["best_params"]), "predweem_parametros_optimos.json", mime="application/json", width="stretch")
