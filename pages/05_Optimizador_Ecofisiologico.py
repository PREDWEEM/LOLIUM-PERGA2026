# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
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
    optimize_parameters_temporal_cv,
    params_to_json,
    prepare_field,
    prepare_weather,
)


st.set_page_config(
    page_title="Optimizador ecofisiológico PREDWEEM",
    page_icon="🧬",
    layout="wide",
)


def read_table(source):
    if source is None:
        return None
    name = str(getattr(source, "name", source)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(source)
    return pd.read_csv(source)


def default_file(candidates):
    for name in candidates:
        path = BASE / name
        if path.exists():
            return path
    return None


def cv_parameter_importance(results, optimized):
    rows = []
    for name in optimized:
        if name in results and results[name].nunique() > 1:
            rho = (
                results[[name, "Score_CV"]]
                .corr(method="spearman")
                .iloc[0, 1]
            )
            if pd.notna(rho):
                rows.append({
                    "Parametro": name,
                    "Rho_Spearman": float(rho),
                    "Importancia_Abs": abs(float(rho)),
                })
    if not rows:
        return pd.DataFrame(
            columns=["Parametro", "Rho_Spearman", "Importancia_Abs"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Importancia_Abs", ascending=False)
        .reset_index(drop=True)
    )


def metric_cards(title, summary, score_key):
    st.markdown(f"### {title}")
    cols = st.columns(6)
    values = [
        ("Score", summary.get(score_key, 0.0), ".3f"),
        ("KGE", summary.get("KGE_Flujos_Media", 0.0), ".3f"),
        ("NSE", summary.get("NSE_Flujos_Media", 0.0), ".3f"),
        ("CCC", summary.get("CCC_Acumulado_Media", 0.0), ".3f"),
        ("RMSE", summary.get("RMSE_Acumulado_Media", 0.0), ".3f"),
        ("T50", summary.get("Desfase_T50_Media", 0.0), "+.0f"),
    ]
    for col, (label, value, fmt) in zip(cols, values):
        try:
            col.metric(
                label,
                format(float(value), fmt),
                "días" if label == "T50" else None,
            )
        except Exception:
            col.metric(label, "N/D")


st.title("🧬 Optimizador de variables ecofisiológicas")
st.caption(
    "PREDWEEM Pergamino 2026 · validación cruzada temporal interna"
)
st.warning(
    "Actualmente existe un único conjunto de campo: VALIDA.xlsx. "
    "La aplicación divide sus intervalos en bloques cronológicos contiguos. "
    "El resultado es validación interna y no debe denominarse validación "
    "independiente entre campañas o localidades."
)

try:
    ann_model = load_ann_model(BASE)
except Exception as exc:
    st.error(f"No se pudo cargar la red neuronal: {exc}")
    st.stop()

weather_default = default_file(
    ["meteo_daily.csv", "meteo_daily.xlsx"]
)
field_default = default_file(
    ["VALIDA.xlsx", "VALIDACION.xlsx", "validacion.xlsx"]
)

with st.expander("1. Datos disponibles", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        weather_upload = st.file_uploader(
            "Meteorología diaria",
            type=["csv", "xlsx", "xls"],
        )
        st.caption(
            "Archivo automático: "
            f"{weather_default.name if weather_default else 'no disponible'}"
        )
    with c2:
        field_upload = st.file_uploader(
            "Observaciones de campo",
            type=["csv", "xlsx", "xls"],
        )
        field_mode = st.selectbox(
            "Formato de la variable observada",
            ["interval", "cumulative"],
            format_func=lambda value: (
                "Flujo/conteo por intervalo"
                if value == "interval"
                else "Conteo acumulado"
            ),
        )
        st.caption(
            "Archivo automático: "
            f"{field_default.name if field_default else 'no disponible'}"
        )

weather_source = weather_upload or weather_default
field_source = field_upload or field_default

with st.expander("2. Espacio de búsqueda", expanded=True):
    optimized = st.multiselect(
        "Variables a optimizar",
        list(PARAMETER_SPACE),
        default=DEFAULT_OPTIMIZED_PARAMETERS,
        format_func=lambda value: value.replace("_", " ").title(),
    )
    st.caption(
        "Los parámetros no seleccionados permanecen en sus valores "
        "ecofisiológicos por defecto."
    )
    a, b, c, d = st.columns(4)
    n_global = a.number_input(
        "Iteraciones globales",
        50,
        5000,
        400,
        50,
    )
    n_local = b.number_input(
        "Refinamiento local",
        0,
        3000,
        200,
        50,
    )
    seed = c.number_input("Semilla", 0, 999999, 42, 1)
    robustness = d.slider(
        "Penalización por inestabilidad",
        0.0,
        0.5,
        0.15,
        0.01,
    )
    e, f = st.columns(2)
    folds = e.slider(
        "Bloques temporales solicitados",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        help=(
            "Cada bloque debe contener al menos dos intervalos. "
            "El sistema reduce automáticamente la cantidad si no existen "
            "suficientes fechas."
        ),
    )
    latitude = f.number_input(
        "Latitud para ET0 Hargreaves",
        value=-33.9443,
        format="%.4f",
    )

run = st.button(
    "🚀 Optimizar con VALIDA.xlsx",
    type="primary",
    width="stretch",
)

if run:
    if weather_source is None or field_source is None:
        st.error(
            "Se requieren la meteorología y el conjunto de campo VALIDA.xlsx."
        )
        st.stop()
    if not optimized:
        st.error("Seleccione al menos una variable para optimizar.")
        st.stop()

    try:
        weather_data = prepare_weather(read_table(weather_source))
        field_data = prepare_field(
            read_table(field_source),
            value_mode=field_mode,
        )
        progress = st.progress(
            10,
            text="Construyendo bloques temporales...",
        )
        result = optimize_parameters_temporal_cv(
            weather_data,
            field_data,
            ann_model,
            optimized_parameters=optimized,
            n_global=int(n_global),
            n_local=int(n_local),
            seed=int(seed),
            latitude=float(latitude),
            robustness_penalty=float(robustness),
            n_folds=int(folds),
            min_intervals_per_fold=2,
        )
        progress.progress(
            100,
            text="Optimización y CV temporal finalizadas.",
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()

    st.session_state["eco_cv_result"] = result
    st.session_state["eco_cv_optimized"] = optimized

if "eco_cv_result" in st.session_state:
    result = st.session_state["eco_cv_result"]
    optimized = st.session_state["eco_cv_optimized"]

    st.success(
        "Parámetros seleccionados por desempeño medio y estabilidad "
        "entre bloques temporales."
    )
    metric_cards(
        "Validación cruzada temporal interna",
        result["best_summary"],
        "Score_CV",
    )

    apparent = result["apparent_summary"]
    apparent_for_cards = {
        "Score_Aparente": apparent.get("Score_Calibracion", 0.0),
        "KGE_Flujos_Media": apparent.get("KGE_Flujos_Media", 0.0),
        "NSE_Flujos_Media": apparent.get("NSE_Flujos_Media", 0.0),
        "CCC_Acumulado_Media": apparent.get(
            "CCC_Acumulado_Media", 0.0
        ),
        "RMSE_Acumulado_Media": apparent.get(
            "RMSE_Acumulado_Media", 0.0
        ),
        "Desfase_T50_Media": apparent.get(
            "Desfase_T50_Media", 0.0
        ),
    }
    metric_cards(
        "Ajuste aparente sobre toda la serie",
        apparent_for_cards,
        "Score_Aparente",
    )

    tabs = st.tabs(
        [
            "Parámetros óptimos",
            "Bloques temporales",
            "Candidatos",
            "Gráficos",
            "Sensibilidad",
            "Descargas",
        ]
    )

    with tabs[0]:
        params_df = pd.DataFrame({
            "Parametro": list(result["best_params"]),
            "Valor_optimo": list(result["best_params"].values()),
            "Optimizado": [
                name in optimized
                for name in result["best_params"]
            ],
        })
        st.dataframe(
            params_df,
            width="stretch",
            hide_index=True,
        )

    with tabs[1]:
        st.caption(
            "Cada fila corresponde a un bloque cronológico retenido. "
            "La variabilidad entre bloques se penaliza en el score final."
        )
        st.dataframe(
            result["cv_by_fold"],
            width="stretch",
            hide_index=True,
        )
        st.markdown("#### Definición de los intervalos")
        st.dataframe(
            result["fold_intervals"],
            width="stretch",
            hide_index=True,
        )

    with tabs[2]:
        columns = [
            "Score_CV",
            "Score_CV_Medio",
            "Score_CV_SD",
            "Score_CV_Peor_Bloque",
            "Etapa",
        ] + list(optimized)
        st.dataframe(
            result["results"][columns].head(50),
            width="stretch",
            hide_index=True,
        )

    with tabs[3]:
        sync = result["apparent_sync"]
        if sync.empty:
            st.warning("No se pudieron construir intervalos.")
        else:
            fig = go.Figure()
            for group, part in sync.groupby("Grupo"):
                fig.add_trace(
                    go.Scatter(
                        x=part["Fecha"],
                        y=part["Campo_Acumulado"],
                        mode="markers+lines",
                        name=f"Campo {group}",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=part["Fecha"],
                        y=part["Sim_Acumulado"],
                        mode="lines",
                        line=dict(dash="dash"),
                        name=f"Modelo {group}",
                    )
                )
            fig.update_layout(
                title=(
                    "Ajuste completo descriptivo "
                    "(no es validación independiente)"
                ),
                yaxis_title="Proporción acumulada",
                xaxis_title="Fecha",
                hovermode="x unified",
                height=480,
            )
            st.plotly_chart(fig, width="stretch")

            cv_sync = result["cv_sync"]
            fig11 = go.Figure()
            fig11.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="1:1",
                    line=dict(dash="dash"),
                )
            )
            fig11.add_trace(
                go.Scatter(
                    x=cv_sync["Campo_Relativo"],
                    y=cv_sync["Sim_Relativo"],
                    mode="markers",
                    text=cv_sync["Fold"],
                    name="Intervalos retenidos",
                )
            )
            fig11.update_layout(
                title=(
                    "CV temporal: observado vs. simulado por intervalo"
                ),
                xaxis_title="Observado relativo",
                yaxis_title="Simulado relativo",
                height=430,
            )
            st.plotly_chart(fig11, width="stretch")

    with tabs[4]:
        importance = cv_parameter_importance(
            result["results"],
            optimized,
        )
        st.caption(
            "Correlación de Spearman con el score de CV. "
            "Es una sensibilidad aproximada y no demuestra causalidad."
        )
        st.dataframe(
            importance,
            width="stretch",
            hide_index=True,
        )
        if not importance.empty:
            fig_imp = go.Figure(
                go.Bar(
                    x=importance["Importancia_Abs"],
                    y=importance["Parametro"],
                    orientation="h",
                )
            )
            fig_imp.update_layout(
                title="Sensibilidad global aproximada",
                xaxis_title="|rho de Spearman|",
                yaxis_title="",
            )
            st.plotly_chart(fig_imp, width="stretch")

    with tabs[5]:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            pd.DataFrame(
                [result["best_params"]]
            ).to_excel(
                writer,
                "Parametros_Optimos",
                index=False,
            )
            pd.DataFrame(
                [result["best_summary"]]
            ).to_excel(
                writer,
                "Resumen_CV_Temporal",
                index=False,
            )
            pd.DataFrame(
                [result["apparent_summary"]]
            ).to_excel(
                writer,
                "Ajuste_Aparente",
                index=False,
            )
            result["cv_by_fold"].to_excel(
                writer,
                "Metricas_Bloques",
                index=False,
            )
            result["fold_intervals"].to_excel(
                writer,
                "Definicion_Bloques",
                index=False,
            )
            result["results"].to_excel(
                writer,
                "Candidatos",
                index=False,
            )
            result["cv_sync"].to_excel(
                writer,
                "Intervalos_CV",
                index=False,
            )
            result["apparent_sync"].to_excel(
                writer,
                "Intervalos_Completos",
                index=False,
            )

        st.download_button(
            "📥 Descargar informe Excel",
            output.getvalue(),
            "PREDWEEM_optimizacion_CV_temporal.xlsx",
            width="stretch",
        )
        st.download_button(
            "📄 Descargar parámetros JSON",
            params_to_json(result["best_params"]),
            "predweem_parametros_optimos.json",
            mime="application/json",
            width="stretch",
        )
