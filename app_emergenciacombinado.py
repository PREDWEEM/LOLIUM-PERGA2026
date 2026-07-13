# -*- coding: utf-8 -*-
"""PREDWEEM vK4.9.20: parámetros óptimos, Ke y modulador manuales."""
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "app_emergenciacombinado_vK4_9_15.py"

P = {
    "w_max": 17.514229108604354,
    "ke_suelo": 0.1,
    "humedad_p50": 0.32739222391608147,
    "pendiente_hidrica": 10.0,
    "humedad_corte": 0.06166578655561889,
    "recarga_relativa": 0.6355771533263329,
    "mod_termico": 0.85,
    "latencia_jd": 28,
    "ventana_termica": 7,
    "umbral_termoinhibicion": 26.02525090190775,
    "ventana_lluvia": 3,
    "umbral_choque_hidrico": 40.47186371294922,
    "fin_choque_jd": 110,
    "techo_choque": 0.75,
    "umbral_primer_pico": 0.7702086591581672,
    "persistencia_primer_pico": 1,
    "lag_dias": 40,
}


def replace_once(text, pattern, replacement, label, flags=0):
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"No se pudo aplicar {label}: coincidencias={count}")
    return updated


def patch(text):
    text = text.replace(
        "# 👑 PREDWEEM INTEGRAL vK4.9.15 — LOLIUM PERGAMINO 2026",
        "# 👑 PREDWEEM INTEGRAL vK4.9.20 — LOLIUM PERGAMINO 2026",
        1,
    )
    text = text.replace(
        "# - LATENCIA INICIAL: Bloqueo estricto de emergencia los primeros 45 días del año.",
        "# - LATENCIA INICIAL: bloqueo calibrado hasta JD 28.",
        1,
    )
    text = text.replace(
        "# - ESCUDO TERMOFISIOLÓGICO: Horizonte de termoinhibición dinámico ajustado a 5 días.",
        "# - ESCUDO TERMOFISIOLÓGICO: ventana 7 días y umbral 26.025251 °C.",
        1,
    )
    text = text.replace(
        "# - CHOQUE HÍDRICO: Umbral acumulado de 3 días fijado en 45 mm.",
        "# - CHOQUE HÍDRICO: umbral 40.471864 mm en 3 días.",
        1,
    )
    text = text.replace(
        "# - PRIMER PICO VÁLIDO: La campaña se habilita únicamente cuando EMERREL > 0.70.",
        "# - PRIMER PICO VÁLIDO: campaña habilitada cuando EMERREL > 0.770209.",
        1,
    )
    text = text.replace(
        "# - SIMULACIÓN: Sin incremento térmico artificial; lag fijo de emergencia de +22 días.",
        "# - CALIBRACIÓN CV TEMPORAL INTERNA: parámetros del 13/07/2026; lag +40 días.",
        1,
    )

    params = '''PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713 = {
    "w_max": 17.514229108604354,
    "humedad_p50": 0.32739222391608147,
    "pendiente_hidrica": 10.0,
    "humedad_corte": 0.06166578655561889,
    "recarga_relativa": 0.6355771533263329,
    "latencia_jd": 28,
    "ventana_termica": 7,
    "umbral_termoinhibicion": 26.02525090190775,
    "ventana_lluvia": 3,
    "umbral_choque_hidrico": 40.47186371294922,
    "fin_choque_jd": 110,
    "techo_choque": 0.75,
    "umbral_primer_pico": 0.7702086591581672,
    "persistencia_primer_pico": 1,
    "lag_dias": 40,
}
REFERENCIA_SUPERFICIE_OPTIMIZADOR = {"ke_suelo": 0.1, "mod_termico": 0.85}
UMBRAL_PRIMER_PICO = PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["umbral_primer_pico"]
LAG_EMERGENCIA_DIAS = int(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["lag_dias"])
'''
    text = replace_once(
        text,
        r"UMBRAL_PRIMER_PICO\s*=\s*0\.70\s*\nLAG_EMERGENCIA_DIAS\s*=\s*22[^\n]*\n",
        params,
        "bloque de parámetros",
    )

    balance = '''def balance_hidrico_superficial(prec, et0, w_max=30.0, ke_suelo=0.4):
    prec = np.asarray(prec, dtype=float)
    et0 = np.asarray(et0, dtype=float)
    n = len(prec)
    w = np.zeros(n, dtype=float)
    if n == 0:
        return w
    w[0] = np.clip(w_max / 2.0 + prec[0] - et0[0] * ke_suelo, 0.0, w_max)
    for i in range(1, n):
        w[i] = np.clip(w[i - 1] + prec[i] - et0[i] * ke_suelo, 0.0, w_max)
    return w
'''
    text = replace_once(
        text,
        r"def balance_hidrico_superficial\(prec, et0, w_max=30\.0, ke_suelo=0\.4\):.*?\n    return w\n",
        balance,
        "balance hídrico",
        re.S,
    )

    manual = '''st.caption("Ke y el modulador térmico se configuran manualmente; los demás parámetros son los óptimos.")
            col_ke, col_mt = st.columns(2)
            with col_ke:
                ke_val = st.number_input(
                    "Coeficiente hídrico del suelo (Ke)", min_value=0.05,
                    max_value=1.20, value=0.25, step=0.01, format="%.2f",
                    help="Valor manual. Referencia del optimizador: 0.10.",
                )
            with col_mt:
                mod_termico = st.number_input(
                    "Modulador térmico del suelo", min_value=0.50,
                    max_value=1.20, value=0.85, step=0.01, format="%.2f",
                    help="Valor manual. Referencia del optimizador: 0.85.",
                )
            cobertura_pct = None
'''
    text = replace_once(
        text,
        r"cobertura_pct = st\.slider\(.*?\n\s*mod_termico = float\(np\.interp\(.*?\)\)\n",
        manual,
        "controles manuales",
        re.S,
    )

    text = replace_once(
        text,
        r'umbral_termoinhibicion = st\.sidebar\.number_input\([^\n]+\)',
        'umbral_termoinhibicion = float(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["umbral_termoinhibicion"])\n'
        'st.sidebar.caption(f"Termoinhibición fija: {umbral_termoinhibicion:.6f} °C")',
        "termoinhibición fija",
    )
    text = replace_once(
        text,
        r"umbral_choque_hidrico = st\.sidebar\.slider\(.*?\n\s*\)\n",
        'umbral_choque_hidrico = float(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["umbral_choque_hidrico"])\n'
        'st.sidebar.caption(f"Choque hídrico fijo: {umbral_choque_hidrico:.6f} mm en 3 días")\n',
        "choque hídrico fijo",
        re.S,
    )
    text = replace_once(
        text,
        r'w_max_val = st\.sidebar\.number_input\("Cap\. de Campo Superficial \(mm\)", value=[0-9.]+, step=1\.0\)',
        'w_max_val = float(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["w_max"])\n'
        'st.sidebar.caption(f"Wmax fijo optimizado: {w_max_val:.6f} mm")',
        "Wmax fijo",
    )

    panel = '''# --- PARÁMETROS CALIBRADOS ---
with st.sidebar.expander("🧬 Parámetros óptimos aplicados", expanded=False):
    st.caption("CV temporal interna con VALIDA.xlsx. Ke y modulador térmico son manuales.")
    st.dataframe(pd.DataFrame({"Parámetro fijo": list(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713), "Valor": list(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713.values())}), width="stretch", hide_index=True)
    st.write({"Ke manual": float(ke_val), "Modulador térmico manual": float(mod_termico)})

# ---------------------------------------------------------
# 6. MOTOR DE CÁLCULO GENERAL
'''
    text = replace_once(
        text,
        r"# --- MODO DESARROLLADOR: CALIBRADOR 2D ---.*?# ---------------------------------------------------------\n# 6\. MOTOR DE CÁLCULO GENERAL\n",
        panel,
        "panel de parámetros",
        re.S,
    )

    motor = '''    # 1. Latencia calibrada.
    latencia_jd = int(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["latencia_jd"])
    df.loc[df["Julian_days"] <= latencia_jd, "EMERREL"] = 0.0

    # 2. Choque hídrico temprano calibrado.
    ventana_lluvia = int(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["ventana_lluvia"])
    col_prec_acum = f"Prec_{ventana_lluvia}d"
    df[col_prec_acum] = df["Prec"].rolling(window=ventana_lluvia, min_periods=1).sum()
    mask_ruptura = ((df["Julian_days"] > latencia_jd)
        & (df["Julian_days"] <= int(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["fin_choque_jd"]))
        & (df[col_prec_acum] >= umbral_choque_hidrico))
    df.loc[mask_ruptura, "EMERREL"] = np.maximum(
        df.loc[mask_ruptura, "EMERREL"],
        float(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["techo_choque"]),
    )

    # 3. Balance hídrico calibrado; Ke es manual.
    df["ET0"] = calcular_et0_hargreaves(df["Julian_days"].values, df["TMAX"].values, df["TMIN"].values, latitud=-33.9443)
    df["W_superficial"] = balance_hidrico_superficial(df["Prec"].values, df["ET0"].values, w_max=w_max_val, ke_suelo=ke_val)
    humedad_relativa = df["W_superficial"] / max(w_max_val, 1e-12)
    pendiente = float(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["pendiente_hidrica"])
    p50 = float(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["humedad_p50"])
    exponente = np.clip(-pendiente * (humedad_relativa - p50), -60, 60)
    df["Hydric_Factor"] = 1.0 / (1.0 + np.exp(exponente))
    df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]
    df.loc[humedad_relativa < float(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["humedad_corte"]), "EMERREL"] = 0.0
    df["Recarga_Habilitada"] = pd.Series(
        humedad_relativa >= float(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["recarga_relativa"]),
        index=df.index,
    ).cummax()
    df.loc[~df["Recarga_Habilitada"], "EMERREL"] = 0.0

    # 4. Termoinhibición calibrada; modulador térmico es manual.
    ventana_termica = int(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["ventana_termica"])
    df["Tmedia"] = df["Tmedia_aire"]
    col_tmedia_movil = f"Tmedia_{ventana_termica}d"
    df[col_tmedia_movil] = df["Tmedia"].rolling(window=ventana_termica, min_periods=1).mean()
    df.loc[df[col_tmedia_movil] >= umbral_termoinhibicion, "EMERREL"] = 0.0
    df["Termoinhibida"] = df[col_tmedia_movil] >= umbral_termoinhibicion
    df["EMERREL"] = np.clip(df["EMERREL"], 0.0, 1.0)

    # 5. Primer pico y lag calibrados.
    df, idx_primer_pico_original = aplicar_filtro_primer_pico(df, umbral=UMBRAL_PRIMER_PICO)
    df = aplicar_lag_emergencia(df, lag_dias=LAG_EMERGENCIA_DIAS, col="EMERREL")
    df, idx_primer_pico = aplicar_filtro_primer_pico(df, umbral=UMBRAL_PRIMER_PICO)

    df["DG"]'''
    return replace_once(
        text,
        r"    # 1\. Choque Hídrico.*?\n    df\[\"DG\"\]",
        motor,
        "motor ecofisiológico",
        re.S,
    )


if not SOURCE.exists():
    raise FileNotFoundError(f"Falta el modelo base: {SOURCE.name}")
source = SOURCE.read_text(encoding="utf-8")
optimized = patch(source)
exec(compile(optimized, str(SOURCE), "exec"), {
    "__name__": "__main__", "__file__": str(SOURCE), "__package__": None
})
