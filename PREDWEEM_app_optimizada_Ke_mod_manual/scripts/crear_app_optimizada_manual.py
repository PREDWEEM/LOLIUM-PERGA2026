#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crea una copia optimizada de app_emergenciacombinado.py.

Todos los parámetros seleccionados por CV temporal se fijan en el modelo,
excepto Ke y el modulador térmico, que permanecen como controles manuales.

Uso:
    python scripts/crear_app_optimizada_manual.py app_emergenciacombinado.py \
        --csv data/parametros_optimos_2026-07-13.csv \
        --output modelo_optimizado_manual/app_emergenciacombinado.py

El archivo original no se modifica. El script se detiene si la estructura
esperada del modelo original no coincide.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

EXPECTED = {
    "w_max",
    "ke_suelo",
    "humedad_p50",
    "pendiente_hidrica",
    "humedad_corte",
    "recarga_relativa",
    "mod_termico",
    "latencia_jd",
    "ventana_termica",
    "umbral_termoinhibicion",
    "ventana_lluvia",
    "umbral_choque_hidrico",
    "fin_choque_jd",
    "techo_choque",
    "umbral_primer_pico",
    "persistencia_primer_pico",
    "lag_dias",
}
INTEGER_PARAMS = {
    "latencia_jd",
    "ventana_termica",
    "ventana_lluvia",
    "fin_choque_jd",
    "persistencia_primer_pico",
    "lag_dias",
}


def read_params(path: Path) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Parametro", "Valor_optimo"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"El CSV debe contener {sorted(required)}; contiene {reader.fieldnames}."
            )
        for row in reader:
            name = str(row["Parametro"]).strip()
            if not name:
                continue
            raw = float(str(row["Valor_optimo"]).replace(",", "."))
            values[name] = int(round(raw)) if name in INTEGER_PARAMS else raw
    missing = EXPECTED - values.keys()
    if missing:
        raise ValueError("Faltan parámetros: " + ", ".join(sorted(missing)))
    return values


def replace_once(text: str, pattern: str, replacement: str, label: str, flags: int = 0) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(
            f"No se pudo aplicar el cambio '{label}' (coincidencias={count}). "
            "Verifique que se usa app_emergenciacombinado.py vK4.9.15 sin incremento y lag 22."
        )
    return updated


def f(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def patch(text: str, p: dict[str, float | int]) -> str:
    if "PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713" in text:
        return text

    text = text.replace(
        "# 👑 PREDWEEM INTEGRAL vK4.9.15 — LOLIUM PERGAMINO 2026",
        "# 👑 PREDWEEM INTEGRAL vK4.9.20 — LOLIUM PERGAMINO 2026",
        1,
    )
    text = text.replace(
        "# - LATENCIA INICIAL: Bloqueo estricto de emergencia los primeros 45 días del año.",
        f"# - LATENCIA INICIAL: bloqueo calibrado hasta JD {p['latencia_jd']}.",
        1,
    )
    text = text.replace(
        "# - ESCUDO TERMOFISIOLÓGICO: Horizonte de termoinhibición dinámico ajustado a 5 días.",
        f"# - ESCUDO TERMOFISIOLÓGICO: ventana {p['ventana_termica']} días y umbral {float(p['umbral_termoinhibicion']):.6f} °C.",
        1,
    )
    text = text.replace(
        "# - CHOQUE HÍDRICO: Umbral acumulado de 3 días fijado en 45 mm.",
        f"# - CHOQUE HÍDRICO: umbral {float(p['umbral_choque_hidrico']):.6f} mm en {p['ventana_lluvia']} días.",
        1,
    )
    text = text.replace(
        "# - PRIMER PICO VÁLIDO: La campaña se habilita únicamente cuando EMERREL > 0.70.",
        f"# - PRIMER PICO VÁLIDO: campaña habilitada cuando EMERREL > {float(p['umbral_primer_pico']):.6f}.",
        1,
    )
    text = text.replace(
        "# - SIMULACIÓN: Sin incremento térmico artificial; lag fijo de emergencia de +22 días.",
        f"# - CALIBRACIÓN CV TEMPORAL INTERNA: parámetros del 13/07/2026; lag +{p['lag_dias']} días.",
        1,
    )

    params_block = f'''PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713 = {{
    "w_max": {f(p['w_max'])},
    "humedad_p50": {f(p['humedad_p50'])},
    "pendiente_hidrica": {f(p['pendiente_hidrica'])},
    "humedad_corte": {f(p['humedad_corte'])},
    "recarga_relativa": {f(p['recarga_relativa'])},
    "latencia_jd": {f(p['latencia_jd'])},
    "ventana_termica": {f(p['ventana_termica'])},
    "umbral_termoinhibicion": {f(p['umbral_termoinhibicion'])},
    "ventana_lluvia": {f(p['ventana_lluvia'])},
    "umbral_choque_hidrico": {f(p['umbral_choque_hidrico'])},
    "fin_choque_jd": {f(p['fin_choque_jd'])},
    "techo_choque": {f(p['techo_choque'])},
    "umbral_primer_pico": {f(p['umbral_primer_pico'])},
    "persistencia_primer_pico": {f(p['persistencia_primer_pico'])},
    "lag_dias": {f(p['lag_dias'])},
}}

# Valores de referencia del optimizador. Se muestran como ayuda,
# pero Ke y el modulador térmico se definen manualmente en la interfaz.
REFERENCIA_SUPERFICIE_OPTIMIZADOR = {{
    "ke_suelo": {f(p['ke_suelo'])},
    "mod_termico": {f(p['mod_termico'])},
}}

UMBRAL_PRIMER_PICO = PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["umbral_primer_pico"]
LAG_EMERGENCIA_DIAS = int(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["lag_dias"])
'''
    text = replace_once(
        text,
        r"UMBRAL_PRIMER_PICO\s*=\s*0\.70\s*\nLAG_EMERGENCIA_DIAS\s*=\s*22[^\n]*\n",
        params_block,
        "bloque de parámetros",
    )

    balance_block = '''def balance_hidrico_superficial(prec, et0, w_max=30.0, ke_suelo=0.4):
    prec = np.asarray(prec, dtype=float)
    et0 = np.asarray(et0, dtype=float)
    n = len(prec)
    w = np.zeros(n, dtype=float)
    if n == 0:
        return w
    # Misma condición inicial utilizada durante la optimización.
    w[0] = np.clip(w_max / 2.0 + prec[0] - et0[0] * ke_suelo, 0.0, w_max)
    for i in range(1, n):
        w[i] = np.clip(w[i - 1] + prec[i] - et0[i] * ke_suelo, 0.0, w_max)
    return w
'''
    text = replace_once(
        text,
        r"def balance_hidrico_superficial\(prec, et0, w_max=30\.0, ke_suelo=0\.4\):.*?\n    return w\n",
        balance_block,
        "balance hídrico",
        re.S,
    )

    surface_block = '''st.caption(
                "Ke y el modulador térmico se configuran manualmente. "
                "Los demás parámetros permanecen fijados en los valores óptimos."
            )
            col_ke, col_mt = st.columns(2)
            with col_ke:
                ke_val = st.number_input(
                    "Coeficiente hídrico del suelo (Ke)",
                    min_value=0.05,
                    max_value=1.20,
                    value=0.25,
                    step=0.01,
                    format="%.2f",
                    help=(
                        "Valor manual. Referencia del optimizador: "
                        f"{REFERENCIA_SUPERFICIE_OPTIMIZADOR['ke_suelo']:.2f}."
                    ),
                )
            with col_mt:
                mod_termico = st.number_input(
                    "Modulador térmico del suelo",
                    min_value=0.50,
                    max_value=1.20,
                    value=0.85,
                    step=0.01,
                    format="%.2f",
                    help=(
                        "Valor manual. Referencia del optimizador: "
                        f"{REFERENCIA_SUPERFICIE_OPTIMIZADOR['mod_termico']:.2f}."
                    ),
                )
            cobertura_pct = None
'''
    text = replace_once(
        text,
        r"cobertura_pct = st\.slider\(.*?\n\s*mod_termico = float\(np\.interp\(.*?\)\)\n",
        surface_block,
        "parámetros de superficie",
        re.S,
    )

    text = replace_once(
        text,
        r'umbral_termoinhibicion = st\.sidebar\.number_input\([^\n]+\)',
        'umbral_termoinhibicion = float(\n'
        '    PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["umbral_termoinhibicion"]\n'
        ')\n'
        'st.sidebar.caption(\n'
        '    f"Termoinhibición fija: {umbral_termoinhibicion:.6f} °C"\n'
        ')',
        "umbral térmico fijo",
    )

    shock_ui = '''umbral_choque_hidrico = float(
    PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["umbral_choque_hidrico"]
)
st.sidebar.caption(
    f"Choque hídrico fijo: {umbral_choque_hidrico:.6f} mm en "
    f"{int(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713['ventana_lluvia'])} días"
)
'''
    text = replace_once(
        text,
        r"umbral_choque_hidrico = st\.sidebar\.slider\(.*?\n\s*\)\n",
        shock_ui,
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

    dev_block = '''# --- PARÁMETROS CALIBRADOS ---
with st.sidebar.expander("🧬 Parámetros óptimos aplicados", expanded=False):
    st.caption(
        "Valores seleccionados con VALIDA.xlsx mediante CV temporal interna. "
        "Ke y modulador térmico se excluyen porque se configuran manualmente."
    )
    tabla_parametros_fijos = pd.DataFrame(
        {
            "Parámetro fijo": list(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713.keys()),
            "Valor": list(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713.values()),
        }
    )
    st.dataframe(tabla_parametros_fijos, width="stretch", hide_index=True)
    st.markdown("**Parámetros manuales actuales**")
    st.write({"Ke": float(ke_val), "Modulador térmico": float(mod_termico)})

# ---------------------------------------------------------
# 6. MOTOR DE CÁLCULO GENERAL
'''
    text = replace_once(
        text,
        r"# --- MODO DESARROLLADOR: CALIBRADOR 2D ---.*?# ---------------------------------------------------------\n# 6\. MOTOR DE CÁLCULO GENERAL\n",
        dev_block,
        "panel de parámetros",
        re.S,
    )

    model_block = '''    # 1. Bloqueo de latencia calibrado.
    latencia_jd = int(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["latencia_jd"])
    df.loc[df["Julian_days"] <= latencia_jd, "EMERREL"] = 0.0

    # 2. Choque hídrico temprano calibrado.
    ventana_lluvia = int(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["ventana_lluvia"])
    col_prec_acum = f"Prec_{ventana_lluvia}d"
    df[col_prec_acum] = df["Prec"].rolling(window=ventana_lluvia, min_periods=1).sum()
    mask_ruptura = (
        (df["Julian_days"] > latencia_jd)
        & (df["Julian_days"] <= int(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["fin_choque_jd"]))
        & (df[col_prec_acum] >= umbral_choque_hidrico)
    )
    df.loc[mask_ruptura, "EMERREL"] = np.maximum(
        df.loc[mask_ruptura, "EMERREL"],
        float(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["techo_choque"]),
    )

    # 3. Balance hídrico superficial y filtro sigmoide calibrados.
    df["ET0"] = calcular_et0_hargreaves(
        df["Julian_days"].values,
        df["TMAX"].values,
        df["TMIN"].values,
        latitud=-33.9443,
    )
    df["W_superficial"] = balance_hidrico_superficial(
        df["Prec"].values,
        df["ET0"].values,
        w_max=w_max_val,
        ke_suelo=ke_val,
    )
    humedad_relativa = df["W_superficial"] / max(w_max_val, 1e-12)
    pendiente_hidrica = float(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["pendiente_hidrica"])
    humedad_p50 = float(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["humedad_p50"])
    exponente = np.clip(-pendiente_hidrica * (humedad_relativa - humedad_p50), -60, 60)
    df["Hydric_Factor"] = 1.0 / (1.0 + np.exp(exponente))
    df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]
    df.loc[
        humedad_relativa < float(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["humedad_corte"]),
        "EMERREL",
    ] = 0.0

    # La recarga depende del estado hídrico alcanzado y admite lluvias sucesivas moderadas.
    df["Recarga_Habilitada"] = pd.Series(
        humedad_relativa >= float(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["recarga_relativa"]),
        index=df.index,
    ).cummax()
    df.loc[~df["Recarga_Habilitada"], "EMERREL"] = 0.0

    # 4. Termoinhibición calibrada.
    ventana_termica = int(PARAMETROS_OPTIMOS_CV_TEMPORAL_20260713["ventana_termica"])
    df["Tmedia"] = df["Tmedia_aire"]
    col_tmedia_movil = f"Tmedia_{ventana_termica}d"
    df[col_tmedia_movil] = df["Tmedia"].rolling(
        window=ventana_termica,
        min_periods=1,
    ).mean()
    df.loc[df[col_tmedia_movil] >= umbral_termoinhibicion, "EMERREL"] = 0.0
    df["Termoinhibida"] = df[col_tmedia_movil] >= umbral_termoinhibicion

    df["EMERREL"] = np.clip(df["EMERREL"], 0.0, 1.0)

    # 5. Primer pico y lag calibrados, antes y después del desplazamiento.
    df, idx_primer_pico_original = aplicar_filtro_primer_pico(
        df,
        umbral=UMBRAL_PRIMER_PICO,
    )
    df = aplicar_lag_emergencia(
        df,
        lag_dias=LAG_EMERGENCIA_DIAS,
        col="EMERREL",
    )
    df, idx_primer_pico = aplicar_filtro_primer_pico(
        df,
        umbral=UMBRAL_PRIMER_PICO,
    )

    df["DG"]'''
    text = replace_once(
        text,
        r"    # 1\. Choque Hídrico.*?\n    df\[\"DG\"\]",
        model_block,
        "motor ecofisiológico principal",
        re.S,
    )

    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "app",
        type=Path,
        help="Ruta al app_emergenciacombinado.py original.",
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("modelo_optimizado_manual") / "app_emergenciacombinado.py",
        help=(
            "Ruta del nuevo archivo. Por defecto se crea "
            "modelo_optimizado_manual/app_emergenciacombinado.py"
        ),
    )
    args = parser.parse_args()

    source = args.app
    if not source.exists():
        raise FileNotFoundError(source)

    destination = args.output
    if source.resolve() == destination.resolve():
        raise ValueError(
            "El archivo de salida debe ser distinto del original para no sobrescribirlo."
        )

    params = read_params(args.csv)
    original = source.read_text(encoding="utf-8")
    updated = patch(original, params)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(updated, encoding="utf-8")

    print(f"Original preservado: {source}")
    print(f"Nuevo modelo creado: {destination}")
    print(
        "Parámetros manuales: Ke y modulador térmico. "
        "El resto quedó fijado con los valores óptimos."
    )


if __name__ == "__main__":
    main()
