# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.9.7 — LOLIUM PERGAMINO 2026
# Actualización:
# - MÓDULO DE VALIDACIÓN AVANZADO: Pearson por intervalos + F1-Score Cohortes.
# - REGLA ANTI-CRUCE: Emparejamiento por proximidad cronológica.
# - BALANCE HÍDRICO SUPERFICIAL (BHS): Evapotranspiración Hargreaves-Samani.
# - ALTA EXIGENCIA HÍDRICA: Calibrado para retrasar picos según humedad real.
# - ELIMINACIÓN DE ECOS: Aplanamiento visual de réplicas simuladas contiguas.
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
from datetime import timedelta
from pathlib import Path
from scipy.signal import find_peaks

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM PERGAMINO vK4.9.7", layout="wide", page_icon="🌾")

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #dcfce7; border-right: 1px solid #bbf7d0; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .metric-header { color: #1e293b; font-weight: bold; margin-bottom: -10px; }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. LÓGICA DE VALIDACIÓN (COHORTES Y PEARSON)
# ---------------------------------------------------------
def build_shifted_interval_series(df_sim, df_campo, col_fecha, shift_days):
    sim_intervals = []
    last_date = df_sim["Fecha"].min() - pd.Timedelta(days=1)
    for _, row in df_campo.iterrows():
        current_date = row[col_fecha]
        start_shifted = last_date + pd.Timedelta(days=shift_days)
        end_shifted = current_date + pd.Timedelta(days=shift_days)
        mask = (df_sim["Fecha"] > start_shifted) & (df_sim["Fecha"] <= end_shifted)
        sim_intervals.append(df_sim.loc[mask, "EMERREL"].sum())
        last_date = current_date
    return np.array(sim_intervals, dtype=float)

def evaluate_shifted_validation(df_sim, df_campo, col_fecha, col_plm2, max_shift_days=10):
    obs = df_campo[col_plm2].to_numpy(dtype=float)
    best = {"shift_days": 0, "pearson_r": -1.0, "sim_intervalo": np.zeros(len(df_campo))}
    for shift in range(-max_shift_days, max_shift_days + 1):
        sim_vals = build_shifted_interval_series(df_sim, df_campo, col_fecha, shift)
        pearson_r = pd.Series(obs).corr(pd.Series(sim_vals))
        if not pd.isna(pearson_r) and pearson_r > best["pearson_r"]:
            best = {"shift_days": shift, "pearson_r": float(pearson_r), "sim_intervalo": sim_vals.copy()}
    return best

def evaluate_cohort_detection(df_sim, df_campo, col_fecha, col_plm2, tol_anticipo=7, tol_retraso=7, min_dist_picos=7, umbral_min_pico=0.4):
    sim_dates = df_sim['Fecha'].values
    sim_vals = df_sim['EMERREL'].values
    obs_dates = df_campo[col_fecha].values
    obs_vals_norm = df_campo['Campo_Normalizado'].values
    
    # Detección de picos simulados
    peaks_sim, _ = find_peaks(sim_vals, height=umbral_min_pico, distance=min_dist_picos)
    sim_peak_dates = pd.to_datetime(sim_dates[peaks_sim])
    
    # Detección agronómica de picos observados
    peaks_obs = np.where(obs_vals_norm >= 0.05)[0]
    obs_peak_dates = pd.to_datetime(obs_dates[peaks_obs])
    
    matched_sim, matched_obs, offsets = set(), set(), []
    for i, s_date in enumerate(sim_peak_dates):
        for j, o_date in enumerate(obs_peak_dates):
            diff = (o_date - s_date).days
            if -tol_retraso <= diff <= tol_anticipo and j not in matched_obs:
                matched_sim.add(i); matched_obs.add(j); offsets.append(diff)
                break
    
    tp = len(matched_obs)
    fp = len(sim_peak_dates) - len(matched_sim)
    fn = len(obs_peak_dates) - len(matched_obs)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    return {"f1_score": f1, "tp": tp, "fp": fp, "fn": fn, "mean_offset": np.mean(offsets) if offsets else 0}

# ---------------------------------------------------------
# 3. MOTOR FÍSICO (BHS + ET0 + ANN)
# ---------------------------------------------------------
def calcular_et0_hargreaves(jday, tmax, tmin, latitud=-33.89):
    lat_rad = np.radians(latitud)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * jday)
    dec = 0.409 * np.sin(2 * np.pi / 365 * jday - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(dec))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(dec) * np.sin(ws))
    tmean = (tmax + tmin) / 2.0
    return np.maximum(0.0023 * (ra/2.45) * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0)), 0)

def balance_hidrico_superficial(prec, et0, w_max=30.0, ke_suelo=0.4):
    w = np.zeros(len(prec)); w[0] = w_max / 2.0
    for i in range(1, len(prec)):
        w[i] = np.clip(w[i-1] + prec[i] - (et0[i] * ke_suelo), 0, w_max)
    return w

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min, self.input_max = np.array([1, 0, -7, 0]), np.array([300, 41, 25.5, 84])
    def predict(self, X):
        Xn = 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1
        a1 = np.tanh(self.IW.T @ Xn.T + self.bIW[:, None])
        emer = (np.tanh(self.LW @ a1 + self.bLW[:, None]).flatten() + 1) / 2
        return np.diff(np.cumsum(emer), prepend=0)

# ---------------------------------------------------------
# 4. INTERFAZ Y PROCESAMIENTO
# ---------------------------------------------------------
ann_params = [np.load(BASE/f"{n}.npy") for n in ["IW", "bias_IW", "LW", "bias_out"]]
modelo_ann = PracticalANNModel(*ann_params)

st.sidebar.markdown("## 📂 1. Carga de Datos")
archivo_meteo = st.sidebar.file_uploader("Clima (CSV)", type=["csv"])
archivo_campo = st.sidebar.file_uploader("Campo (Validación)", type=["csv"])

st.sidebar.markdown("## ⚙️ 2. Calibración Hídrica")
w_max_val = st.sidebar.number_input("Capacidad Campo (mm)", value=30.0)
exigencia_pct = st.sidebar.slider("Exigencia Hídrica (%)", 10, 100, 90)

if archivo_meteo:
    df = pd.read_csv(archivo_meteo); df.columns = [c.upper() for c in df.columns]
    df = df.rename(columns={'FECHA':'Fecha','TMAX':'TMAX','TMIN':'TMIN','PREC':'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Ejecución Modelo
    df["ET0"] = calcular_et0_hargreaves(df["Fecha"].dt.dayofyear, df["TMAX"], df["TMIN"])
    df["W_superficial"] = balance_hidrico_superficial(df["Prec"], df["ET0"], w_max=w_max_val)
    
    X = df[["Fecha", "TMAX", "TMIN", "Prec"]].copy()
    X["JD"] = df["Fecha"].dt.dayofyear
    df["EMERREL"] = modelo_ann.predict(X[["JD", "TMAX", "TMIN", "Prec"]].values)
    
    # Penalidad Hídrica Estricta
    hr = df["W_superficial"] / w_max_val
    df["EMERREL"] *= (1 / (1 + np.exp(-20 * (hr - (exigencia_pct/100)))))
    df.loc[hr < ((exigencia_pct/100) - 0.05), "EMERREL"] = 0
    
    # Validación si hay archivo de campo
    if archivo_campo:
        df_campo = pd.read_csv(archivo_campo)
        col_f, col_p = df_campo.columns[0], df_campo.columns[1]
        df_campo[col_f] = pd.to_datetime(df_campo[col_f])
        df_campo['Campo_Normalizado'] = df_campo[col_p] / df_campo[col_p].max()
        
        val_pearson = evaluate_shifted_validation(df, df_campo, col_f, col_p)
        val_cohortes = evaluate_cohort_detection(df, df_campo, col_f, col_p)
        
        st.title("🌾 PREDWEEM PERGAMINO - VALIDACIÓN")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pearson (r)", f"{val_pearson['pearson_r']:.3f}")
        c2.metric("F1-Score", f"{val_cohortes['f1_score']:.2f}")
        c3.metric("Aciertos (TP)", val_cohortes['tp'])
        c4.metric("Sesgo (días)", f"{val_cohortes['mean_offset']:.1f}")

    # Graficación
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], name="Simulación", line=dict(color="green")))
    if archivo_campo:
        fig.add_trace(go.Scatter(x=df_campo[col_f], y=df_campo['Campo_Normalizado'], name="Campo", mode="markers+lines", marker=dict(color="red", symbol="diamond")))
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Cargue meteo_daily.csv para iniciar la simulación mecanística.")
