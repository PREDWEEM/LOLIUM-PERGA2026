# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM vK4.9.21 — HÍBRIDO ANN + FUNCIÓN BIMODAL
# Arquitectura:
#   1. Red Neuronal (ANN) genera la tasa base
#   2. Función Bimodal actúa como modulador de forma (multiplicativo)
#   3. Control "Fuerza de la Bimodal" para regular la influencia
#   4. Mantiene optimizador automático y todos los filtros Pergamino
# ===============================================================
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
import time
from datetime import timedelta
from pathlib import Path
import base64

try:
    from scipy.optimize import differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ---------------------------------------------------------
# CONFIGURACIÓN INICIAL
# ---------------------------------------------------------
if 'arranque_fase' not in st.session_state:
    st.set_page_config(page_title="PREDWEEM HÍBRIDO ANN+BIMODAL", layout="wide", page_icon="🌾")
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.info("🚜 **Iniciando PREDWEEM Híbrido (ANN + Bimodal)...**")
    st.progress(20)
    st.session_state.arranque_fase = 1
    time.sleep(0.1)
    st.rerun()

if 'arranque_fase' in st.session_state and st.session_state.arranque_fase == 1:
    st.session_state.arranque_fase = 2

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #dcfce7; border-right: 1px solid #bbf7d0; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p { color: #166534 !important; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .bio-alert { padding: 10px; border-radius: 5px; background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; margin-bottom: 10px; font-size: 0.9em; }
    .metric-header { color: #1e293b; font-weight: bold; margin-bottom: -10px; }
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

def set_bg_hack(main_bg_file):
    try:
        with open(main_bg_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(f"""<style>.stApp {{ background-image: url(data:image/png;base64,{encoded_string}); background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed; }}</style>""", unsafe_allow_html=True)
    except FileNotFoundError:
        pass
set_bg_hack("fondo_predweem_v3.png")

# ---------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------
def calcular_emergencia_bimodal(julian_day, offset, mean1, mean2, sigma1, sigma2, amp1, amp2):
    x = julian_day - offset
    term1 = amp1 * np.exp(-((x - mean1)**2) / (2 * sigma1**2))
    term2 = amp2 * np.exp(-((x - mean2)**2) / (2 * sigma2**2))
    return term1 + term2

def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na + 1, nb + 1), np.inf)
    dp[0, 0] = 0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[na, nb]

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit: return (t - t_base) * ((t_crit - t) / (t_crit - t_opt))
    else: return 0.0

def calcular_et0_hargreaves(jday, tmax, tmin, latitud=-33.9443):
    lat_rad = np.radians(latitud)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * jday)
    dec = 0.409 * np.sin(2 * np.pi / 365 * jday - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(dec))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(lat_rad) * np.sin(ws))
    ra_mm = ra / 2.45
    tmean = (tmax + tmin) / 2.0
    trange = np.maximum(tmax - tmin, 0)
    return np.maximum(0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(trange), 0)

def balance_hidrico_superficial(prec, et0, w_max=30.0, ke_suelo=0.4):
    n = len(prec)
    w = np.zeros(n)
    w[0] = w_max / 2.0
    for i in range(1, n):
        evaporacion_real = et0[i] * ke_suelo
        w[i] = max(0.0, min(w_max, w[i-1] + prec[i] - evaporacion_real))
    return w

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])
    def normalize(self, X): return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1
    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        a1 = np.tanh(Xn @ self.IW + self.bIW)
        emerrel = (np.tanh((a1 @ self.LW.T).flatten() + self.bLW) + 1) / 2
        return emerrel, np.cumsum(emerrel)

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(np.load(BASE / "IW.npy"), np.load(BASE / "bias_IW.npy"),
                                np.load(BASE / "LW.npy"), np.load(BASE / "bias_out.npy"))
        with open(BASE / "modelo_clusters_k3.pkl", "rb") as f: k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None

def load_data(file_uploader, default_name):
    if file_uploader:
        return pd.read_excel(file_uploader) if file_uploader.name.endswith((".xlsx", ".xls")) else pd.read_csv(file_uploader)
    elif (BASE / f"{default_name}.csv").exists():
        return pd.read_csv(BASE / f"{default_name}.csv")
    elif (BASE / f"{default_name}.xlsx").exists():
        return pd.read_excel(BASE / f"{default_name}.xlsx")
    try:
        return pd.read_csv(f"https://raw.githubusercontent.com/PREDWEEM/LOLIUM-PERGA2026/main/{default_name}.csv")
    except: return None

def sincronizar_series_flexibles(df_sim, df_campo, col_fecha, col_plm2, freq_dias=7):
    fecha_min = min(df_sim["Fecha"].min(), df_campo[col_fecha].min())
    fecha_max = max(df_sim["Fecha"].max(), df_campo[col_fecha].max())
    df_grid = pd.DataFrame({'Fecha': pd.date_range(start=fecha_min, end=fecha_max, freq='D')})
    df_sim_clean = df_sim[['Fecha', 'EMERREL']].copy()
    df_grid = pd.merge(df_grid, df_sim_clean, on='Fecha', how='left').fillna({'EMERREL': 0})
    df_grid['Sim_Acum_Abs'] = df_grid['EMERREL'].cumsum()
    df_campo_sorted = df_campo.sort_values(col_fecha).copy()
    df_campo_sorted['Campo_Acum_Abs'] = df_campo_sorted[col_plm2].cumsum()
    df_grid = pd.merge(df_grid, df_campo_sorted[[col_fecha, 'Campo_Acum_Abs']], left_on='Fecha', right_on=col_fecha, how='left')
    df_grid['Campo_Acum_Abs'] = df_grid['Campo_Acum_Abs'].interpolate(method='linear').fillna(0).ffill()
    df_grid.set_index('Fecha', inplace=True)
    df_resampled = df_grid.resample(f'{freq_dias}D').last().reset_index()
    df_resampled['Simulado_Intervalo'] = df_resampled['Sim_Acum_Abs'].diff().fillna(df_resampled['Sim_Acum_Abs'])
    df_resampled['Campo_Intervalo'] = df_resampled['Campo_Acum_Abs'].diff().fillna(df_resampled['Campo_Acum_Abs'])
    df_resampled['Simulado_Intervalo'] = df_resampled['Simulado_Intervalo'].clip(lower=0)
    df_resampled['Campo_Intervalo'] = df_resampled['Campo_Intervalo'].clip(lower=0)
    total_sim = df_resampled['Sim_Acum_Abs'].max() or 1
    total_campo = df_resampled['Campo_Acum_Abs'].max() or 1
    df_resampled['Sim_Relativo'] = df_resampled['Simulado_Intervalo'] / total_sim
    df_resampled['Campo_Relativo'] = df_resampled['Campo_Intervalo'] / total_campo
    df_resampled['Sim_Acumulado'] = df_resampled['Sim_Acum_Abs'] / total_sim
    df_resampled['Campo_Acumulado'] = df_resampled['Campo_Acum_Abs'] / total_campo
    return df_resampled

def calcular_metricas_validacion_integral(df_sync):
    mask_activos = (df_sync['Campo_Relativo'] > 0) | (df_sync['Sim_Relativo'] > 0)
    df_activos = df_sync[mask_activos].copy()
    if len(df_activos) < 2:
        return {"Pearson_Flujos": 0.0, "NSE_Flujos": 0.0, "KGE_Flujos": 0.0, "RMSE_Acumulado": 0.0, "CCC_Acumulado": 0.0}
    obs = df_activos['Campo_Relativo'].values
    sim = df_activos['Sim_Relativo'].values
    pearson_r = np.corrcoef(obs, sim)[0, 1] if np.std(obs) > 0 and np.std(sim) > 0 else 0.0
    var_obs_sum = np.sum((obs - np.mean(obs))**2)
    nse_flujos = 1 - (np.sum((sim - obs)**2) / var_obs_sum) if var_obs_sum > 0 else 0.0
    if np.mean(obs) > 0 and np.std(obs) > 0:
        r = pearson_r
        alpha = np.std(sim) / np.std(obs)
        beta = np.mean(sim) / np.mean(obs)
        kge_flujos = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    else:
        kge_flujos = 0.0
    obs_acum, sim_acum = df_sync['Campo_Acumulado'].values, df_sync['Sim_Acumulado'].values
    rmse_acumulado = np.sqrt(np.mean((obs_acum - sim_acum)**2))
    mean_obs_ac, mean_sim_ac = np.mean(obs_acum), np.mean(sim_acum)
    var_obs_ac, var_sim_ac = np.var(obs_acum), np.var(sim_acum)
    covar_ac = np.mean((obs_acum - mean_obs_ac) * (sim_acum - mean_sim_ac))
    denominador_ccc = var_obs_ac + var_sim_ac + (mean_obs_ac - mean_sim_ac)**2
    ccc_acumulado = (2 * covar_ac) / denominador_ccc if denominador_ccc > 0 else 0.0
    return {"Pearson_Flujos": pearson_r, "NSE_Flujos": nse_flujos, "KGE_Flujos": kge_flujos,
            "RMSE_Acumulado": rmse_acumulado, "CCC_Acumulado": ccc_acumulado}

# ---------------------------------------------------------
# INTERFAZ
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

st.title("🌾 PREDWEEM HÍBRIDO — ANN + Función Bimodal vK4.9.21")

with st.expander("📂 1. Datos del Lote", expanded=True):
    col_upload, col_rastrojo = st.columns(2)
    with col_upload:
        archivo_meteo = st.file_uploader("1. Clima (Pergamino)", type=["xlsx", "csv"])
        archivo_campo = st.file_uploader("2. Campo (Validación)", type=["xlsx", "csv"])
    with col_rastrojo:
        with st.container(border=True):
            st.markdown("#### 🌾 Manejo de Superficie")
            cobertura_pct = st.slider("Cobertura de Rastrojo (%)", 0, 100, 50, 5)
            x_cob = [0, 30, 70, 100]
            ke_val = float(np.interp(cobertura_pct, x_cob, [0.95, 0.50, 0.25, 0.10]))
            mod_termico = float(np.interp(cobertura_pct, x_cob, [1.00, 0.95, 0.90, 0.80]))
            st.caption(f"Ke = {ke_val:.2f}  |  Mod. Térmico = {mod_termico:.2f}")

# ===================== SIDEBAR =====================
st.sidebar.image("https://raw.githubusercontent.com/PREDWEEM/LOLIUM-PERGA2026/main/logo.png", use_container_width=True)
st.sidebar.markdown("## ⚙️ 2. Fisiología y Logística")

umbral_er = st.sidebar.slider("Umbral Tasa Diaria", 0.001, 0.80, 0.001, 0.001)
umbral_termoinhibicion = st.sidebar.number_input("Umbral Termoinhibición (°C)", 15.0, 35.0, 24.0, 0.5)
umbral_choque_hidrico = st.sidebar.slider("Choque Hídrico 3 días (mm)", 20.0, 100.0, 30.0)
residualidad = st.sidebar.number_input("Residualidad Herbicida (días)", 0, 60, 0)

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1: t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2: t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)
t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)

st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = st.sidebar.number_input("Objetivo Control", value=600, step=10)
dga_critico = st.sidebar.number_input("Límite Ventana", value=800, step=10)

st.sidebar.divider()
st.sidebar.markdown("## 💧 3. Balance Hídrico")
w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=30.0, step=1.0)

st.sidebar.divider()
st.sidebar.markdown("## 📊 4. Flexibilidad Estadística")
ventana_agrupacion = st.sidebar.slider("Ventana de Validación (días)", 1, 30, 11, 1)

# ===================== PARÁMETROS DE LA BIMODAL =====================
st.sidebar.divider()
st.sidebar.markdown("## 🔧 5. PARÁMETROS DE LA FUNCIÓN BIMODAL")

offset_bimodal = st.sidebar.number_input("Offset (días julianos)", min_value=50, max_value=150, value=94, step=1)
col_m1, col_m2 = st.sidebar.columns(2)
with col_m1:
    mean1 = st.sidebar.number_input("Media Pico 1", min_value=-40.0, max_value=80.0, value=-1.4, step=0.5)
with col_m2:
    mean2 = st.sidebar.number_input("Media Pico 2", min_value=-40.0, max_value=80.0, value=6.2, step=0.5)
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    sigma1 = st.sidebar.number_input("Ancho Pico 1 (sigma)", min_value=1.0, max_value=50.0, value=12.0, step=0.5)
with col_s2:
    sigma2 = st.sidebar.number_input("Ancho Pico 2 (sigma)", min_value=1.0, max_value=50.0, value=8.0, step=0.5)
col_a1, col_a2 = st.sidebar.columns(2)
with col_a1:
    amp1 = st.sidebar.number_input("Amplitud Pico 1", min_value=100.0, max_value=3000.0, value=900.0, step=10.0)
with col_a2:
    amp2 = st.sidebar.number_input("Amplitud Pico 2", min_value=100.0, max_value=3000.0, value=580.0, step=10.0)

# ===================== CONTROL HÍBRIDO =====================
st.sidebar.divider()
st.sidebar.markdown("## ⚖️ 6. CONTROL HÍBRIDO ANN + BIMODAL")

bimodal_weight = st.sidebar.slider(
    "Fuerza de la Bimodal",
    min_value=0.0,
    max_value=1.0,
    value=0.65,
    step=0.05,
    help="0.0 = Solo usa la Red Neuronal (ANN)\n"
         "1.0 = ANN × Bimodal (máxima influencia de la forma bimodal)\n"
         "0.5-0.8 suele ser un buen punto de partida"
)

st.sidebar.caption("La bimodal multiplica la salida de la ANN para forzar el patrón de forma.")

df_meteo_raw = load_data(archivo_meteo, "meteo_daily")
df_campo_raw = load_data(archivo_campo, "pergamino_campo")

# ---------------------------------------------------------
# MOTOR DE CÁLCULO HÍBRIDO
# ---------------------------------------------------------
if df_meteo_raw is not None and modelo_ann is not None:
    df = df_meteo_raw.copy()
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    df["Tmedia_aire"] = (df["TMAX"] + df["TMIN"]) / 2
    amplitud_termica = (df["TMAX"] - df["TMIN"]) / 2
    df["TMAX_suelo"] = df["Tmedia_aire"] + (amplitud_termica * mod_termico)
    df["TMIN_suelo"] = df["Tmedia_aire"] - (amplitud_termica * mod_termico)

    df_campo, col_fecha, col_plm2 = None, None, None
    if df_campo_raw is not None:
        df_campo = df_campo_raw.copy()
        col_fecha = 'FECHA' if 'FECHA' in df_campo.columns else df_campo.columns[0]
        col_plm2 = 'PLM2' if 'PLM2' in df_campo.columns else df_campo.columns[1]
        df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
        df_campo = df_campo.sort_values(col_fecha).reset_index(drop=True)
        max_plm2 = df_campo[col_plm2].max() or 1
        df_campo['Campo_Normalizado'] = df_campo[col_plm2] / max_plm2

    # === 1. PREDICCIÓN DE LA RED NEURONAL ===
    X = df[["Julian_days", "TMAX_suelo", "TMIN_suelo", "Prec"]].to_numpy(float)
    emerrel_ann, _ = modelo_ann.predict(X)
    emerrel_ann = np.maximum(emerrel_ann, 0.0)

    # === 2. FUNCIÓN BIMODAL (normalizada) ===
    raw_bimodal = np.array([
        calcular_emergencia_bimodal(jd, offset_bimodal, mean1, mean2, sigma1, sigma2, amp1, amp2)
        for jd in df["Julian_days"].values
    ])
    max_bim = np.max(raw_bimodal) if np.max(raw_bimodal) > 0 else 1.0
    bimodal_shape = raw_bimodal / max_bim

    # === 3. HÍBRIDO: ANN × Bimodal ===
    # bimodal_weight = 0 → solo ANN
    # bimodal_weight = 1 → ANN * Bimodal (máxima restricción de forma)
    emerrel_raw = emerrel_ann * (bimodal_shape ** bimodal_weight)
    df["EMERREL"] = np.maximum(emerrel_raw, 0.0)

    # === 4. FILTROS PERGAMINO (latencia, hídrico, térmico, etc.) ===
    df.loc[df["Julian_days"] <= 25, "EMERREL"] = 0.0

    df["Prec_3d"] = df["Prec"].rolling(window=3, min_periods=1).sum()
    mask_ruptura = (df["Julian_days"] <= 110) & (df["Prec_3d"] >= umbral_choque_hidrico)
    df.loc[mask_ruptura, "EMERREL"] = np.maximum(df.loc[mask_ruptura, "EMERREL"], 0.75)

    df["ET0"] = calcular_et0_hargreaves(df["Julian_days"].values, df["TMAX"].values, df["TMIN"].values, latitud=-33.9443)
    df["W_superficial"] = balance_hidrico_superficial(df["Prec"].values, df["ET0"].values, w_max=w_max_val, ke_suelo=ke_val)
    humedad_relativa = df["W_superficial"] / w_max_val
    df["Hydric_Factor"] = 1 / (1 + np.exp(-10 * (humedad_relativa - 0.3)))
    df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]
    df.loc[humedad_relativa < 0.20, "EMERREL"] = 0.0
    df['Lluvia_Recarga'] = (df['Prec'] >= w_max_val).cummax()
    df.loc[~df['Lluvia_Recarga'], "EMERREL"] = 0.0

    df["Tmedia"] = df["Tmedia_aire"]
    df["Tmedia_10d"] = df["Tmedia"].rolling(window=10, min_periods=1).mean()
    df.loc[df["Tmedia_10d"] >= umbral_termoinhibicion, "EMERREL"] = 0.0
    df["EMERREL"] = np.clip(df["EMERREL"], 0, 1.0)

    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    # Métricas y lógica de picos
    fecha_hoy = pd.Timestamp.now().normalize()
    if fecha_hoy not in df['Fecha'].values: fecha_hoy = df['Fecha'].max()

    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    dga_hoy, dga_7dias = 0.0, 0.0
    fecha_inicio_ventana, fecha_control = None, None
    if indices_pulso:
        fecha_inicio_ventana = df.loc[indices_pulso[0], "Fecha"]
        df_desde_pico = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_desde_pico["DGA_cum"] = df_desde_pico["DG"].cumsum()
        df_control = df_desde_pico[df_desde_pico["DGA_cum"] >= dga_optimo]
        if not df_control.empty: fecha_control = df_control.iloc[0]["Fecha"]
        dga_hoy = df.loc[(df["Fecha"] >= fecha_inicio_ventana) & (df["Fecha"] <= fecha_hoy), "DG"].sum()
        idx_hoy = df[df["Fecha"] == fecha_hoy].index[0]
        dga_7dias = dga_hoy + df.iloc[idx_hoy + 1: idx_hoy + 8]["DG"].sum() if idx_hoy + 8 <= len(df) else dga_hoy

    pearson_r = nse_flujos = kge_flujos = rmse_acum = ccc_acum = 0.0
    if df_campo is not None:
        df_sincronizado = sincronizar_series_flexibles(df, df_campo, col_fecha, col_plm2, freq_dias=ventana_agrupacion)
        metricas = calcular_metricas_validacion_integral(df_sincronizado)
        pearson_r, nse_flujos, kge_flujos, rmse_acum, ccc_acum = (
            metricas["Pearson_Flujos"], metricas["NSE_Flujos"], metricas["KGE_Flujos"],
            metricas["RMSE_Acumulado"], metricas["CCC_Acumulado"]
        )

    # VISUALIZACIÓN
    c_log = 0.01
    df["EMERREL_LOG"] = np.log10(df["EMERREL"] + c_log)
    umbral_er_log = np.log10(umbral_er + c_log)
    if df_campo is not None:
        df_campo['Campo_Normalizado_LOG'] = np.log10(df_campo['Campo_Normalizado'] + c_log)

    colorscale_hard = [[0.0, "green"], [0.01, "green"], [0.02, "red"], [1.0, "red"]]
    st.plotly_chart(go.Figure(data=go.Heatmap(z=[df["EMERREL"].values], x=df["Fecha"], y=["Emergencia"],
                                  colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False))
                    .update_layout(height=85, margin=dict(t=10, b=0, l=5, r=5)), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["📊 MONITOR HÍBRIDO", "💧 HIDRICO", "📈 VALIDACIÓN"])

    with tab1:
        if df_campo is not None:
            st.markdown(f"<p class='metric-header'>🚜 FIDELIDAD DE SIMULACIÓN (Flujos a {ventana_agrupacion} días)</p>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("KGE", f"{kge_flujos:.3f}")
            c2.metric("NSE", f"{nse_flujos:.3f}")
            c3.metric("CCC", f"{ccc_acum:.3f}")
            c4.metric("RMSE", f"{rmse_acum:.3f}")

        col_main, col_gauge = st.columns([2.3, 1])
        with col_main:
            fig_emer = go.Figure()
            fecha_actual = df["Fecha"].min()
            sombreado = True
            while fecha_actual < df["Fecha"].max():
                siguiente = fecha_actual + pd.Timedelta(days=ventana_agrupacion)
                if sombreado:
                    fig_emer.add_vrect(x0=fecha_actual, x1=siguiente, fillcolor="rgba(148,163,184,0.1)", layer="below", line_width=0)
                sombreado = not sombreado
                fecha_actual = siguiente

            fig_emer.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL_LOG"], mode='lines',
                                          name='Híbrido ANN×Bimodal (Log)', line=dict(color='#166534', width=2.5),
                                          fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.1)'))
            fig_emer.add_hline(y=umbral_er_log, line_dash="dash", line_color="orange", annotation_text=f"Umbral {umbral_er}")

            if df_campo is not None:
                fig_emer.add_trace(go.Scatter(x=df_campo[col_fecha], y=df_campo['Campo_Normalizado_LOG'],
                                              mode='markers+lines', name='Observado Campo (Log)',
                                              marker=dict(color='#dc2626', size=9, symbol='diamond'),
                                              line=dict(color='rgba(220,38,38,0.5)', dash='dot')))

            if fecha_control:
                fig_emer.add_vline(x=fecha_control.timestamp()*1000, line_dash="dot", line_color="red", line_width=3,
                                   annotation_text=f"Control {dga_optimo}°Cd")

            fig_emer.update_layout(title=f"Dinámica Híbrida (ANN × Bimodal, fuerza={bimodal_weight:.2f}) — Ventana {ventana_agrupacion}d",
                                   yaxis_title="Log10(Emergencia + 0.01)", height=450, hovermode="x unified",
                                   legend=dict(orientation="h", y=1.02, x=1))
            st.plotly_chart(fig_emer, use_container_width=True)

            if fecha_inicio_ventana:
                st.success(f"📅 Pico detectado: **{fecha_inicio_ventana.strftime('%d-%m-%Y')}**")
                if fecha_control:
                    st.error(f"🎯 Control recomendado: **{fecha_control.strftime('%d-%m-%Y')}** ({dga_optimo} °Cd)")

        with col_gauge:
            max_axis = dga_critico * 1.2
            st.plotly_chart(go.Figure().add_trace(go.Indicator(
                mode="gauge+number", value=dga_hoy,
                title={'text': "<b>TT ACUMULADO (°Cd)</b>"},
                gauge={'axis': {'range': [None, max_axis]},
                       'bar': {'color': "#1e293b", 'thickness': 0.3},
                       'steps': [{'range': [0, dga_optimo], 'color': "#4ade80"},
                                 {'range': [dga_optimo, dga_critico], 'color': "#facc15"},
                                 {'range': [dga_critico, max_axis], 'color': "#f87171"}],
                       'threshold': {'line': {'color': "#2563eb", 'width': 6}, 'value': dga_7dias}}))
                .add_annotation(x=0.5, y=-0.1, text=f"Pronóstico +7d: <b>{dga_7dias:.1f} °Cd</b>",
                                showarrow=False, font=dict(size=12)),
                use_container_width=True)

    with tab2:
        st.header("💧 Dinámica Hídrica del Suelo")
        fig_hidrico = go.Figure()
        fig_hidrico.add_trace(go.Bar(x=df["Fecha"], y=df["Prec"], name='Lluvia (mm)', marker_color='#93c5fd', opacity=0.7))
        fig_hidrico.add_trace(go.Scatter(x=df["Fecha"], y=df["W_superficial"], name='Agua en Suelo',
                                         mode='lines', line=dict(color='#0284c7', width=3),
                                         fill='tozeroy', fillcolor='rgba(2, 132, 199, 0.2)'))
        fig_hidrico.add_hline(y=w_max_val, line_dash="dot", line_color="#334155", annotation_text=f"Cap. Máx ({w_max_val} mm)")
        st.plotly_chart(fig_hidrico.update_layout(height=380), use_container_width=True)

    with tab3:
        st.header("📈 Validación vs Datos Observados")
        if df_campo is not None and 'df_sincronizado' in locals():
            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(x=df_sincronizado['Fecha'], y=df_sincronizado['Campo_Acumulado']*100,
                                         mode='markers+lines', name='Observado (%)',
                                         marker=dict(color='#dc2626', size=8, symbol='diamond')))
            fig_val.add_trace(go.Scatter(x=df_sincronizado['Fecha'], y=df_sincronizado['Sim_Acumulado']*100,
                                         mode='lines', name='Simulado Híbrido (%)',
                                         line=dict(color='#166534', width=3, dash='dash')))
            st.plotly_chart(fig_val.update_layout(title="Curvas Acumuladas - Híbrido ANN×Bimodal", height=380), use_container_width=True)
        else:
            st.info("Cargá datos de campo para ver la validación.")

    # Exportar
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Diaria')
        if df_campo is not None:
            df_campo.to_excel(writer, index=False, sheet_name='Campo')
        pd.DataFrame({
            'Parametro': ['Offset', 'Mean1', 'Mean2', 'Sigma1', 'Sigma2', 'Amp1', 'Amp2', 'Bimodal_Weight'],
            'Valor': [offset_bimodal, mean1, mean2, sigma1, sigma2, amp1, amp2, bimodal_weight]
        }).to_excel(writer, sheet_name='Config_Hibrida', index=False)
    st.sidebar.download_button("📥 Descargar Reporte Híbrido", output.getvalue(), "PREDWEEM_Hibrido_ANN_Bimodal.xlsx")

else:
    st.info("👋 Cargá archivos de clima y campo + asegurate de que los modelos ANN estén disponibles.")