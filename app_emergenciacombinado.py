
# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM OPERATIVO vK4.9.10 — LOLIUM PERGAMINO 2026
# Actualización:
# - ADAPTACIÓN PERGAMINO: Coordenadas precisas actualizadas a LAT=-33.9443 y LON=-60.5745.
# - ET0: Cálculo de Hargreaves-Samani anclado estrictamente en -33.9443.
# - ESPECÍFICO PERGAMINO: Bypass por choque hídrico temprano limitado a 0.75.
# - ESPECÍFICO PERGAMINO: Modulador de agotamiento de banco de semillas y clip 0-1.
# - VISUALIZACIÓN LOGARÍTMICA: Transformación analítica log10(x + 0.01) en la gráfica principal.
# - MODO OPERATIVO: Sin módulo de validación de campo (optimizado para predicción rápida).
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

# ---------------------------------------------------------
# 1. PANTALLA DE CARGA Y CONFIGURACIÓN
# ---------------------------------------------------------
if 'arranque_fase' not in st.session_state:
    st.set_page_config(page_title="PREDWEEM PERGAMINO (Operativo)", layout="wide", page_icon="🌾")
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.info("🚜 **Iniciando Servidor PREDWEEM Operativo (Pergamino)...** Cargando motor predictivo.")
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stContainerBorder"],
    div[data-testid="stContainer"] > div > div[style*="border"],
    div[data-testid="stVerticalBlock"] > div[style*="border-radius"] { background-color: #ffffff !important; border-radius: 12px !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important; padding: 15px !important; border: 1px solid #e2e8f0 !important; }
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
# 2. ROBUSTEZ Y ARCHIVOS (MOCKS)
# ---------------------------------------------------------
def create_mock_files_if_missing():
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))

    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        p1 = np.exp(-((jd - 100) ** 2) / 600)
        p2 = np.exp(-((jd - 160) ** 2) / 900) + 0.3 * np.exp(-((jd - 260) ** 2) / 1200)
        p3 = np.exp(-((jd - 230) ** 2) / 1500)
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump({"JD_common": jd, "curves_interp": [p2, p1, p3], "medoids_k3": [0, 1, 2]}, f)

create_mock_files_if_missing()

# ---------------------------------------------------------
# 3. LÓGICA TÉCNICA Y PERGAMINO-SPECIFIC
# ---------------------------------------------------------
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
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(dec) * np.sin(ws))
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

def aplicar_patron_agotamiento(df, col_emer='EMERREL', patron=[0.640, 0.177, 0.137, 0.038, 0.008]):
    df_mod = df.copy()
    emer = df_mod[col_emer].values
    is_emerging = emer > 0.01
    cambios = np.diff(is_emerging.astype(int))
    inicios = np.where(cambios == 1)[0] + 1
    fines = np.where(cambios == -1)[0] + 1
    if is_emerging[0]: inicios = np.insert(inicios, 0, 0)
    if is_emerging[-1]: fines = np.append(fines, len(emer))
    suma_total_original = np.sum(emer)
    if suma_total_original == 0 or len(inicios) == 0: return df_mod
    nuevo_emer = np.zeros_like(emer)
    for idx, (ini, fin) in enumerate(zip(inicios, fines)):
        peso_objetivo = patron[idx] if idx < len(patron) else 0.0
        suma_bloque = np.sum(emer[ini:fin])
        if suma_bloque > 0:
            factor = (suma_total_original * peso_objetivo) / suma_bloque
            nuevo_emer[ini:fin] = emer[ini:fin] * factor
    df_mod[col_emer] = nuevo_emer
    return df_mod

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
        ann = PracticalANNModel(np.load(BASE / "IW.npy"), np.load(BASE / "bias_IW.npy"), np.load(BASE / "LW.npy"), np.load(BASE / "bias_out.npy"))
        with open(BASE / "modelo_clusters_k3.pkl", "rb") as f: k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None

def load_data(file_uploader=None):
    if file_uploader:
        return pd.read_excel(file_uploader) if file_uploader.name.endswith((".xlsx", ".xls")) else pd.read_csv(file_uploader)
    ruta_local = BASE / "meteo_daily.csv"
    if ruta_local.exists(): return pd.read_csv(ruta_local)
    github_url = "https://raw.githubusercontent.com/PREDWEEM/LOLIUM-PERGA2026/main/meteo_daily.csv"
    try: return pd.read_csv(github_url)
    except: return None

# ---------------------------------------------------------
# 4. INTERFAZ PRINCIPAL Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

st.title("🌾 PREDWEEM LOLIUM - PERGAMINO (BA) LAT = -33.9443 LON = -60.5745")

with st.expander("📂 1. Datos del Lote", expanded=True):
    col_upload, col_rastrojo = st.columns(2)
    
    with col_upload:
        archivo_meteo = st.file_uploader("Subir Clima Manual (Opcional)", type=["xlsx", "csv"], help="Si no subes nada, el sistema leerá automáticamente meteo_daily.csv de Pergamino")
        df_meteo_raw = load_data(archivo_meteo)
        if df_meteo_raw is not None:
            st.success("✅ Datos climáticos cargados.")
        else:
            st.error("❌ No se encontró 'meteo_daily.csv' ni se subió ningún archivo.")
            
    with col_rastrojo:
        with st.container(border=True):
            st.markdown("#### 🌾 Manejo de Superficie")
            cobertura_pct = st.slider("Cobertura de Rastrojo en Suelo (%)", min_value=0, max_value=100, value=85, step=5)
            x_cobertura = [0, 30, 70, 100]
            ke_val = float(np.interp(cobertura_pct, x_cobertura, [0.95, 0.50, 0.25, 0.10]))
            mod_termico = float(np.interp(cobertura_pct, x_cobertura, [1.00, 0.95, 0.90, 0.80]))

            html_card = f"""
            <div style="background-color:#fff; padding:15px 20px; border-radius:10px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.1); border:1px solid #e2e8f0; margin-top:15px;">
                <h5 style="color:#1e293b; margin-top:0; margin-bottom:12px; font-size:0.95rem;">Parámetros Dinámicos Aplicados</h5>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="color:#475569; font-size:0.9rem;">Coeficiente Hídrico Suelo (Ke):</span>
                    <span style="color:#0284c7; font-weight:bold; font-size:1.05rem;">{ke_val:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#475569; font-size:0.9rem;">Modulador Térmico Suelo:</span>
                    <span style="color:#b91c1c; font-weight:bold; font-size:1.05rem;">{mod_termico:.2f}</span>
                </div>
            </div>
            """
            st.markdown(html_card, unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.image("https://raw.githubusercontent.com/PREDWEEM/LOLIUM-PERGA2026/main/logo.png", use_container_width=True)
st.sidebar.markdown("## ⚙️ 2. Fisiología y Logística")
umbral_er = st.sidebar.slider("Umbral Alerta Temprana", 0.01, 0.80, 0.01)
umbral_termoinhibicion = st.sidebar.number_input("Umbral Termoinhibición (°C)", 15.0, 35.0, 24.0, 0.5)
umbral_choque_hidrico = st.sidebar.slider("Choque Hídrico 3 días (mm)", 20.0, 100.0, 30.0)
residualidad = st.sidebar.number_input("Residualidad Herbicida (días)", 0, 60, 20)
col_t1, col_t2 = st.sidebar.columns(2)
with col_t1: t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2: t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)
t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)
st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = st.sidebar.number_input("Objetivo Control", value=600, step=10)
dga_critico = st.sidebar.number_input("Límite Ventana", value=800, step=10)
st.sidebar.divider()
st.sidebar.markdown("## 💧 3. Balance Hídrico (Suelo)")
w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=30.0, step=1.0)

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO
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

    X = df[["Julian_days", "TMAX_suelo", "TMIN_suelo", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel_raw, 0.0)

    # Bypass Ruptura Temprana (PERGAMINO: Limitado a 0.75)
    df["Prec_3d"] = df["Prec"].rolling(window=3, min_periods=1).sum()
    mask_ruptura = (df["Julian_days"] <= 110) & (df["Prec_3d"] >= umbral_choque_hidrico)
    df.loc[mask_ruptura, "EMERREL"] = np.maximum(df.loc[mask_ruptura, "EMERREL"], 0.75)

    # Balance Hídrico Superficial (Pergamino: Lat=-33.9443)
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

    # PERGAMINO: Patrón de Agotamiento y Techo 0-1
    df = aplicar_patron_agotamiento(df)
    df["EMERREL"] = np.clip(df["EMERREL"], 0, 1.0)

    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))
    fecha_hoy = pd.Timestamp.now().normalize()
    if fecha_hoy not in df['Fecha'].values: fecha_hoy = df['Fecha'].max()
    
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    dga_hoy, dga_7dias, dias_stress = 0.0, 0.0, 0
    fecha_inicio_ventana, fecha_control = None, None
    msg_estado = "Esperando pico de emergencia..."

    if indices_pulso:
        fecha_inicio_ventana = df.loc[indices_pulso[0], "Fecha"]
        df_desde_pico = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_desde_pico["DGA_cum"] = df_desde_pico["DG"].cumsum()
        df_control = df_desde_pico[df_desde_pico["DGA_cum"] >= dga_optimo]
        if not df_control.empty: fecha_control = df_control.iloc[0]["Fecha"]
        dga_hoy = df.loc[(df["Fecha"] >= fecha_inicio_ventana) & (df["Fecha"] <= fecha_hoy), "DG"].sum()
        idx_hoy = df[df["Fecha"] == fecha_hoy].index[0]
        dga_7dias = dga_hoy + df.iloc[idx_hoy + 1: idx_hoy + 8]["DG"].sum() if idx_hoy + 8 <= len(df) else dga_hoy
        msg_estado = f"Pico detectado el {fecha_inicio_ventana.strftime('%d/%m')}"
        dias_stress = len(df_desde_pico[df_desde_pico["Tmedia"] > t_opt_max])

    # -----------------------------------------------------
    # TRANSFORMACIÓN LOGARÍTMICA (Opción Analítica)
    # -----------------------------------------------------
    c_log = 0.01
    df["EMERREL_LOG"] = np.log10(df["EMERREL"] + c_log)
    umbral_er_log = np.log10(umbral_er + c_log)

    # -----------------------------------------------------
    # VISUALIZACIÓN FRONT-END
    # -----------------------------------------------------
    colorscale_hard = [[0.0, "green"], [0.01, "green"], [0.02, "red"], [1.0, "red"]]
    st.plotly_chart(go.Figure(data=go.Heatmap(z=[df["EMERREL"].values], x=df["Fecha"], y=["Emergencia"], colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False)).update_layout(height=120, margin=dict(t=30, b=0, l=10, r=10), title="Mapa de Riesgo (Tasa Diaria)"), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 MONITOR DE DECISIÓN", "💧 PRECIPITACIONES Y SUELO", "📈 ANÁLISIS ESTRATÉGICO", "🧪 BIO-CALIBRACIÓN"])

    with tab1:
        col_main, col_gauge = st.columns([2, 1])

        with col_main:
            fig_emer = go.Figure()
            fig_emer.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL_LOG"], mode='lines', name='Tasa Diaria Sim. (Log)', line=dict(color='#166534', width=2.5), fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.1)'))
            fig_emer.add_hline(y=umbral_er_log, line_dash="dash", line_color="orange", annotation_text=f"Umbral Alerta ({umbral_er})")

            if fecha_control:
                fig_emer.add_vline(x=fecha_control.timestamp() * 1000, line_dash="dot", line_color="red", line_width=3, annotation_text=f"Control ({dga_optimo}°Cd)", annotation_position="top left", annotation_font=dict(color="red", size=12))
                fig_emer.add_vrect(x0=fecha_control.timestamp() * 1000, x1=(fecha_control + timedelta(days=residualidad)).timestamp() * 1000, fillcolor="blue", opacity=0.1, layer="below", line_width=0, annotation_text=f"Protección ({residualidad}d)", annotation_position="top left")

            fig_emer.update_layout(title="Dinámica de Emergencia y Momento Crítico (Escala Log Analítica)", yaxis_title="Log10(Emergencia + 0.01)", height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_emer, use_container_width=True)

            if fecha_inicio_ventana:
                st.success(f"📅 **Inicio de Conteo Térmico:** {fecha_inicio_ventana.strftime('%d-%m-%Y')} (Primer pico detectado)")
                if dias_stress > 0: st.markdown(f"""<div class="bio-alert">🔥 <b>Estrés Térmico:</b> {dias_stress} días con T > {t_opt_max}°C desde el inicio.</div>""", unsafe_allow_html=True)
                if fecha_control: st.error(f"🎯 **MOMENTO CRÍTICO DE CONTROL:** {fecha_control.strftime('%d-%m-%Y')}. Se acumularon **{dga_optimo} °Cd** post-emergencia.")
                else: st.info(f"⏳ **En Progreso:** Aún no se han acumulado los {dga_optimo} °Cd requeridos para el control.")
            else: st.warning(f"⏳ Esperando primera alerta (Tasa >= {umbral_er}). El perfil necesita recargarse hasta alcanzar la Capacidad de Campo ({w_max_val} mm) en un solo evento.")

        with col_gauge:
            max_axis = dga_critico * 1.2
            st.plotly_chart(go.Figure().add_trace(go.Indicator(mode="gauge+number", value=dga_hoy, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "<b>TT ACUMULADO (°Cd)</b>", 'font': {'size': 18}}, gauge={'axis': {'range': [None, max_axis]}, 'bar': {'color': "#1e293b", 'thickness': 0.3}, 'steps': [{'range': [0, dga_optimo], 'color': "#4ade80"}, {'range': [dga_optimo, dga_critico], 'color': "#facc15"}, {'range': [dga_critico, max_axis], 'color': "#f87171"}], 'threshold': {'line': {'color': "#2563eb", 'width': 6}, 'thickness': 0.8, 'value': dga_7dias}})).add_annotation(x=0.5, y=-0.1, text=f"{msg_estado}<br>Pronóstico +7d: <b>{dga_7dias:.1f} °Cd</b>", showarrow=False, font=dict(size=14, color="#1e3a8a"), align="center").update_layout(height=350, margin=dict(t=80, b=50, l=30, r=30)), use_container_width=True)

    with tab2:
        st.header("💧 Dinámica Hídrica del Suelo")
        fig_hidrico = go.Figure()
        fig_hidrico.add_trace(go.Bar(x=df["Fecha"], y=df["Prec"], name='Lluvia Diaria (mm)', marker_color='#93c5fd', opacity=0.7))
        fig_hidrico.add_trace(go.Scatter(x=df["Fecha"], y=df["W_superficial"], name='Agua en Suelo (0-10cm)', mode='lines', line=dict(color='#0284c7', width=3), fill='tozeroy', fillcolor='rgba(2, 132, 199, 0.2)'))
        fig_hidrico.add_hline(y=w_max_val, line_dash="dot", line_color="#334155", annotation_text=f"Capacidad Máx. ({w_max_val} mm)", annotation_position="top left")
        st.plotly_chart(fig_hidrico.update_layout(title="Precipitación vs. Retención Real de Humedad", xaxis_title="Fecha", yaxis_title="Milímetros (mm)", height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)), use_container_width=True)

    with tab3:
        st.header("🔍 Clasificación DTW (Pergamino)")
        df_obs = df[df["Fecha"] < pd.Timestamp("2026-05-01")].copy()
        if not df_obs.empty and df_obs["EMERREL"].sum() > 0:
            jd_corte = df_obs["Julian_days"].max()
            max_e = df_obs["EMERREL"].max() if df_obs["EMERREL"].max() > 0 else 1.0
            JD_COM = cluster_model["JD_common"]
            jd_grid = JD_COM[JD_COM <= jd_corte]
            obs_norm = np.interp(jd_grid, df_obs["Julian_days"], df_obs["EMERREL"] / max_e)
            dists = [dtw_distance(obs_norm, m[JD_COM <= jd_corte] / m[JD_COM <= jd_corte].max() if m[JD_COM <= jd_corte].max() > 0 else m[JD_COM <= jd_corte]) for m in cluster_model["curves_interp"]]
            pred = int(np.argmin(dists))
            cols = {0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}
            c1, c2 = st.columns([3, 1])
            with c1:
                fp = go.Figure()
                fp.add_trace(go.Scatter(x=JD_COM, y=cluster_model["curves_interp"][pred], name="Patrón Histórico", line=dict(dash='dash', color=cols.get(pred))))
                fp.add_trace(go.Scatter(x=jd_grid, y=obs_norm * cluster_model["curves_interp"][pred].max(), name="2026", line=dict(color='black', width=3)))
                st.plotly_chart(fp, use_container_width=True)
            with c2:
                nombres_patrones = {0: "🌾 Bimodal", 1: "🌱 Temprano", 2: "🍂 Tardío"}
                st.success(f"### {nombres_patrones.get(pred, 'Desconocido')}")
                st.metric("DTW Score", f"{min(dists):.2f}")
        else: st.info("Datos insuficientes para clasificación DTW.")

    with tab4:
        st.subheader("🧪 Curva de Respuesta Fisiológica")
        x_temps = np.linspace(0, 45, 200)
        st.plotly_chart(go.Figure().add_trace(go.Scatter(x=x_temps, y=[calculate_tt_scalar(t, t_base_val, t_opt_max, t_critica) for t in x_temps], mode='lines', line=dict(color='#2563eb', width=4), fill='tozeroy')), use_container_width=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Diaria')
        pd.DataFrame({'Configuracion': ['T_Base', 'T_Optima', 'T_Critica', 'W_Max', 'Ke', 'Mod_Termico', 'Umbral_Termoinhibicion'], 'Valor': [t_base_val, t_opt_max, t_critica, w_max_val, ke_val, mod_termico, umbral_termoinhibicion]}).to_excel(writer, sheet_name='Bio_Params', index=False)

    st.sidebar.download_button("📥 Descargar Reporte Completo", output.getvalue(), "PREDWEEM_Operativo_Pergamino_vK4_9_10_clean.xlsx")

else:
    st.info("👋 Bienvenido a PREDWEEM. El sistema está esperando los datos climáticos para comenzar.")
