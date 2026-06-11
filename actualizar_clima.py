
import requests
import pandas as pd
import sys
import os

# Coordenadas específicas de PERGAMINO, Provincia de Buenos Aires
LAT = -33.9443
LON = -60.5745
ARCHIVO_CSV = 'meteo_daily.csv'

def actualizar_pronostico():
    url = "https://api.open-meteo.com/v1/forecast"
    
    # ESTRATEGIA DE REANÁLISIS CONTINUO:
    # Captura 7 días hacia atrás (datos reales observados) y 7 días de pronóstico predictivo.
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "America/Argentina/Buenos_Aires",
        "past_days": 7,
        "forecast_days": 7
    }
    
    print("Consultando a Open-Meteo para Pergamino (Ventana Híbrida: -7d a +7d)...")
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"Error en la API: {response.text}")
            sys.exit(1)
    except Exception as e:
        print(f"Error de conexión con la API: {e}")
        sys.exit(1)
        
    data = response.json()
    
    # DataFrame con el bloque temporal de 14 días para Pergamino
    df_nuevo = pd.DataFrame({
        'Fecha': data['daily']['time'],
        'TMAX': data['daily']['temperature_2m_max'],
        'TMIN': data['daily']['temperature_2m_min'],
        'Prec': data['daily']['precipitation_sum']
    })
    
    # Forzar parseo a datetime para evitar inconsistencias de tipos string al concatenar
    df_nuevo['Fecha'] = pd.to_datetime(df_nuevo['Fecha'])
    
    if df_nuevo.isnull().values.any():
        print("ADVERTENCIA: Datos incompletos para Pergamino. Aplicando forward-fill temporal.")
        df_nuevo = df_nuevo.ffill()

    # Integración con el archivo de la serie histórica actual
    if os.path.exists(ARCHIVO_CSV):
        print(f"Leyendo historial desde {ARCHIVO_CSV}...")
        df_historico = pd.read_csv(ARCHIVO_CSV)
        df_historico['Fecha'] = pd.to_datetime(df_historico['Fecha'])
        
        # Unión de estructuras de datos
        df_final = pd.concat([df_historico, df_nuevo], ignore_index=True)
        
        # CONTROL DE CALIDAD SINOÓPTICA:
        # 'keep=last' descarta las filas de pronóstico predictivo viejo y conserva los registros 
        # actualizados que Open-Meteo corrigió tras consolidar los modelos globales y satelitales.
        df_final = df_final.drop_duplicates(subset=['Fecha'], keep='last')
        df_final = df_final.sort_values(by='Fecha').reset_index(drop=True)
    else:
        print(f"No se encontró {ARCHIVO_CSV}, inicializando nuevo registro para Pergamino...")
        df_final = df_nuevo

    # Persistencia en disco con formato estricto ISO (YYYY-MM-DD)
    df_final['Fecha'] = df_final['Fecha'].dt.strftime('%Y-%m-%d')
    df_final.to_csv(ARCHIVO_CSV, index=False)
    
    print("Base meteorológica de Pergamino actualizada y depurada. Últimos 10 registros:")
    print(df_final.tail(10))

if __name__ == "__main__":
    actualizar_pronostico()
