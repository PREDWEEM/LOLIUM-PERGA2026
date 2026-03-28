import requests
import pandas as pd
import sys

# Coordenadas de Pergamino
LAT = -33.8895
LON = -60.5736

def obtener_pronostico():
    # Usamos el modelo seamless (por defecto) que une ECMWF + locales para evitar vacíos
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "America/Argentina/Buenos_Aires",
        "forecast_days": 7
    }
    
    print("Consultando a Open-Meteo...")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error en la API: {response.text}")
        sys.exit(1)
        
    data = response.json()
    
    # Crear DataFrame
    df = pd.DataFrame({
        'Fecha': data['daily']['time'],
        'TMAX': data['daily']['temperature_2m_max'],
        'TMIN': data['daily']['temperature_2m_min'],
        'Prec': data['daily']['precipitation_sum']
    })
    
    # --- FILTRO DE SEGURIDAD ---
    # Comprobamos si hay algún valor nulo/vacío en los datos descargados
    if df.isnull().values.any():
        print("ERROR: La API devolvió datos incompletos o vacíos.")
        print(df)
        # Salimos con error (exit 1) para que GitHub Actions aborte el proceso
        # y NO sobreescriba tu archivo actual con datos en blanco.
        sys.exit(1)
    
    # Si todo está bien, guardamos el archivo
    df.to_csv('pergamino_pronostico.csv', index=False)
    print("Archivo 'pergamino_pronostico.csv' actualizado exitosamente con estos datos:")
    print(df)

if __name__ == "__main__":
    obtener_pronostico()
