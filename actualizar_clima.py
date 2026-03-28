
import requests
import pandas as pd
from datetime import datetime

# Coordenadas de Pergamino
LAT = -33.8895
LON = -60.5736

def obtener_pronostico():
    # Usamos el modelo ECMWF (Europeo) para máxima precisión en lluvias
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "America/Argentina/Buenos_Aires",
        "models": "ecmwf_ifs04", # Modelo de alta resolución
        "forecast_days": 7
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Crear DataFrame con el formato solicitado
    df = pd.DataFrame({
        'Fecha': data['daily']['time'],
        'TMAX': data['daily']['temperature_2m_max'],
        'TMIN': data['daily']['temperature_2m_min'],
        'Prec': data['daily']['precipitation_sum']
    })
    
    # Guardar archivo
    df.to_csv('pergamino_pronostico.csv', index=False)
    print(f"Actualizado el: {datetime.now()}")

if __name__ == "__main__":
    obtener_pronostico()
