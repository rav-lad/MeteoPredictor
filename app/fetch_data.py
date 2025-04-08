# fetch_data.py

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

def get_coordinates(city_name: str):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1"
    response = requests.get(url)
    data = response.json()

    if "results" not in data:
        raise ValueError(f"Ville '{city_name}' introuvable.")

    result = data["results"][0]
    return result["latitude"], result["longitude"]


def fetch_weather(city_name: str, past_days: int = 730, use_cache: bool = True):
    """
    Récupère les données météo pour une ville. Si un fichier .csv existe localement, l’utilise.
    
    Les fichiers sont enregistrés sous la forme : data/_genève_730d.csv
    """
    os.makedirs("data", exist_ok=True)
    safe_city_name = city_name.lower().replace(" ", "_")
    filename = f"data/_{safe_city_name}_{past_days}d.csv"

    # Utiliser les données locales si elles existent
    if use_cache and os.path.exists(filename):
        print(f"Chargement des données locales depuis {filename}")
        return pd.read_csv(filename)

    print("Récupération des données depuis l’API...")
    latitude, longitude = get_coordinates(city_name)

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=past_days)

    daily_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "rain_sum",
        "showers_sum",
        "snowfall_sum",
        "windspeed_10m_max",
        "windgusts_10m_max",
        "sunshine_duration",
        "uv_index_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        "weathercode"
    ]

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily={','.join(daily_vars)}"
        f"&timezone=auto"
    )

    response = requests.get(url)
    data = response.json()

    if "daily" not in data:
        raise ValueError("Pas de données météo disponibles pour cette période.")

    df = pd.DataFrame(data["daily"])
    df["city"] = city_name

    # Sauvegarde
    df.to_csv(filename, index=False)
    print(f"💾 Données sauvegardées dans {filename}")
    return df
