# update_weather_data.py

import os
import pandas as pd
from datetime import datetime, timedelta
from fetch_data import fetch_weather, get_coordinates
import requests


def fetch_forecast_j_minus_1(city: str) -> pd.DataFrame:
    """
    Récupère la météo de J-1 depuis l’API forecast (limitée à 7 jours).
    """
    latitude, longitude = get_coordinates(city)
    today = datetime.utcnow().date()
    j_minus_1 = today - timedelta(days=1)

    daily_vars = [
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
        "rain_sum", "showers_sum", "snowfall_sum", "windspeed_10m_max",
        "windgusts_10m_max", "sunshine_duration", "uv_index_max",
        "shortwave_radiation_sum", "et0_fao_evapotranspiration", "weathercode"
    ]

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&daily={','.join(daily_vars)}"
        f"&forecast_days=7"
        f"&timezone=auto"
    )

    response = requests.get(url)
    data = response.json()

    if "daily" not in data:
        raise ValueError("Pas de données forecast disponibles.")

    df = pd.DataFrame(data["daily"])
    df["city"] = city
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"].dt.date == j_minus_1]

    return df


def update_weather_data(city: str, min_days_missing: int = 1) -> pd.DataFrame:
    safe_city = city.lower().replace(" ", "_")
    file_path = f"data/_{safe_city}_all.csv"

    today = datetime.utcnow().date()
    j_minus_1 = today - timedelta(days=1)
    archive_cutoff = today - timedelta(days=5)

    # Lire fichier existant
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_existing["time"] = pd.to_datetime(df_existing["time"])
    else:
        df_existing = pd.DataFrame()

    df_combined = df_existing.copy()

    # Étape 1 : compléter avec archive jusqu’à J−5
    if df_existing.empty or df_existing["time"].max().date() < archive_cutoff:
        days_missing = (archive_cutoff - df_existing["time"].max().date()).days if not df_existing.empty else 730
        if days_missing >= min_days_missing:
            print(f"📦 Archive: {city} → {days_missing} jours manquants...")
            df_archive = fetch_weather(city, past_days=days_missing, use_cache=False)
            df_archive["time"] = pd.to_datetime(df_archive["time"])
            df_combined = pd.concat([df_combined, df_archive], ignore_index=True)

    # Étape 2 : ajouter J−1 si manquant
    if j_minus_1 not in df_combined["time"].dt.date.values:
        try:
            df_forecast = fetch_forecast_j_minus_1(city)
            if not df_forecast.empty:
                print(f"🔮 Forecast: {city} → ajout de J−1 ({j_minus_1})")
                df_combined = pd.concat([df_combined, df_forecast], ignore_index=True)
        except Exception as e:
            print(f"⚠️ Impossible de récupérer J−1 pour {city} : {e}")

    if df_combined.empty:
        print(f"Rien à mettre à jour pour {city}")
        return df_combined

    # Nettoyage final
    df_combined = df_combined.drop_duplicates("time").sort_values("time")
    os.makedirs("data", exist_ok=True)
    df_combined.to_csv(file_path, index=False)
    print(f"✅ Données mises à jour dans {file_path}")

    return df_combined


if __name__ == "__main__":
    SWISS_CITIES = [
        "Genève", "Lausanne", "Zurich", "Berne", "Bâle",
        "Neuchâtel", "Fribourg", "Lugano", "Lucerne", "Sion"
    ]

    # ----- PIPELINE QUOTIDIEN -----
    for city in cities:
        try:
            print(f"🌍 Traitement de {city}...")

            # Mise à jour des données
            update_weather_data(city)

            # Prédiction pour aujourd'hui (J)
            predict_city(city, day_offset=0)

            # Prédiction pour demain (J+1)
            predict_city(city, day_offset=1)

        except Exception as e:
            print(f"❌ Erreur lors du traitement de {city} : {e}")
