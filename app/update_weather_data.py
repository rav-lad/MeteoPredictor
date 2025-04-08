# update_weather_data.py

import os
import pandas as pd
from datetime import datetime
from fetch_data import fetch_weather


def update_weather_data(city: str, min_days_missing: int = 1) -> pd.DataFrame:
    safe_city = city.lower().replace(" ", "_")
    file_path = f"data/_{safe_city}_all.csv"

    # Charger données existantes
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_existing["time"] = pd.to_datetime(df_existing["time"])
        last_date = df_existing["time"].max().date()
    else:
        df_existing = pd.DataFrame()
        last_date = None

    today = pd.Timestamp.now().normalize().date()

    if last_date is None or last_date < today:
        # Nombre de jours manquants
        days_needed = (today - last_date).days if last_date else 365 * 2
        if days_needed >= min_days_missing:
            print(f"Mise à jour des données pour {city} ({days_needed} jours manquants)...")
            df_new = fetch_weather(city, past_days=days_needed, use_cache=False)

            if not df_existing.empty:
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates("time").sort_values("time")
            else:
                df_combined = df_new

            os.makedirs("data", exist_ok=True)
            df_combined.to_csv(file_path, index=False)
            print(f"Données mises à jour et sauvegardées dans {file_path}")
            return df_combined

    print(f"Données déjà à jour pour {city} ({last_date})")
    return df_existing


if __name__ == "__main__":
    SWISS_CITIES = [
        "Genève", "Lausanne", "Zurich", "Berne", "Bâle",
        "Neuchâtel", "Fribourg", "Lugano", "Lucerne", "Sion"
    ]

    for city in SWISS_CITIES:
        try:
            update_weather_data(city)
        except Exception as e:
            print(f" Erreur pour {city} : {e}")