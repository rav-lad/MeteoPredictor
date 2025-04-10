# update_weather_data.py

import os
import pandas as pd
from datetime import datetime
from fetch_data import fetch_weather

def update_weather_data(city: str, min_days_missing: int = 1) -> pd.DataFrame:
    safe_city = city.lower().replace(" ", "_")
    file_path = f"data/_{safe_city}_all.csv"

    # Charger les données d'archive (en cache) pour les 730 derniers jours
    try:
        df_archive = fetch_weather(city, past_days=730, use_cache=True)
        df_archive["time"] = pd.to_datetime(df_archive["time"])
    except Exception as e:
        print(f"⚠️ Erreur chargement des données archive pour {city} : {e}")
        df_archive = pd.DataFrame()

    # Charger les données existantes (par ex. avec J-1 issues d'un précédent update)
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_existing["time"] = pd.to_datetime(df_existing["time"])
    else:
        df_existing = pd.DataFrame()

    # Fusionner les données archivées et existantes, en éliminant les doublons et en triant par date
    df_combined = pd.concat([df_archive, df_existing], ignore_index=True)
    df_combined = df_combined.drop_duplicates("time").sort_values("time")

    today = pd.Timestamp.now().normalize().date()
    # Déterminer la dernière date disponible dans les données combinées
    last_date = df_combined["time"].max().date() if not df_combined.empty else None

    # Si aucune donnée ou la dernière date est antérieure à aujourd'hui, récupérer les jours manquants
    if last_date is None or last_date < today:
        # Calculer le nombre de jours manquants
        days_needed = (today - last_date).days if last_date else 365 * 2
        if days_needed >= min_days_missing:
            print(f"Mise à jour des données pour {city} ({days_needed} jours manquants)...")
            try:
                df_new = fetch_weather(city, past_days=days_needed, use_cache=False)
                df_new["time"] = pd.to_datetime(df_new["time"])
                df_combined = pd.concat([df_combined, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates("time").sort_values("time")
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement des nouvelles données pour {city} : {e}")
                return df_combined

            os.makedirs("data", exist_ok=True)
            df_combined.to_csv(file_path, index=False)
            print(f"Données mises à jour et sauvegardées dans {file_path}")
            return df_combined

    print(f"✅ Données déjà à jour pour {city} (jusqu'à {last_date})")
    return df_combined

if __name__ == "__main__":
    SWISS_CITIES = [
        "Genève", "Lausanne", "Zurich", "Berne", "Bâle",
        "Neuchâtel", "Fribourg", "Lugano", "Lucerne", "Sion"
    ]

    for city in SWISS_CITIES:
        try:
            update_weather_data(city)
        except Exception as e:
            print(f"❌ Erreur pour {city} : {e}")
