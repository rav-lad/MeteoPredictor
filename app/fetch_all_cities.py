# fetch_all_cities.py

import time
from fetch_data import fetch_weather

# ----- CONFIGURATION -----
cities = ["Genève", "Lausanne", "Zurich", "Berne", "Bâle", "Neuchâtel", "Fribourg", "Lugano", "Lucerne", "Sion"]
past_days = 730
max_requests_per_minute = 2000
pause_seconds = 60 / max_requests_per_minute


# ----- FETCH TOUTES LES VILLES -----
for city in cities:
    try:
        print(f"Téléchargement de {city}...")
        fetch_weather(city, past_days=past_days, use_cache=False)
        print(f"{city} téléchargée avec succès.")
    except Exception as e:
        print(f"Erreur pour {city} : {e}")
    time.sleep(pause_seconds)