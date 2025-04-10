# daily_update.py

from predict import predict_city
from update_weather_data import update_weather_data

# ----- LISTE DES VILLES -----
cities = [
    "Genève", "Lausanne", "Zurich", "Berne", "Bâle",
    "Neuchâtel", "Fribourg", "Lugano", "Lucerne", "Sion"
]

# ----- PIPELINE QUOTIDIEN -----
for city in cities:
    try:
        print(f"\n🌍 Traitement de {city}...")

        # Mise à jour des données (via update_weather_data.py)
        update_weather_data(city)

        # Prédiction pour aujourd'hui (J)
        predict_city(city, day_offset=0)

        # Prédiction pour demain (J+1)
        predict_city(city, day_offset=1)

    except Exception as e:
        print(f"❌ Erreur lors du traitement de {city} : {e}")
