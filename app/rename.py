import os
import unicodedata

def sanitize_city(city: str) -> str:
    return unicodedata.normalize('NFD', city.lower()).encode('ascii', 'ignore').decode('utf-8').replace(" ", "_")

# Résout correctement le chemin du dossier "model"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "..", "model")

# Villes cibles
cities = ["Genève", "Lausanne", "Zurich", "Berne", "Bâle", "Neuchâtel",
          "Fribourg", "Lugano", "Lucerne", "Sion"]

rename_map = {city.lower().replace(" ", "_"): sanitize_city(city) for city in cities}

# Renommage des fichiers
for filename in os.listdir(model_dir):
    for old_prefix, new_prefix in rename_map.items():
        if filename.startswith(old_prefix + "_"):
            old_path = os.path.join(model_dir, filename)
            new_filename = filename.replace(old_prefix, new_prefix, 1)
            new_path = os.path.join(model_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"✅ Renommé : {filename} → {new_filename}")
            break
