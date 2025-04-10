import os
import json
import joblib
import torch
import numpy as np
from datetime import datetime
from preprocessing import clean_raw_data, create_features, prepare_training_data
from fetch_data import fetch_weather
from update_weather_data import update_weather_data
from torch import nn
import unicodedata

# ----- CONFIG -----
cities = ["Genève", "Lausanne", "Zurich", "Berne", "Bâle", "Neuchâtel",
          "Fribourg", "Lugano", "Lucerne", "Sion"]
target_columns = [
    "target_temperature_2m_max",
    "target_temperature_2m_min",
    "target_temperature_2m_mean",
    "target_will_rain",
    "target_windspeed_10m_max",
]

def sanitize_city(city: str) -> str:
    return unicodedata.normalize('NFD', city.lower()).encode('ascii', 'ignore').decode('utf-8').replace(" ", "_")

def get_model_path(relative_path: str) -> str:
    """Retourne le chemin absolu vers un fichier modèle en partant de app/"""
    base_dir = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(base_dir, "..", relative_path))

class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

def predict_city(city: str, day_offset: int = 0):
    day_str = "J+1" if day_offset else "J"
    print(f"\n Prédiction pour {city} ({day_str})")

    df = update_weather_data(city)
    df_clean = clean_raw_data(df)
    df_features = create_features(df_clean)
    X, _, scaler = prepare_training_data(df_features, target_columns)

    if day_offset == 0:
        X_last = X[-1:]
    elif day_offset == 1:
        if len(X) < 2:
            raise ValueError(f"Pas assez de données pour prédire J+1 à {city}")
        X_last = X[-2:-1]
    else:
        raise ValueError("day_offset invalide. Seulement 0 (J) ou 1 (J+1) autorisé.")
    if X_last.shape[0] == 0:
        raise ValueError(f"Aucune donnée suffisante pour la prédiction {day_str} à {city}")

    safe_city = sanitize_city(city)
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Lecture du fichier modèle
    model_info_path = get_model_path("model/best_model_info.json")
    with open(model_info_path, "r") as f:
        best_info = json.load(f)

    model_name = best_info.get("model_name")
    model_path = get_model_path(best_info.get("model_path"))

    if model_name == "cnn_lstm":
        n_total_features = X.shape[1]
        n_lags = 3
        if n_total_features % n_lags != 0:
            X_last = X_last[:, :-(n_total_features % n_lags)]
        n_features = X_last.shape[1] // n_lags
        model = WeatherLSTM(input_size=n_features)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        X_tensor = torch.tensor(X_last.reshape(1, n_lags, n_features), dtype=torch.float32)
        Y_pred = model(X_tensor).detach().numpy()[0]

    elif model_name == "blend":
        with open(model_path, "r") as f:
            blend_info = json.load(f)
        preds = []
        weights = blend_info["weights"]
        names = blend_info["models"]

        for name in names:
            if name == "cnn":
                cnn_path = get_model_path(f"model/{safe_city}_cnn.pt")
                n_total_features = X.shape[1]
                n_lags = 3
                if n_total_features % n_lags != 0:
                    X_last = X_last[:, :-(n_total_features % n_lags)]
                n_features = X_last.shape[1] // n_lags
                model = WeatherLSTM(input_size=n_features)
                model.load_state_dict(torch.load(cnn_path))
                model.eval()
                X_tensor = torch.tensor(X_last.reshape(1, n_lags, n_features), dtype=torch.float32)
                preds.append(model(X_tensor).detach().numpy()[0])
            else:
                path = get_model_path(f"model/{safe_city}_{name}.pkl")
                if os.path.exists(path):
                    m = joblib.load(path)
                    preds.append(m.predict(X_last)[0])

        if len(preds) != len(weights):
            print("  Nombre de poids incompatible avec les modèles chargés. Réajustement automatique.")
            weights = [1.0] * len(preds)

        Y_pred = np.average(preds, axis=0, weights=weights)

    else:
        model = joblib.load(model_path)
        Y_pred = model.predict(X_last)[0]

    result = dict(zip(target_columns, Y_pred))

    os.makedirs("predictions", exist_ok=True)
    suffix = "j1" if day_offset else "j"
    filename = f"predictions/{safe_city}_{date_str}_{suffix}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Prédiction enregistrée dans {filename}")
    return result

if __name__ == "__main__":
    import sys
    offset = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for city in cities:
        try:
            predict_city(city, day_offset=offset)
        except Exception as e:
            print(f"❌ Erreur pour {city} ({'J+1' if offset else 'J'}) : {e}")
