import os
import json
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
from fetch_data import fetch_weather
from preprocessing import clean_raw_data, create_features, prepare_training_data

# ----- CONFIGURATION -----
cities = ["Genève", "Lausanne", "Zurich", "Berne", "Bâle", "Neuchâtel", "Fribourg", "Lugano", "Lucerne", "Sion"]
past_days = 730
n_lags = 3
target_columns = [
    "target_temperature_2m_max",
    "target_temperature_2m_min",
    "target_temperature_2m_mean",
    "target_will_rain",
    "target_windspeed_10m_max"
]

def blend_predictions(preds, weights):
    weights = np.array(weights)
    weights = weights / weights.sum()
    return np.average(preds, axis=0, weights=weights)


for city in cities:
    try:
        print(f"\n Blending pour {city}")
        df = fetch_weather(city, past_days=past_days)
        df_clean = clean_raw_data(df)
        df_features = create_features(df_clean, n_lags=n_lags)
        X, Y, _ = prepare_training_data(df_features, target_columns)

        X_last = X[-1:]
        Y_true = Y[-1]
        safe_city = city.lower().replace(" ", "_")

        preds = []
        names = []
        weights = []

        model_files = [
            (f"model/{safe_city}_model.pkl", "rf"),
            (f"model/{safe_city}_xgb.pkl", "xgb"),
            (f"model/{safe_city}_lgbm.pkl", "lgbm"),
            (f"model/{safe_city}_cat.pkl", "cat"),
        ]

        for path, name in model_files:
            if os.path.exists(path):
                try:
                    model = joblib.load(path)
                    pred = model.predict(X_last)[0]
                    preds.append(pred)
                    names.append(name)
                    weights.append(1.0)
                except Exception as e:
                    print(f"Erreur {name} : {e}")

        # CNN LSTM
        cnn_path = f"model/{safe_city}_cnn.pt"
        if os.path.exists(cnn_path):
            try:
                import torch
                from torch import nn

                class WeatherLSTM(nn.Module):
                    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=5):
                        super().__init__()
                        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                        self.fc = nn.Linear(hidden_size, output_size)

                    def forward(self, x):
                        out, _ = self.lstm(x)
                        return self.fc(out[:, -1])

                n_total_features = X.shape[1]
                while n_total_features % n_lags != 0:
                    X_last = X_last[:, :-1]
                    n_total_features -= 1

                n_features = n_total_features // n_lags
                model = WeatherLSTM(input_size=n_features)
                model.load_state_dict(torch.load(cnn_path, weights_only=True))
                model.eval()
                X_tensor = torch.tensor(X_last.reshape(1, n_lags, n_features), dtype=torch.float32)
                pred_cnn = model(X_tensor).detach().numpy()[0]
                preds.append(pred_cnn)
                names.append("cnn")
                weights.append(1.0)
            except Exception as e:
                print(f" Erreur CNN : {e}")

        if len(preds) < 2:
            print("Pas assez de modèles pour blender")
            continue

        Y_blend = blend_predictions(preds, weights)
        mae = mean_absolute_error(Y_true, Y_blend)
        print(f"MAE du blend : {mae:.2f}")

        # Vérifier si meilleur que le modèle global
        best_info_path = "model/best_model_info.json"
        if os.path.exists(best_info_path):
            with open(best_info_path, "r") as f:
                best_info = json.load(f)
            current_best_score = best_info.get("score", float("inf"))
        else:
            current_best_score = float("inf")

        if mae < current_best_score:
            with open(best_info_path, "w") as f:
                json.dump({
                    "model_name": "blend",
                    "model_path": f"model/{safe_city}_blend.json",
                    "score": mae
                }, f)
            print("Nouveau meilleur modèle global (Blend)")

        os.makedirs("model", exist_ok=True)
        with open(f"model/{safe_city}_blend.json", "w") as f:
            json.dump({"models": names, "weights": weights}, f)

    except Exception as e:
        print(f"Erreur pour {city} : {e}")
