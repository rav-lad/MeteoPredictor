# test_model.py

import os
import json
import joblib
import numpy as np
from fetch_data import fetch_weather
from preprocessing import clean_raw_data, create_features, prepare_training_data
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# ----- CONFIGURATION -----
city = "Genève"
past_days = 730
n_lags = 3
target_columns = (
    "target_temperature_2m_max",
    "target_temperature_2m_min",
    "target_temperature_2m_mean",
    "target_will_rain",
    "target_windspeed_10m_max",
)

# ----- CHARGEMENT DU MEILLEUR MODÈLE -----
with open("model/best_model_info.json") as f:
    model_info = json.load(f)
model_path = model_info["model_path"]
model_type = model_info.get("model_name", "")

print(f"Chargement du meilleur modèle : {model_type} ({model_path})")

# ----- PRÉPARATION DES DONNÉES -----
print("Chargement et preprocessing...")
df = fetch_weather(city, past_days=past_days)
df_clean = clean_raw_data(df)
df_features = create_features(df_clean, n_lags=n_lags)
X, Y, _ = prepare_training_data(df_features, target_columns=target_columns)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ----- PRÉDICTIONS -----
print("Évaluation sur les données de test...")

if model_type == "blended":
    blend = joblib.load(model_path)
    all_preds = []
    for model_file in blend["models"]:
        model = joblib.load(f"model/{model_file}")
        pred = model.predict(X_test)
        all_preds.append(pred)
    Y_pred = np.tensordot(blend["weights"], np.array(all_preds), axes=([0], [0]))
else:
    model = joblib.load(model_path)
    Y_pred = model.predict(X_test)

for i, col in enumerate(target_columns):
    y_true = Y_test[:, i]
    y_hat = Y_pred[:, i]

    if col == "target_will_rain":
        acc = accuracy_score(y_true, np.round(y_hat))
        f1 = f1_score(y_true, np.round(y_hat))
        print(f"{col} - Accuracy: {acc:.2f} | F1-score: {f1:.2f}")
        print(classification_report(y_true, np.round(y_hat)))
    else:
        mae = mean_absolute_error(y_true, y_hat)
        print(f"{col} - MAE: {mae:.2f}")
