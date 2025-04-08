# train_cnn.py (multi-ville)

import os
import json
import torch
import joblib
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from fetch_data import fetch_weather
from preprocessing import clean_raw_data, create_features, prepare_training_data
from sklearn.metrics import mean_absolute_error

# ----- CONFIGURATION -----
cities = ["Genève", "Lausanne", "Zurich", "Berne", "Bâle", "Neuchâtel", "Fribourg", "Lugano", "Lucerne", "Sion"]
past_days = 730
n_lags = 3
epochs = 30
batch_size = 32
learning_rate = 1e-3
train_ratio = 0.8

# ----- LSTM MODEL -----
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


# ----- ENTRAÎNEMENT PAR VILLE -----
for city in cities:
    try:
        print(f"\n LSTM pour {city}")
        df = fetch_weather(city, past_days=past_days)
        df_clean = clean_raw_data(df)
        df_features = create_features(df_clean, n_lags=n_lags)
        X, Y, _ = prepare_training_data(df_features, target_columns=[
            "target_temperature_2m_max",
            "target_temperature_2m_min",
            "target_temperature_2m_mean",
            "target_will_rain",
            "target_windspeed_10m_max"
        ])

        while X.shape[1] % n_lags != 0:
            X = X[:, :-1]

        n_samples, n_total_features = X.shape
        n_features = n_total_features // n_lags
        X_seq = X.reshape(n_samples, n_lags, n_features)

        # Préparation DataLoader
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, Y_tensor)
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        model = WeatherLSTM(input_size=n_features)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, Y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, Y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"✅ Validation MAE: {val_loss:.2f}")
        safe_city = city.lower().replace(" ", "_")
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), f"model/{safe_city}_cnn.pt")

        # Mise à jour global best model
        best_info_path = "model/best_model_info.json"
        if os.path.exists(best_info_path):
            with open(best_info_path, "r") as f:
                best_info = json.load(f)
            current_best_score = best_info.get("score", float("inf"))
        else:
            current_best_score = float("inf")

        if val_loss < current_best_score:
            with open(best_info_path, "w") as f:
                json.dump({
                    "model_name": "cnn_lstm",
                    "model_path": f"model/{safe_city}_cnn.pt",
                    "score": val_loss
                }, f)
            print("🏆 Nouveau meilleur modèle global (LSTM)")
        else:
            joblib.dump({"model_path": f"model/{safe_city}_cnn.pt", "score": val_loss}, f"model/{safe_city}_cnn.pkl")
            print("📦 Modèle LSTM sauvegardé sans promotion globale")

    except Exception as e:
        print(f"Erreur pour {city} : {e}")