# gridsearch_model.py (multi-ville avec RF, XGB, LGBM, CatBoost)

import os
import json
import joblib
import numpy as np
from fetch_data import fetch_weather
from preprocessing import clean_raw_data, create_features, prepare_training_data
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ----- CONFIGURATION -----
cities = ["Genève", "Lausanne", "Zurich", "Berne", "Bâle", "Neuchâtel", "Fribourg", "Lugano", "Lucerne", "Sion"]
past_days = 730
n_lags = 3
target_columns = [
    "target_temperature_2m_max",
    "target_temperature_2m_min",
    "target_temperature_2m_mean",
    "target_will_rain",
    "target_windspeed_10m_max",
]

# ----- HYPERPARAMÈTRES PAR MODÈLE -----
param_grids = {
    "random_forest": {
        "estimator__n_estimators": [100, 200],
        "estimator__max_depth": [10, None],
        "estimator__min_samples_split": [2, 5],
    },
    "xgboost": {
        "estimator__n_estimators": [100, 200],
        "estimator__max_depth": [3, 6],
        "estimator__learning_rate": [0.05, 0.1],
    },
    "lightgbm": {
        "estimator__n_estimators": [100, 200],
        "estimator__max_depth": [5, 10],
        "estimator__learning_rate": [0.05, 0.1],
    },
    "catboost": {
        "estimator__iterations": [100, 200],
        "estimator__depth": [4, 6],
        "estimator__learning_rate": [0.05, 0.1],
    }
}

# ----- MAE moyen -----
def multioutput_mae(y_true, y_pred):
    return np.mean([mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])

scorer = make_scorer(multioutput_mae, greater_is_better=False)

# ----- BEST GLOBAL -----
best_info_path = "model/best_model_info.json"
best_score = float("inf")
if os.path.exists(best_info_path):
    with open(best_info_path, "r") as f:
        best_info = json.load(f)
        best_score = best_info.get("score", float("inf"))

# ----- TRAINING LOOP -----
for city in cities:
    try:
        print(f"\n Entraînement pour {city}")
        df = fetch_weather(city, past_days=past_days)
        df_clean = clean_raw_data(df)
        df_features = create_features(df_clean, n_lags=n_lags)
        X, Y, scaler = prepare_training_data(df_features, target_columns=target_columns)

        safe_city = city.lower().replace(" ", "_")
        os.makedirs("model", exist_ok=True)

        best_model_city = None
        best_score_city = float("inf")
        best_name_city = None

        for name, grid in param_grids.items():
            print(f"🔍 GridSearch pour {name}...")

            if name == "random_forest":
                base_model = RandomForestRegressor(random_state=42)
            elif name == "xgboost":
                base_model = XGBRegressor(random_state=42, verbosity=0)
            elif name == "lightgbm":
                base_model = LGBMRegressor(random_state=42, verbose=-1)
            elif name == "catboost":
                base_model = CatBoostRegressor(verbose=0, random_state=42)
            else:
                continue

            multi_model = MultiOutputRegressor(base_model)
            grid_search = GridSearchCV(multi_model, grid, scoring=scorer, cv=3, n_jobs=-1)
            grid_search.fit(X, Y)

            score = -np.mean(cross_val_score(grid_search.best_estimator_, X, Y, cv=3, scoring=scorer, n_jobs=-1))
            print(f"{name} MAE: {score:.2f}")

            model_path = f"model/{safe_city}_{name}.pkl"
            joblib.dump(grid_search.best_estimator_, model_path)

            if score < best_score_city:
                best_model_city = grid_search.best_estimator_
                best_score_city = score
                best_name_city = name

        # Sauvegarder le scaler (commun)
        joblib.dump(scaler, f"model/{safe_city}_scaler.pkl")
        print(f"Meilleur modèle pour {city}: {best_name_city} ({best_score_city:.2f})")

        if best_score_city < best_score:
            with open(best_info_path, "w") as f:
                json.dump({
                    "model_name": best_name_city,
                    "model_path": f"model/{safe_city}_{best_name_city}.pkl",
                    "score": best_score_city
                }, f)
            best_score = best_score_city
            print("Nouveau meilleur modèle global !")

    except Exception as e:
        print(f"Erreur pour {city} : {e}")
