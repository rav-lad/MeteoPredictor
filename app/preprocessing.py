# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    df = df.dropna(axis=1, how="all")  # Supprimer colonnes 100% NA
    df = df.drop(columns=["city"], errors="ignore")

    imputer = KNNImputer(n_neighbors=5)
    df[df.columns.difference(["time"])] = imputer.fit_transform(df[df.columns.difference(["time"])])
    
    return df

def create_features(df: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
    df = df.copy()
    
    # --------- TARGETS décalées d'un jour (J) ---------
    df["target_temperature_2m_max"] = df["temperature_2m_max"].shift(-1)
    df["target_temperature_2m_min"] = df["temperature_2m_min"].shift(-1)
    df["target_temperature_2m_mean"] = ((df["temperature_2m_max"] + df["temperature_2m_min"]) / 2).shift(-1)
    df["target_will_rain"] = (df["precipitation_sum"].shift(-1) > 1.0).astype(int)
    df["target_windspeed_10m_max"] = df["windspeed_10m_max"].shift(-1)

    # --------- Feature engineering avec données J−1, J−2, etc. ---------
    df["temp_amplitude"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    df["rain_to_precip_ratio"] = df["rain_sum"] / (df["precipitation_sum"] + 1e-3)
    df["wind_diff"] = df["windgusts_10m_max"] - df["windspeed_10m_max"]
    df["sun_per_hour"] = df["sunshine_duration"] / 3600
    df["shortwave_per_hour"] = df["shortwave_radiation_sum"] / 24

    df["dayofyear"] = df["time"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df = df.drop(columns=["dayofyear"])

    # Lags (J−1, J−2, ...)
    columns_to_lag = [
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum",
        "showers_sum", "snowfall_sum", "windspeed_10m_max", "windgusts_10m_max",
        "sunshine_duration", "shortwave_radiation_sum", "et0_fao_evapotranspiration"
    ]
    for col in columns_to_lag:
        for lag in range(1, n_lags + 1):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    df["temp_max_rolling_mean_3"] = df["temperature_2m_max"].rolling(3).mean()
    df["rain_sum_rolling_std_3"] = df["rain_sum"].rolling(3).std()

    # Supprimer les colonnes brutes du jour J (sauf time)
    df = df.drop(columns=columns_to_lag + ["rain_sum", "temperature_2m_max", "temperature_2m_min"])

    # Supprimer les lignes avec NaN
    df = df.dropna()

    return df




def prepare_training_data(df: pd.DataFrame, target_columns):
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import KNNImputer

    features = df.drop(columns=["time"] + list(target_columns))
    targets = df[list(target_columns)]

    imputer = KNNImputer(n_neighbors=5)
    features_imputed = imputer.fit_transform(features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_imputed)

    return X_scaled, targets.values, scaler



