
# ☁️ MeteoPredictor

**MeteoPredictor** is a weather forecasting tool powered by **machine learning**, designed to predict key weather variables such as temperature, humidity, and precipitation. This project serves as a foundational block for larger systems aiming to anticipate energy consumption based on weather patterns.

---

##  Project Goals

- Deliver **accurate and robust weather forecasts** using historical data.
- Act as a core module for energy consumption analysis based on weather.
- Experiment with advanced techniques for **time series forecasting**.
- Web App Preview:
   Built with simple HTML/CSS/JavaScript and hosted via GitHub Pages
   Displays forecast results for major Swiss cities
   Forecasts are updated daily for: Today (J) and Tomorrow (J+1)
---

##  Models & Methodology

-  **Main model**: [CatBoost](https://catboost.ai/),[XGBoost],[LigthGBM], [CNN] – highly effective for tabular and time series data.
-  **Feature Engineering**: includes normalization, time-based encoding, and relevant variable selection.
-  **Evaluation Metrics**: RMSE, MAE, R².

---

##  Project Structure

```bash
MeteoPredictor/
├── app/                 # Core application logic
├── data/                # Raw and preprocessed datasets
├── docs/                # Project documentation
├── model/               # Trained model files (checkpoints)
├── notebooks/           # Exploratory notebooks & prototypes
├── predictions/         # Forecasting outputs
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

##  Getting Started

###  Prerequisites

- Python 3.7 or higher
- (Recommended) Use a virtual environment

###  Installation

```bash
git clone https://github.com/rav-lad/MeteoPredictor.git
cd MeteoPredictor
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```



##  Datasets

The project uses historical weather data including features like:
- Temperature (min, max, apparent)
- Precipitation, snow, and rain levels
- Wind speed and direction
- Solar radiation and evapotranspiration

>  This project uses data from **[Open-Meteo](https://open-meteo.com/)** – a free and open API providing historical and forecast weather data.


---

##  Results & Evaluation

Include example predictions, model performance metrics (e.g., RMSE, R²), and a visual comparison between predicted and actual values.

---

##  Contributing

Contributions, issues, and feature requests are welcome!

Feel free to fork the repo and submit a pull request.

---

##  Author

Created by **rav-lad** – check out more at [https://github.com/rav-lad](https://github.com/rav-lad)

```
