const config = {
  cities: {
    fr: ["Genève", "Lausanne", "Zurich", "Berne", "Bâle", "Neuchâtel", "Fribourg", "Lugano", "Lucerne", "Sion"],
    en: ["Geneva", "Lausanne", "Zurich", "Bern", "Basel", "Neuchatel", "Fribourg", "Lugano", "Lucerne", "Sion"],
    de: ["Genf", "Lausanne", "Zürich", "Bern", "Basel", "Neuenburg", "Freiburg", "Lugano", "Luzern", "Sitten"],
    it: ["Ginevra", "Losanna", "Zurigo", "Berna", "Basilea", "Neuchâtel", "Friburgo", "Lugano", "Lucerna", "Sion"]
  },
  translations: {
    fr: {
      title: "Prédictions météo",
      chooseCity: "Choisissez une ville :",
      seePredictions: "Voir les prédictions",
      error: "Erreur : Fichier introuvable :",
      loading: "Chargement...",
      welcome: "Bienvenue sur le service de prédictions météo",
      instructions: "Sélectionnez une ville et cliquez sur \"Voir les prédictions\"",
      today: "Aujourd'hui",
      tomorrow: "Demain",
      maxTemp: "Max :",
      minTemp: "Min :",
      avgTemp: "Moy :",
      wind: "Vent :",
      rainProb: "Pluie :"
    },
    en: {
      title: "Weather Forecast",
      chooseCity: "Choose a city:",
      seePredictions: "See predictions",
      error: "Error: File not found:",
      loading: "Loading...",
      welcome: "Welcome to the weather forecast service",
      instructions: "Select a city and click \"See predictions\"",
      today: "Today",
      tomorrow: "Tomorrow",
      maxTemp: "Max:",
      minTemp: "Min:",
      avgTemp: "Avg:",
      wind: "Wind:",
      rainProb: "Rain:"
    },
    de: {
      title: "Wettervorhersage",
      chooseCity: "Wählen Sie eine Stadt:",
      seePredictions: "Vorhersagen anzeigen",
      error: "Fehler: Datei nicht gefunden:",
      loading: "Laden...",
      welcome: "Willkommen beim Wettervorhersage-Service",
      instructions: "Wählen Sie eine Stadt und klicken Sie auf \"Vorhersagen anzeigen\"",
      today: "Heute",
      tomorrow: "Morgen",
      maxTemp: "Max:",
      minTemp: "Min:",
      avgTemp: "Durchschnitt:",
      wind: "Wind:",
      rainProb: "Regen:"
    },
    it: {
      title: "Previsioni meteo",
      chooseCity: "Scegli una città:",
      seePredictions: "Vedi previsioni",
      error: "Errore: File non trovato:",
      loading: "Caricamento...",
      welcome: "Benvenuti nel servizio di previsioni meteo",
      instructions: "Seleziona una città e clicca \"Vedi previsioni\"",
      today: "Oggi",
      tomorrow: "Domani",
      maxTemp: "Max:",
      minTemp: "Min:",
      avgTemp: "Med:",
      wind: "Vento:",
      rainProb: "Pioggia:"
    }
  }
};

let currentLang = "fr";

function init() {
  setLanguage(currentLang);
  setupEventListeners();
}

function setLanguage(lang) {
  currentLang = lang;
  const t = config.translations[lang];

  document.getElementById('title').textContent = `🌤️ ${t.title}`;
  document.getElementById('city-label').textContent = t.chooseCity;
  document.getElementById('loadButton').textContent = t.seePredictions;
  const welcomeMessage = document.querySelector('.welcome-message');
  if (welcomeMessage) {
    welcomeMessage.querySelector('p:first-child').textContent = t.welcome;
    welcomeMessage.querySelector('p:last-child').textContent = t.instructions;
  }
  document.getElementById('lang-select').value = lang;

  updateCityDropdown();
}

function updateCityDropdown() {
  const citySelect = document.getElementById('city');
  citySelect.innerHTML = '';

  config.cities[currentLang].forEach((city, index) => {
    const option = document.createElement('option');
    option.value = config.cities.fr[index]; // Valeur en français pour les fichiers (clé)
    option.textContent = city;
    citySelect.appendChild(option);
  });
}

function setupEventListeners() {
  document.getElementById('lang-select').addEventListener('change', (e) => {
    setLanguage(e.target.value);
  });

  document.getElementById('loadButton').addEventListener('click', loadPrediction);
}

async function loadPrediction() {
  const cityInput = document.getElementById('city').value;
  const output = document.getElementById('output');

  if (!cityInput) {
    output.innerHTML = `<div class="error">${config.translations[currentLang].chooseCity}</div>`;
    return;
  }

  output.innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      <p>${config.translations[currentLang].loading}</p>
    </div>
  `;

  try {
    const dateToday = new Date().toISOString().split('T')[0];
    const dateTomorrow = new Date(Date.now() + (24 * 60 * 60 * 1000)).toISOString().split('T')[0];

    const predictionToday = await fetchPrediction(cityInput, dateToday, 'j');
    const predictionTomorrow = await fetchPrediction(cityInput, dateTomorrow, 'j1');

    displayPredictions({ today: predictionToday, tomorrow: predictionTomorrow });

  } catch (error) {
    output.innerHTML = `
      <div class="error">
        ${config.translations[currentLang].error} ${error.message}
      </div>
    `;
  }
}

function sanitizeCity(city) {
  return city
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z]/g, '');
}

async function fetchPrediction(city, date, suffix) {
  const cityKey = sanitizeCity(city); // Ex: "Genève" → "geneve"
  const fileName = `${cityKey}_${date}_${suffix}.json`; // Ex: geneve_2025-04-10_j.json
  const filePath = `predictions/${fileName}`; // Ex: predictions/geneve_2025-04-10_j.json

  try {
    const response = await fetch(filePath);
    if (!response.ok) {
      throw new Error(`${filePath}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(error.message);
  }
}

function displayPredictions(data) {
  const output = document.getElementById('output');
  output.innerHTML = '';
  const t = config.translations[currentLang];

  const todayData = data.today;
  const tomorrowData = data.tomorrow;

  const todayDiv = document.createElement('div');
  todayDiv.classList.add('forecast-day');
  todayDiv.innerHTML = `
    <h3>${t.today}</h3>
    <p>${t.maxTemp} ${todayData.maxTemp}°C</p>
    <p>${t.minTemp} ${todayData.minTemp}°C</p>
    <p>${t.avgTemp} ${todayData.avgTemp}°C</p>
    <p>${t.wind} ${todayData.windSpeed} km/h</p>
    <p>${t.rainProb} ${todayData.rainProbability}%</p>
  `;

  const tomorrowDiv = document.createElement('div');
  tomorrowDiv.classList.add('forecast-day');
  tomorrowDiv.innerHTML = `
    <h3>${t.tomorrow}</h3>
    <p>${t.maxTemp} ${tomorrowData.maxTemp}°C</p>
    <p>${t.minTemp} ${tomorrowData.minTemp}°C</p>
    <p>${t.avgTemp} ${tomorrowData.avgTemp}°C</p>
    <p>${t.wind} ${tomorrowData.windSpeed} km/h</p>
    <p>${t.rainProb} ${tomorrowData.rainProbability}%</p>
  `;

  output.appendChild(todayDiv);
  output.appendChild(tomorrowDiv);
}

// Lancer le script à la fin du chargement de la page
document.addEventListener('DOMContentLoaded', init);
