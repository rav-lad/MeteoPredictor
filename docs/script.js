// script.js

const translations = {
  fr: {
    title: "Prédictions météo",
    chooseCity: "Choisissez une ville :",
    seePredictions: "Voir les prédictions",
    error: "Erreur : Fichier introuvable :",
    loading: "Chargement..."
  },
  en: {
    title: "Weather Predictions",
    chooseCity: "Choose a city:",
    seePredictions: "See predictions",
    error: "Error: File not found:",
    loading: "Loading..."
  },
  de: {
    title: "Wettervorhersage",
    chooseCity: "Wählen Sie eine Stadt:",
    seePredictions: "Vorhersagen anzeigen",
    error: "Fehler: Datei nicht gefunden:",
    loading: "Laden..."
  },
  it: {
    title: "Previsioni meteo",
    chooseCity: "Scegli una città:",
    seePredictions: "Vedi le previsioni",
    error: "Errore: File non trovato:",
    loading: "Caricamento..."
  }
};

let currentLang = "fr";

function setLanguage(lang) {
  currentLang = lang;
  document.getElementById("title").textContent = translations[lang].title;
  document.querySelector("label[for='city']").textContent = translations[lang].chooseCity;
  document.getElementById("loadButton").textContent = translations[lang].seePredictions;
}

async function loadPrediction() {
  const cityInput = document.getElementById("city").value;
  const city = cityInput.toLowerCase()
    .normalize("NFD").replace(/[̀-ͯ]/g, "")
    .replace(" ", "_");
  const date = new Date().toISOString().split("T")[0];
  const suffixes = ["j", "j1"];
  const output = document.getElementById("output");

  output.textContent = translations[currentLang].loading;

  try {
    const results = await Promise.all(suffixes.map(async (suffix) => {
      const file = `docs/predictions/${city}_${date}_${suffix}.json`;
      const response = await fetch(file);
      if (!response.ok) throw new Error(`${translations[currentLang].error} ${file}`);
      const data = await response.json();
      return { suffix, data };
    }));

    output.textContent = results.map(r => ` ${r.suffix.toUpperCase()} :\n` + JSON.stringify(r.data, null, 2)).join("\n\n");
  } catch (err) {
    output.textContent = `${translations[currentLang].error} ${err.message}`;
  }
}