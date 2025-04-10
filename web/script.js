async function loadPrediction() {
    const cityInput = document.getElementById("city").value;
    const city = cityInput.toLowerCase()
      .normalize("NFD").replace(/[\u0300-\u036f]/g, "")
      .replace(" ", "_");
    const date = new Date().toISOString().split("T")[0]; // format YYYY-MM-DD
    const suffixes = ["j", "j1"];
    const output = document.getElementById("output");
  
    output.textContent = "Chargement...";
  
    try {
      const results = await Promise.all(suffixes.map(async (suffix) => {
        const file = `../predictions/${city}_${date}_${suffix}.json`;
        const response = await fetch(file);
        if (!response.ok) throw new Error(`Fichier introuvable : ${file}`);
        const data = await response.json();
        return { suffix, data };
      }));
  
      output.textContent = results.map(r => `📆 ${r.suffix.toUpperCase()} :\n` + JSON.stringify(r.data, null, 2)).join("\n\n");
    } catch (err) {
      output.textContent = "❌ Erreur : " + err.message;
    }
  }
  