async function loadShoes() {
  try {
    const response = await fetch("shoes.json");
    if (!response.ok) throw new Error("shoes.json konnte nicht geladen werden");
    const data = await response.json();
    if (!Array.isArray(data)) throw new Error("JSON-Format ungültig");

    const shuffled = data.sort(() => Math.random() - 0.5);
    const container = document.getElementById("scroll-container");

    if (shuffled.length === 0) {
      container.innerHTML = `<div style="padding: 20px;">Keine Schuhe gefunden.</div>`;
      return;
    }

    shuffled.forEach((shoe) => {
      const slide = document.createElement("section");
      slide.className = "slide";
      slide.innerHTML = `
        <div class="bg-rectangle" style="background-color: ${shoe.light};"></div>
        <div class="bg-circle" style="background-color: ${shoe.strong};"></div>
        <img class="shoe-img" src="${shoe.image}" alt="${shoe.name}" onerror="this.src='placeholder.png'" />
        <div class="text-box">
          <div class="shoe-name-text">${shoe.name}</div>
          <div class="product-line">
            <span class="product-label">Produktinformationen</span>
            <button class="more-btn">mehr</button>
          </div>
          <div class="description-text">${shoe.description}</div>
        </div>`;
      container.appendChild(slide);
    });
  } catch (err) {
    const container = document.getElementById("scroll-container");
    container.innerHTML = `<div style="padding: 20px; color: red;">Fehler: ${err.message}</div>`;
    console.error("Fehler beim Laden der Daten:", err);
  }
}

loadShoes();
