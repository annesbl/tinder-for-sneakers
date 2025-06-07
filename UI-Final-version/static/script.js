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

    const bookmarked = JSON.parse(localStorage.getItem("bookmarkedShoes") || "[]");

    shuffled.forEach((shoe) => {
      const isBookmarked = bookmarked.includes(shoe.id);

      const slide = document.createElement("section");
      slide.className = "slide";

      slide.innerHTML = `
        <div class="bg-rectangle" style="background-color: ${shoe.light};"></div>
        <div class="bg-circle" style="background-color: ${shoe.strong};"></div>

        <img class="shoe-img" src="${shoe.image}" alt="${shoe.name}" onerror="this.src='placeholder.png'" />
        
        <div class="like-button Sohle">
          <img src="icons/heart-white.png" alt="like" class="heart-img" />
        </div>
        <div class="like-button Farbe">
          <img src="icons/heart-white.png" alt="like" class="heart-img" />
        </div>
        <div class="like-button Schnürsenkel">
          <img src="icons/heart-white.png" alt="like" class="heart-img" />
        </div>
        
        <div class="bookmark-button">
        <img src="icons/${isBookmarked ? "bookmark_filled" : "bookmark"}.png" 
             alt="bookmark" 
             class="bookmark-img" 
             data-id="${shoe.id}" />
        </div>

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


// ❤️ Like Button Farbe wechseln
document.addEventListener("click", function (e) {
  if (e.target.classList.contains("heart-img")) {
    const isLiked = e.target.classList.toggle("liked");
    e.target.src = isLiked ? "icons/heart-red.png" : "icons/heart-white.png";
  }
});


// 🔖 Bookmark-Button Farbe + speichern
document.addEventListener("click", function (e) {
  if (e.target.classList.contains("bookmark-img")) {
    const img = e.target;
    const shoeId = img.dataset.id;
    const isMarked = img.classList.toggle("marked");

    img.src = isMarked ? "icons/bookmark_filled.png" : "icons/bookmark.png";

    let bookmarks = JSON.parse(localStorage.getItem("bookmarkedShoes") || "[]");

    if (isMarked && !bookmarks.includes(shoeId)) {
      bookmarks.push(shoeId);
    } else if (!isMarked) {
      bookmarks = bookmarks.filter((id) => id !== shoeId);
    }

    localStorage.setItem("bookmarkedShoes", JSON.stringify(bookmarks));
  }
});


// 📖 Beschreibung ein-/ausblenden
document.addEventListener("click", function (e) {
  if (e.target.classList.contains("more-btn")) {
    const desc = e.target.closest(".product-line").nextElementSibling;
    const isOpen = desc.style.display === "block";
    desc.style.display = isOpen ? "none" : "block";
    e.target.textContent = isOpen ? "mehr" : "weniger";
  }
});



//kommunikation mit dem backend
async function sendFeedbackAndLoadShoe(vector) {
  try {
    const response = await fetch("http://localhost:8000/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(vector),
    });
    const data = await response.json();

    const recommendedId = data.recommendedId;
    if (!recommendedId) return;

    const response2 = await fetch("shoes.json");
    const shoesData = await response2.json();
    const newShoe = shoesData.find(shoe => shoe.id === recommendedId);

    if (newShoe) {
      showShoe(newShoe);
    } else {
      console.error("Empfohlene ID nicht gefunden:", recommendedId);
    }
  } catch (err) {
    console.error("Fehler bei /recommend:", err);
  }
}