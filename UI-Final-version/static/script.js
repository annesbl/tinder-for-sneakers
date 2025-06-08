let shoeStack = []; // enthält immer max. 2 Schuhe
let vector = [0, 0, 0, 0]; // [sohle, farbe, schnuersenkel, mehr]
const container = document.getElementById("scroll-container");
let lastScrollIndex = 0;

// Initial zwei Schuhe laden
async function initShoes() {
  const response = await fetch("/shoes");
  const allShoes = await response.json();
  // random die Schuhe
  const shuffled = allShoes.sort(() => Math.random() - 0.5);
  shoeStack = shuffled.slice(0, 2);
  console.log("Initial shoes loaded:", shoeStack);
  // Merke dir die restlichen Schuhe für später
  window.remainingShoes = shuffled.slice(2);
  renderShoes();
}

// Renderfunktion für die zwei aktuellen Schuhe
function renderShoes() {
  container.innerHTML = "";
  const bookmarked = JSON.parse(localStorage.getItem("bookmarkedShoes") || "[]");
  shoeStack.forEach(shoe => {
    const isBookmarked = bookmarked.includes(String(shoe.id));
    const slide = document.createElement("section");
    slide.className = "slide";
    slide.innerHTML = `
      <div class="bg-rectangle" style="background-color: ${shoe.light};"></div>
      <div class="bg-circle" style="background-color: ${shoe.strong};"></div>
      <img class="shoe-img" src="/static/${shoe.image}" alt="${shoe.name}" />
      <div class="like-button Sohle">
        <img src="/static/icons/heart-white.png" alt="like" class="heart-img" />
      </div>
      <div class="like-button Farbe">
        <img src="/static/icons/heart-white.png" alt="like" class="heart-img" />
      </div>
      <div class="like-button Schnürsenkel">
        <img src="/static/icons/heart-white.png" alt="like" class="heart-img" />
      </div>
      <div class="bookmark-button">
        <img src="/static/icons/${isBookmarked ? "bookmark_filled" : "bookmark"}.png" 
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
        <div class="description-text" style="display:none;">${shoe.description}</div>
      </div>
    `;
    console.log("Rendered shoe:", shoe.name);
    container.appendChild(slide);
  });
}

// SCROLL-LOGIK: Nur wenn zweiter Schuh sichtbar, Feedback senden und neuen Schuh nachladen
container.addEventListener('scroll', async function () {
  const slides = Array.from(container.getElementsByClassName('slide'));
  let currentIndex = 0;
  let minDiff = Infinity;

  slides.forEach((slide, idx) => {
    const rect = slide.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    const diff = Math.abs(rect.top - containerRect.top);
    if (diff < minDiff) {
      minDiff = diff;
      currentIndex = idx;
    }
  });

  if (currentIndex === 1 && lastScrollIndex !== 1) {
    lastScrollIndex = 1;
    const feedback = {
      sohle: vector[0] || 0,
      farbe: vector[1] || 0,
      schnuersenkel: vector[2] || 0,
      mehr: vector[3] || 0,
      id: shoeStack[0].id
    };
    console.log("Sending feedback:", feedback);

    const newShoe = await sendFeedbackAndLoadShoe(feedback);
    if (!newShoe || newShoe.error) {
      console.warn("Kein neuer Schuh, überspringe.");
      return;
    }

    shoeStack.shift();
    shoeStack.push(newShoe);
    renderShoes();

    setTimeout(() => {
      container.scrollTop = 0;
      lastScrollIndex = 0;
    }, 50);
  }
});


// Feedback senden und neuen Schuh laden
async function sendFeedbackAndLoadShoe(feedback) {
  // Wenn noch Schuhe in remainingShoes sind, nimm einen davon, sonst vom Backend
  // Fallback: vom Backend einen neuen empfehlen lassen
  const response = await fetch("/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(feedback),
  });
  const data = await response.json();
  const recommendedId = data.recommendedId;
  
  // Vector zurücksetzen nach dem Senden
  vector = [0, 0, 0, 0];
  console.log("Vector reset after feedback");
  
  if (!recommendedId) return null;
  const response2 = await fetch(`/shoe/${recommendedId}`);
  const newShoe = await response2.json();
  return newShoe;
}

// Like-Buttons - Herz-Grafik ändern
document.addEventListener("click", function (e) {
  if (e.target.classList.contains("heart-img")) {
    const isLiked = e.target.classList.toggle("liked");
    e.target.src = isLiked ? "/static/icons/heart-red.png" : "/static/icons/heart-white.png";
  }
});

// Sohle Button - Vector togglen
document.addEventListener("click", function (e) {
  if (e.target.closest(".like-button.Sohle")) {
    vector[0] = vector[0] === 1 ? 0 : 1;
    console.log("Sohle vector:", vector[0], "Complete vector:", vector);
  }
});

// Farbe Button - Vector togglen
document.addEventListener("click", function (e) {
  if (e.target.closest(".like-button.Farbe")) {
    vector[1] = vector[1] === 1 ? 0 : 1;
    console.log("Farbe vector:", vector[1], "Complete vector:", vector);
  }
});

// Schnürsenkel Button - Vector togglen
document.addEventListener("click", function (e) {
  if (e.target.closest(".like-button.Schnürsenkel")) {
    vector[2] = vector[2] === 1 ? 0 : 1;
    console.log("Schnürsenkel vector:", vector[2], "Complete vector:", vector);
  }
});

// Bookmark-Button
document.addEventListener("click", function (e) {
  if (e.target.classList.contains("bookmark-img")) {
    const img = e.target;
    const shoeId = img.dataset.id;
    const isMarked = img.classList.toggle("marked");

    img.src = isMarked ? "/static/icons/bookmark_filled.png" : "/static/icons/bookmark.png";

    let bookmarks = JSON.parse(localStorage.getItem("bookmarkedShoes") || "[]");

    if (isMarked && !bookmarks.includes(shoeId)) {
      bookmarks.push(shoeId);
    } else if (!isMarked) {
      bookmarks = bookmarks.filter((id) => id !== shoeId);
    }

    localStorage.setItem("bookmarkedShoes", JSON.stringify(bookmarks));
  }
});

// Beschreibung ein-/ausblenden
document.addEventListener("click", function (e) {
  if (e.target.classList.contains("more-btn")) {
    const desc = e.target.closest(".product-line").nextElementSibling;
    const isOpen = desc.style.display === "block";
    desc.style.display = isOpen ? "none" : "block";
    e.target.textContent = isOpen ? "mehr" : "weniger";
    
    // Vector für "mehr" togglen
    vector[3] = vector[3] === 1 ? 0 : 1;
    console.log("Mehr vector:", vector[3], "Complete vector:", vector);
  }
});

window.onload = initShoes;