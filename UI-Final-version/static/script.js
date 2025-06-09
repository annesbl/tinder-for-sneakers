let shoeStack = []; // enthält immer max. 2 Schuhe
let vector = [0, 0, 0, 0]; // [sohle, farbe, schnuersenkel, mehr]
const container = document.getElementById("scroll-container");
let lastScrollIndex = 0;
let isProcessingFeedback = false; // Flag für Race Condition Prevention

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
  console.log("🔄 Starting renderShoes...");
  container.innerHTML = "";
  const bookmarked = JSON.parse(localStorage.getItem("bookmarkedShoes") || "[]");
  
  shoeStack.forEach((shoe, index) => {
    const isBookmarked = bookmarked.includes(String(shoe.id));
    const slide = document.createElement("section");
    slide.className = "slide";
    slide.innerHTML = `
      <div class="bg-rectangle" style="background-color: ${shoe.light};"></div>
      <div class="bg-circle" style="background-color: ${shoe.strong};"></div>
      <img class="shoe-img" src="/static/${shoe.image}" alt="${shoe.name}" />
      <div class="like-button Sohle" data-debug="sohle-${shoe.id}">
        <img src="/static/icons/heart-white.png" alt="like" class="heart-img" />
      </div>
      <div class="like-button Farbe" data-debug="farbe-${shoe.id}">
        <img src="/static/icons/heart-white.png" alt="like" class="heart-img" />
      </div>
      <div class="like-button Schnürsenkel" data-debug="schnuersenkel-${shoe.id}">
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
          <button class="more-btn" data-debug="mehr-${shoe.id}">mehr</button>
        </div>
        <div class="description-text" style="display:none;">${shoe.description}</div>
      </div>
    `;
    console.log(`✅ Rendered shoe ${index}:`, shoe.name, `(ID: ${shoe.id})`);
    container.appendChild(slide);
  });
  
  // Teste ob Buttons da sind
  setTimeout(() => {
    const buttons = container.querySelectorAll('.like-button, .more-btn, .bookmark-img');
    console.log(`🔍 Found ${buttons.length} clickable elements after render`);
    buttons.forEach((btn, i) => {
      console.log(`  Button ${i}:`, btn.className, btn.dataset.debug || 'no-debug');
    });
  }, 100);
}

// Debounce-Funktion für bessere Performance
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Throttle-Funktion als Alternative
function throttle(func, limit) {
  let inThrottle;
  return function() {
    const args = arguments;
    const context = this;
    if (!inThrottle) {
      func.apply(context, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  }
}

// VERBESSERTE SCROLL-LOGIK mit Race Condition Prevention
const handleScroll = throttle(async function() {
  // Verhindere mehrfache gleichzeitige Ausführung
  if (isProcessingFeedback) {
    console.log("Feedback wird bereits verarbeitet, überspringe...");
    return;
  }

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

  // Nur ausführen wenn wirklich zum zweiten Schuh gescrollt wurde
  if (currentIndex === 1 && lastScrollIndex !== 1) {
    lastScrollIndex = 1;
    isProcessingFeedback = true; // Flag setzen
    
    console.log("Scrolled to second shoe, processing feedback...");
    
    // Aktuellen Vector kopieren BEVOR er zurückgesetzt wird
    const currentVector = [...vector];
    const currentShoeId = shoeStack[0].id;
    
    const feedback = {
      sohle: currentVector[0] || 0,
      farbe: currentVector[1] || 0,
      schnuersenkel: currentVector[2] || 0,
      mehr: currentVector[3] || 0,
      id: currentShoeId
    };
    
    console.log("Sending feedback:", feedback);

    try {
      const newShoe = await sendFeedbackAndLoadShoe(feedback);
      if (!newShoe || newShoe.error) {
        console.warn("Kein neuer Schuh erhalten:", newShoe);
        isProcessingFeedback = false;
        return;
      }

      // Stack aktualisieren
      shoeStack.shift();
      shoeStack.push(newShoe);
      
      // Neu rendern
      renderShoes();

      // Scroll-Position zurücksetzen
      setTimeout(() => {
        container.scrollTop = 0;
        lastScrollIndex = 0;
        console.log("Scroll position reset");
      }, 100); // Kurze Verzögerung für bessere UX
      
    } catch (error) {
      console.error("Fehler beim Verarbeiten des Feedbacks:", error);
    } finally {
      isProcessingFeedback = false; // Flag zurücksetzen
    }
  }
}, 150); // Throttle auf 150ms

// Event Listener mit verbesserter Scroll-Logik
container.addEventListener('scroll', handleScroll);

// Feedback senden und neuen Schuh laden
async function sendFeedbackAndLoadShoe(feedback) {
  try {
    const response = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(feedback),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    const recommendedId = data.recommendedId;
    
    // Vector erst NACH erfolgreichem Senden zurücksetzen
    vector = [0, 0, 0, 0];
    console.log("Vector reset after successful feedback");
    
    if (!recommendedId) {
      console.warn("Keine recommendedId erhalten");
      return null;
    }
    
    const response2 = await fetch(`/shoe/${recommendedId}`);
    if (!response2.ok) {
      throw new Error(`HTTP error! status: ${response2.status}`);
    }
    
    const newShoe = await response2.json();
    console.log("New shoe loaded:", newShoe.name);
    return newShoe;
    
  } catch (error) {
    console.error("Fehler beim Senden des Feedbacks:", error);
    return null;
  }
}

// ZENTRALER EVENT HANDLER mit verbessertem Debugging
document.addEventListener("click", function (e) {
  console.log("🖱️ Click detected on:", e.target.tagName, e.target.className, e.target.dataset.debug || 'no-debug');
  
  // Like-Buttons - Herz-Grafik ändern
  if (e.target.classList.contains("heart-img")) {
    console.log("❤️ Heart button clicked");
    const isLiked = e.target.classList.toggle("liked");
    e.target.src = isLiked ? "/static/icons/heart-red.png" : "/static/icons/heart-white.png";
    return;
  }

  // Sohle Button - Vector togglen
  if (e.target.closest(".like-button.Sohle")) {
    console.log("👟 Sohle button clicked");
    vector[0] = vector[0] === 1 ? 0 : 1;
    console.log("Sohle vector:", vector[0], "Complete vector:", vector);
    return;
  }

  // Farbe Button - Vector togglen
  if (e.target.closest(".like-button.Farbe")) {
    console.log("🎨 Farbe button clicked");
    vector[1] = vector[1] === 1 ? 0 : 1;
    console.log("Farbe vector:", vector[1], "Complete vector:", vector);
    return;
  }

  // Schnürsenkel Button - Vector togglen
  if (e.target.closest(".like-button.Schnürsenkel")) {
    console.log("🔗 Schnürsenkel button clicked");
    vector[2] = vector[2] === 1 ? 0 : 1;
    console.log("Schnürsenkel vector:", vector[2], "Complete vector:", vector);
    return;
  }

  // Bookmark-Button
  if (e.target.classList.contains("bookmark-img")) {
    console.log("🔖 Bookmark button clicked");
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
    return;
  }

  // Beschreibung ein-/ausblenden
  if (e.target.classList.contains("more-btn")) {
    console.log("📖 More button clicked");
    const desc = e.target.closest(".product-line").nextElementSibling;
    const isOpen = desc.style.display === "block";
    desc.style.display = isOpen ? "none" : "block";
    e.target.textContent = isOpen ? "mehr" : "weniger";
    
    // Vector für "mehr" togglen
    vector[3] = vector[3] === 1 ? 0 : 1;
    console.log("Mehr vector:", vector[3], "Complete vector:", vector);
    return;
  }
  
  // Falls kein bekannter Button geklickt wurde
  console.log("🤷 Unknown element clicked");
});

window.onload = initShoes;