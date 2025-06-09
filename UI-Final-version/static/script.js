document.addEventListener("DOMContentLoaded", () => {
  let shoeStack = [];
  let vector = [0, 0, 0, 0];
  let lastScrollIndex = 0;
  let isProcessingFeedback = false;

  function getContainer() {
    return document.getElementById("scroll-container");
  }

  async function initShoes() {
    const response = await fetch("/shoes");
    const allShoes = await response.json();
    const shuffled = allShoes.sort(() => Math.random() - 0.5);
    shoeStack = shuffled.slice(0, 2);
    window.remainingShoes = shuffled.slice(2);
    renderShoes();
  }

  function renderShoes() {
    const container = getContainer();
    container.innerHTML = "";
    const bookmarked = JSON.parse(localStorage.getItem("bookmarkedShoes") || "[]");

    shoeStack.forEach((shoe) => {
      const isBookmarked = bookmarked.includes(String(shoe.id));
      const slide = document.createElement("section");
      slide.className = "slide";
      slide.innerHTML = `
        <div class="bg-rectangle" style="background-color: ${shoe.light};"></div>
        <div class="bg-circle" style="background-color: ${shoe.strong};"></div>
        <img class="shoe-img" src="/static/${shoe.image}" alt="${shoe.name}" />
        <div class="like-button Sohle" data-debug="sohle-${shoe.id}" data-type="sohle">
          <img src="/static/icons/heart-white.png" alt="like" class="heart-img" />
        </div>
        <div class="like-button Farbe" data-debug="farbe-${shoe.id}" data-type="farbe">
          <img src="/static/icons/heart-white.png" alt="like" class="heart-img" />
        </div>
        <div class="like-button Schnürsenkel" data-debug="schnuersenkel-${shoe.id}" data-type="schnuersenkel">
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
            <button class="more-btn" data-debug="mehr-${shoe.id}" data-type="mehr">mehr</button>
          </div>
          <div class="description-text" style="display:none;">${shoe.description}</div>
        </div>
      `;
      container.appendChild(slide);
    });

    // Kleine Pause, damit Layout & Styles greifen
    setTimeout(() => {
      attachEventListeners();
    }, 100);
  }

  function attachEventListeners() {
    const container = getContainer();
    container.removeEventListener("click", handleContainerClick);
    container.removeEventListener("scroll", handleScroll);
    container.addEventListener("click", handleContainerClick);
    container.addEventListener("scroll", handleScroll);
  }

  function handleContainerClick(e) {
    if (e.target.classList.contains("heart-img")) {
      const isLiked = e.target.classList.toggle("liked");
      e.target.src = isLiked ? "/static/icons/heart-red.png" : "/static/icons/heart-white.png";
      const parentButton = e.target.closest('.like-button');
      if (parentButton) updateVector(parentButton.dataset.type, isLiked ? 1 : 0);
      return;
    }

    if (e.target.classList.contains("like-button") || e.target.closest(".like-button")) {
      const button = e.target.closest(".like-button");
      const heartImg = button.querySelector('.heart-img');
      const isLiked = heartImg.classList.toggle("liked");
      heartImg.src = isLiked ? "/static/icons/heart-red.png" : "/static/icons/heart-white.png";
      updateVector(button.dataset.type, isLiked ? 1 : 0);
      return;
    }

    if (e.target.classList.contains("bookmark-img")) {
      const img = e.target;
      const shoeId = img.dataset.id;
      const isMarked = img.classList.toggle("marked");
      img.src = isMarked ? "/static/icons/bookmark_filled.png" : "/static/icons/bookmark.png";
      let bookmarks = JSON.parse(localStorage.getItem("bookmarkedShoes") || "[]");
      if (isMarked && !bookmarks.includes(shoeId)) bookmarks.push(shoeId);
      else if (!isMarked) bookmarks = bookmarks.filter((id) => id !== shoeId);
      localStorage.setItem("bookmarkedShoes", JSON.stringify(bookmarks));
      return;
    }

    if (e.target.classList.contains("more-btn")) {
      const desc = e.target.closest(".product-line").nextElementSibling;
      const isOpen = desc.style.display === "block";
      desc.style.display = isOpen ? "none" : "block";
      e.target.textContent = isOpen ? "mehr" : "weniger";
      if (!isOpen && vector[3] === 0) {
        updateVector("mehr", 1);
      }
      return;
    }
  }

  function updateVector(type, value) {
    const indexMap = { sohle: 0, farbe: 1, schnuersenkel: 2, mehr: 3 };
    if (indexMap.hasOwnProperty(type)) vector[indexMap[type]] = value;
  }

  function throttle(func, limit) {
    let inThrottle;
    return function () {
      const args = arguments;
      const context = this;
      if (!inThrottle) {
        func.apply(context, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  }

  const handleScroll = throttle(async function () {
    const container = getContainer();
    const slides = Array.from(container.getElementsByClassName('slide'));
    let currentIndex = 0;
    let minDiff = Infinity;

    slides.forEach((slide, idx) => {
      const diff = Math.abs(slide.getBoundingClientRect().top - container.getBoundingClientRect().top);
      if (diff < minDiff) {
        minDiff = diff;
        currentIndex = idx;
      }
    });

    if (currentIndex === 1 && !isProcessingFeedback) {
      isProcessingFeedback = true;

      const feedback = {
        sohle: vector[0] || 0,
        farbe: vector[1] || 0,
        schnuersenkel: vector[2] || 0,
        mehr: vector[3] || 0,
        id: shoeStack[0].id
      };

      try {
        const newShoe = await sendFeedbackAndLoadShoe(feedback);
        if (!newShoe || newShoe.error) {
          isProcessingFeedback = false;
          return;
        }

        shoeStack.shift();
        shoeStack.push(newShoe);
        renderShoes();

        // Scroll sanft zurücksetzen und danach Index resetten
        requestAnimationFrame(() => {
          const container = getContainer();
          container.scrollTo({ top: 0, behavior: "smooth" });

          setTimeout(() => {
            lastScrollIndex = 0;
            isProcessingFeedback = false;
          }, 400);
        });
      } catch (error) {
        console.error("Fehler beim Scroll-Feedback:", error);
        isProcessingFeedback = false;
      }
    }
  }, 150);

  async function sendFeedbackAndLoadShoe(feedback) {
    try {
      const response = await fetch("/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(feedback),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      vector = [0, 0, 0, 0];

      if (!data.recommendedId) return null;
      const shoeRes = await fetch(`/shoe/${data.recommendedId}`);
      if (!shoeRes.ok) throw new Error(`HTTP error! status: ${shoeRes.status}`);
      return await shoeRes.json();
    } catch (err) {
      console.error("Fehler beim Laden des neuen Schuhs:", err);
      return null;
    }
  }

  initShoes();
});
