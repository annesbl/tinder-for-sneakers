document.addEventListener("DOMContentLoaded", () => {
  let shoeStack = [];
  let vector = [0, 0, 0, 0];  // User interaction feedback vector
  let isProcessingFeedback = false;

  function getContainer() {
    return document.getElementById("scroll-container");
  }

  async function initShoes() {
    // Fetch initial shoes and shuffle them
    const response = await fetch("/shoes");
    const allShoes = await response.json();
    const shuffled = allShoes.sort(() => Math.random() - 0.5);
    shoeStack = shuffled.slice(0, 2);
    renderShoes();
  }

  function renderShoes() {
    // Render current shoes in the container
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
        <img class="shoe-img" src="/static/Shoes/${shoe.image}" alt="${shoe.name}" />
        <div class="like-button Sohle" data-type="sohle">
          <img src="/static/icons/heart-white.png" alt="like" class="heart-img" />
        </div>
        <div class="like-button Farbe" data-type="farbe">
          <img src="/static/icons/heart-white.png" alt="like" class="heart-img" />
        </div>
        <div class="like-button Schnürsenkel" data-type="schnuersenkel">
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
            <button class="more-btn" data-type="mehr">mehr</button>
          </div>
          <div class="description-text" style="display:none;">${shoe.description}</div>
        </div>
      `;
      container.appendChild(slide);
    });

    setTimeout(attachEventListeners, 100);
  }

  function attachEventListeners() {
    // Attach click and scroll event listeners
    const container = getContainer();
    container.removeEventListener("click", handleContainerClick);
    container.removeEventListener("scroll", handleScroll);
    container.addEventListener("click", handleContainerClick);
    container.addEventListener("scroll", handleScroll);
  }

  function handleContainerClick(e) {
    // Handle click events for like, bookmark, and more buttons
    if (e.target.classList.contains("heart-img")) {
      const heart = e.target;
      const isLiked = heart.classList.toggle("liked");
      heart.src = isLiked ? "/static/icons/heart-red.png" : "/static/icons/heart-white.png";
      const type = heart.closest(".like-button").dataset.type;
      updateVector(type, isLiked ? 1 : 0);
    }

    if (e.target.classList.contains("bookmark-img")) {
      toggleBookmark(e.target);
    }

    if (e.target.classList.contains("more-btn")) {
      toggleDescription(e.target);
    }
  }

  function toggleBookmark(img) {
    // Toggle bookmark icon and update local storage
    const shoeId = img.dataset.id;
    const isMarked = img.classList.toggle("marked");
    img.src = isMarked ? "/static/icons/bookmark_filled.png" : "/static/icons/bookmark.png";
    let bookmarks = JSON.parse(localStorage.getItem("bookmarkedShoes") || "[]");
    if (isMarked && !bookmarks.includes(shoeId)) {
      bookmarks.push(shoeId);
    } else if (!isMarked) {
      bookmarks = bookmarks.filter(id => id !== shoeId);
    }
    localStorage.setItem("bookmarkedShoes", JSON.stringify(bookmarks));
  }

  function toggleDescription(button) {
    // Show/hide product description text
    const desc = button.closest(".product-line").nextElementSibling;
    const isOpen = desc.style.display === "block";
    desc.style.display = isOpen ? "none" : "block";
    button.textContent = isOpen ? "mehr" : "weniger";
    if (!isOpen) {
      updateVector("mehr", 1);
    }
  }

  function updateVector(type, value) {
    // Update feedback vector values
    const indexMap = { sohle: 0, farbe: 1, schnuersenkel: 2, mehr: 3 };
    if (type in indexMap) {
      vector[indexMap[type]] = value;
    }
  }

  function throttle(func, limit) {
    // Utility function to limit how often a function runs
    let inThrottle;
    return function () {
      if (!inThrottle) {
        func.apply(this, arguments);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  }

  const handleScroll = throttle(async function () {
    // Handle scroll and trigger feedback processing when needed
    const container = getContainer();
    const slides = Array.from(container.getElementsByClassName("slide"));
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
        sohle: vector[0],
        farbe: vector[1],
        schnuersenkel: vector[2],
        mehr: vector[3],
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
        requestAnimationFrame(() => {
          container.scrollTo({ top: 0, behavior: "smooth" });
          setTimeout(() => {
            isProcessingFeedback = false;
          }, 400);
        });
      } catch (err) {
        console.error("Fehler beim Scroll-Feedback:", err);
        isProcessingFeedback = false;
      }
    }
  }, 150);

  async function sendFeedbackAndLoadShoe(feedback) {
    // Send feedback to server and load the next recommended shoe
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

  // Initialize the app
  initShoes();
});
