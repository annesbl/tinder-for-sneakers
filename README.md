# TikTok for Sneaker

A web-based sneaker discovery app that allows users to explore a collection of sneaker designs in a TikTok-like swipe experience.

Users can:

- Swipe vertically through sneaker cards and view each design in detail.
- Like specific shoe components, such as the sole, color, or laces, using interactive heart buttons that change color on click.
- Bookmark favorite sneakers with a single click; bookmarks are saved in the browser's local storage for quick reference.
- Toggle detailed descriptions using a “mehr” button to reveal additional product information.
- Automatically receive new sneaker recommendations as they scroll to the next card.

---

## 🚀 **Features**

* **Dynamic sneaker data from SQLite (`sneakers.db`)**
* **FastAPI backend** with API endpoints:

  * `/shoes` — Get all sneakers
  * `/shoe/{id}` — Get details for a specific sneaker
  * `/recommend` — Get recommended next sneaker (currently random)
* **Frontend** with:

  * Swipe-like scrollable interface
  * Like buttons (sole, color, laces) with heart icon toggle
  * Bookmark button (saved in localStorage)
  * "More" button to toggle description
  * Auto-loads a new sneaker on scroll
* **YOLO + CLIP** used for shoe part detection and embedding generation
* Images and icons served from `/static/`

---

## 📂 **Project structure**

```

TINDER-FOR-SNEAKERS/
├── recommender/                        # Recommendation logic (Annoy index + helper scripts)
│   ├── indices/                        # Saved Annoy index files
│   ├── create\_annoy.py                # Script to build Annoy index from embeddings
│   └── recommender.py                  # Logic for finding similar sneakers
├── UI-DB/                              # App frontend + FastAPI backend
│   ├── static/                         # Frontend assets
│   │   ├── icons/                      # Icons (heart, bookmark)
│   │   │   ├── heart-white.png
│   │   │   ├── heart-red.png
│   │   │   ├── bookmark.png
│   │   │   ├── bookmark\_filled.png
│   │   │   └── website.png
│   │   ├── Shoes/                      # Sneaker image files
│   │   ├── metadata.json               # Metadata for initial DB fill (name, colors, etc.)
│   │   ├── script.js                   # JS logic for rendering, scroll, likes, bookmarks
│   │   └── style.css                   # CSS for app design (mobile-style, snap scroll, etc.)
│   ├── templates/
│   │   └── index.html                  # Main HTML template with app layout
│   ├── init\_db.py                     # Script to create & populate SQLite DB with embeddings
│   ├── main.py                         # FastAPI app with API routes and static file serving
│   └── sneakers.db                     # Generated SQLite database storing sneaker data
├── YOLO/                               # YOLO part detection scripts & config
│   ├── anns/                           # YOLO annotations for training/testing
│   ├── models/                         # YOLO model files (weights, configs)
│   └── yolo\_utils.py                  # Helper functions for YOLO processing
├── decision_tree/                      # Conceptual decision logic (not yet implemented)
│ └── decision_tree_concept.jpg         # Visual concept of rule-based sneaker recommendation
├── .gitignore                          # Git ignore rules
├── README.md                           # This documentation file
└── requirements.txt                    # Python dependencies (FastAPI, Uvicorn, etc.)

```

---

## ⚡ **How to run**

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Initialize DB (only if you want to recreate it):

```bash
cd UI-DB
python init_db.py
```

3. Start the server:

```bash
uvicorn main:app --reload
```

4. Open your browser:

```
http://127.0.0.1:8000
```

---

## 🛠 **Tech stack**

* **Backend:** FastAPI, SQLite
* **Frontend:** HTML, CSS, Vanilla JS
* **AI:** YOLO (for part detection), CLIP (for embeddings)
* **Recommendation:** Annoy index (planned / partial)

---

## 🌱 **Next Steps & Future Potential**

The current app lays the foundation for a powerful, component-based sneaker recommendation system. Possible directions for future development include:

### 🔍 Smarter Recommendations

- **Weight sneaker parts** differently in the similarity calculation (e.g., prioritize soles if users click on soles frequently)
- **Incorporate user history** to refine and personalize the recommendation path over time
- Build an **explainable recommendation engine**, so users understand _why_ a sneaker was recommended (e.g., “Because you liked yellow soles and pink uppers”)

> ⚠️ **Note**: During testing, the model occasionally misclassified sneaker parts – e.g., confusing the sole with the colored upper area. This suggests a need for better label quality and clearer class distinction. Additionally, **AI-generated or stylized (animated) sneakers are currently not detected at all**, as they deviate too much from real-world training data.

### 🧑‍💻 Feature Extensions

- Add **persistent user accounts** to store bookmarks and preference profiles across sessions
- Create an **admin interface or upload API** for adding new sneakers and regenerating indices dynamically

### 🌐 Deployment

- Deploy the app publicly using platforms like **PythonAnywhere**, **Vercel**, or **AWS**  
- Optimize the system for speed, scalability, and mobile responsiveness

---
