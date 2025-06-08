from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, random
from pathlib import Path
class FeedbackVector(BaseModel):
    sohle: int
    farbe: int
    schnuersenkel: int
    mehr: int
    id: int | None = None  # Schuh-ID, optional
app = FastAPI()

# CORS – nur für Entwicklung nötig, wenn du Frontend und Backend getrennt startest
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = Path(__file__).resolve().parent

# ➕ Statische Dateien wie style.css, script.js, images
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# ➕ HTML-Templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 🔁 JSON-Daten laden
def load_shoes():
    try:
        with open(BASE_DIR / "shoes.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("Fehler beim Laden von shoes.json:", e)
        return []

shoes_data = load_shoes()

# 🏠 Startseite mit HTML
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 📦 Empfehlung
class FeedbackVector(BaseModel):
    sohle: int
    farbe: int
    schnuersenkel: int
    mehr: int
    bookmark: int

@app.get("/shoe/{shoe_id}")
def get_shoe(shoe_id: int):
    shoe = next((s for s in shoes_data if s["id"] == shoe_id), None)
    return shoe or {"error": "Nicht gefunden"}

@app.get("/shoes")
def get_shoes():
    return shoes_data
@app.post("/recommend")
def recommend(vector: FeedbackVector):
    print("Feedback:", vector.dict())
    # Zugriff auf die Schuh-ID:
    shoe_id = vector.id
    print(f"Feedback gehört zu Schuh-ID: {shoe_id}")

    if not shoes_data:
        return {"error": "Keine Schuhe"}
    # Beispiel: Empfehle einen zufälligen Schuh, der NICHT der aktuelle ist
    possible_ids = [s["id"] for s in shoes_data if s["id"] != shoe_id]
    if not possible_ids:
        recommended_id = shoe_id  # Fallback: den aktuellen Schuh zurückgeben
    else:
        recommended_id = random.choice(possible_ids)
    return {"recommendedId": recommended_id}
