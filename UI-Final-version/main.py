from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, random
from pathlib import Path

app = FastAPI()

# CORS – nur wenn du JS nutzt, das mit API spricht
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
templates = Jinja2Templates(directory="/templates")

# 🔁 JSON-Daten laden
#def load_shoes():
#    try:
#        with open("shoes.json", "r", encoding="utf-8") as f:
#            return json.load(f)
#    except:
#        return []

#shoes_data = load_shoes()

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

@app.post("/recommend")
def recommend(vector: FeedbackVector):
    print("Feedback:", vector.dict())
    if not shoes_data:
        return {"error": "Keine Schuhe"}
    recommended_id = random.choice([s["id"] for s in shoes_data])
    return {"recommendedId": recommended_id}

@app.get("/shoe/{shoe_id}")
def get_shoe(shoe_id: int):
    shoe = next((s for s in shoes_data if s["id"] == shoe_id), None)
    return shoe or {"error": "Nicht gefunden"}
