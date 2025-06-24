from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sqlite3
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "sneakers.db"

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

class FeedbackVector(BaseModel):
    sohle: int
    farbe: int
    schnuersenkel: int
    mehr: int
    id: int | None = None

def fetch_shoes_from_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, description, image, light, strong FROM sneakers")
    rows = c.fetchall()
    conn.close()
    shoes = []
    for row in rows:
        image_filename = row[3]
        # Entferne statische Präfixe, falls sie in DB gespeichert sind
        image_filename = image_filename.replace("static/", "").replace("Shoes/", "")
        shoes.append({
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "image": image_filename,  # Nur Dateiname, im JS ergänzt mit /static/Shoes/
            "light": row[4],
            "strong": row[5]
        })
    return shoes

def fetch_shoe_by_id(shoe_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, description, image, light, strong FROM sneakers WHERE id = ?", (shoe_id,))
    row = c.fetchone()
    conn.close()
    if row:
        image_filename = row[3].replace("static/", "").replace("Shoes/", "")
        return {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "image": image_filename,
            "light": row[4],
            "strong": row[5]
        }
    return None

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/shoes")
def get_shoes():
    return fetch_shoes_from_db()

@app.get("/shoe/{shoe_id}")
def get_shoe(shoe_id: int):
    shoe = fetch_shoe_by_id(shoe_id)
    return shoe or {"error": "Nicht gefunden"}

@app.post("/recommend")
def recommend(vector: FeedbackVector):
    print("Feedback:", vector.dict())
    shoes = fetch_shoes_from_db()
    if not shoes:
        return {"error": "Keine Schuhe"}
    possible_ids = [s["id"] for s in shoes if s["id"] != vector.id]
    recommended_id = random.choice(possible_ids) if possible_ids else vector.id
    return {"recommendedId": recommended_id}
