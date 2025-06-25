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

# Enable CORS for all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "sneakers.db"

# Serve static files from /static
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Setup Jinja2 templates directory
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

class FeedbackVector(BaseModel):
    """
    Data model for feedback sent from the client.
    Represents user ratings and the current shoe ID.
    """
    sohle: int
    farbe: int
    form: int
    mehr: int
    id: int | None = None

def fetch_shoes_from_db():
    """
    Retrieve all shoes from the database.
    
    Returns:
        list: A list of shoe dictionaries.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, description, image, light, strong FROM sneakers")
    rows = c.fetchall()
    conn.close()

    shoes = []
    for row in rows:
        image_filename = row[3]
        # Remove static prefixes if present in the DB
        image_filename = image_filename.replace("static/", "").replace("Shoes/", "")
        shoes.append({
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "image": image_filename,  # Only the filename; JS will prepend /static/Shoes/
            "light": row[4],
            "strong": row[5]
        })
    return shoes

def fetch_shoe_by_id(shoe_id):
    """
    Retrieve a single shoe by ID from the database.

    Args:
        shoe_id (int): ID of the shoe.

    Returns:
        dict or None: The shoe data if found, else None.
    """
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
    """
    Serve the main HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/shoes")
def get_shoes():
    """
    API endpoint to get all shoes.
    
    Returns:
        list: List of shoes.
    """
    return fetch_shoes_from_db()

@app.get("/shoe/{shoe_id}")
def get_shoe(shoe_id: int):
    """
    API endpoint to get a shoe by its ID.

    Args:
        shoe_id (int): ID of the shoe.

    Returns:
        dict: Shoe data or error message.
    """
    shoe = fetch_shoe_by_id(shoe_id)
    return shoe or {"error": "Not found"}

@app.post("/recommend")
def recommend(vector: FeedbackVector):
    """
    API endpoint to handle feedback and return a recommended shoe ID.
    
    Args:
        vector (FeedbackVector): Feedback from the client.

    Returns:
        dict: Recommended shoe ID.
    """
    print("Feedback:", vector.dict())

    shoes = fetch_shoes_from_db()
    if not shoes:
        return {"error": "No shoes available"}

    # Exclude the current shoe ID from recommendations
    possible_ids = [s["id"] for s in shoes if s["id"] != vector.id]

    # Randomly pick a recommended shoe ID
    recommended_id = random.choice(possible_ids) if possible_ids else vector.id

    # Debugging output
    current_shoe = fetch_shoe_by_id(vector.id)
    recommended_shoe = fetch_shoe_by_id(recommended_id)

    print("➡️ Aktueller Schuh:", current_shoe["name"] if current_shoe else "Unbekannt")
    print("➡️ Empfohlener Schuh:", recommended_shoe["name"] if recommended_shoe else "Unbekannt")

    return {"recommendedId": recommended_id}
