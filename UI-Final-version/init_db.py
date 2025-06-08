# init_db.py
import sqlite3
import json
from pathlib import Path

def init_db():
    BASE_DIR = Path(__file__).resolve().parent
    json_path = BASE_DIR / "shoes.json"
    db_path = BASE_DIR / "sneaker.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("⚠️ Fehler beim Laden der shoes.json:", e)
        data = []

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS sneakers (
        id INTEGER PRIMARY KEY,
        name TEXT,
        description TEXT,
        image_path TEXT,
        bg_color_light TEXT,
        bg_color_strong TEXT
    )
    """)

    for shoe in data[:8]:
        c.execute("""
        INSERT OR REPLACE INTO sneakers (id, name, description, image_path, bg_color_light, bg_color_strong)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            shoe.get("id"),
            shoe.get("name", ""),
            shoe.get("description", ""),
            shoe.get("image", ""),
            shoe.get("bg_color_light", "unknown"),
            shoe.get("bg_color_strong", "unknown")
        ))

    conn.commit()
    conn.close()
    print("✅ Datenbank erfolgreich erstellt oder aktualisiert.")
