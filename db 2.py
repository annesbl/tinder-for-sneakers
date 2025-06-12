import os
import sqlite3
import json

# Klassen (von classes.txt)
classes = ["Farbe", "Schnürsenkel", "Sohle", "ganzer_schuh"]

# Pfade
base_dir = "data"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")
db_path = os.path.join(base_dir, "sneakers.db")

# DB Verbindung
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Tabellen erstellen
cur.execute("""CREATE TABLE IF NOT EXISTS sneakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    image_path TEXT,
    color_light TEXT,
    color_dark TEXT
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sneaker_id INTEGER,
    class_name TEXT,
    polygon TEXT,
    FOREIGN KEY(sneaker_id) REFERENCES sneakers(id)
)""")

# Alle Labels durchgehen
for filename in os.listdir(labels_dir):
    if not filename.endswith(".txt"):
        continue

    image_id = os.path.splitext(filename)[0]
    label_path = os.path.join(labels_dir, filename)
    image_filename = f"{image_id}.jpg"
    image_path = os.path.join(images_dir, image_filename)

    if not os.path.exists(image_path):
        print(f"⚠️  Bild fehlt für {image_id}, überspringe…")
        continue

    # Sneaker-Eintrag
    cur.execute("INSERT INTO sneakers (name, image_path, color_light, color_dark) VALUES (?, ?, ?, ?)",
                (f"Sneaker {image_id}", image_path, "#f5f7fa", "#1a1a1a"))
    sneaker_id = cur.lastrowid

    # Label-Polygon-Daten einfügen
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:  # class_id + mindestens 1 Punkt
                continue

            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            polygon = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
            class_name = classes[class_id]

            cur.execute("INSERT INTO labels (sneaker_id, class_name, polygon) VALUES (?, ?, ?)",
                        (sneaker_id, class_name, json.dumps(polygon)))

    print(f"✅ Importiert: {image_id}")

# Abschluss
conn.commit()
conn.close()
print("🎉 Fertig! Alle Daten importiert.")