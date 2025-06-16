import os
import sqlite3
from ultralytics import YOLO

# Datenbank initialisieren
conn = sqlite3.connect('sneakers.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS sneakers (
        id TEXT PRIMARY KEY,
        name TEXT,
        description TEXT,
        image_link TEXT,
        light_color TEXT,
        strong_color TEXT,
        pfad_sohle_ann TEXT,
        pfad_schnuersenkel_ann TEXT,
        pfad_farbe_ann TEXT
    )
''')
conn.commit()

IMAGES_DIR = "images_db"
ANN_DIR = "anns"
YOLO_MODEL_PATH = "yolov8m_sneakers.pt"
MODEL = YOLO(YOLO_MODEL_PATH)

for img_file in os.listdir(IMAGES_DIR):
    if img_file.lower().endswith((".jpg", ".png")):
        img_id = os.path.splitext(img_file)[0]
        image_path = os.path.join(IMAGES_DIR, img_file)

        # YOLO inference (Dummy-Example!)
        results = MODEL(image_path)
        # Ann-Dateien pro Teil speichern
        sohle_ann = f"{ANN_DIR}/sohle_{img_id}.ann"
        schnuer_ann = f"{ANN_DIR}/schnuer_{img_id}.ann"
        farbe_ann = f"{ANN_DIR}/farbe_{img_id}.ann"
        # Hier würde man die Ergebnisse der YOLO-Erkennung reinschreiben!
        with open(sohle_ann, "w") as f: f.write("sohle ann content")
        with open(schnuer_ann, "w") as f: f.write("schnuersenkel ann content")
        with open(farbe_ann, "w") as f: f.write("farbe ann content")

        # Evtl. schon Eintrag vorhanden?
        c.execute('SELECT * FROM sneakers WHERE id=?', (img_id,))
        if not c.fetchone():
            c.execute('''
                INSERT INTO sneakers (
                    id, name, description, image_link, light_color, strong_color,
                    pfad_sohle_ann, pfad_schnuersenkel_ann, pfad_farbe_ann
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                img_id, "", "", image_path, "", "",
                sohle_ann, schnuer_ann, farbe_ann
            ))
conn.commit()
conn.close()
