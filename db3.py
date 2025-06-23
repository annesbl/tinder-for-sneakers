import os
import sqlite3
import json
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import clip
import torch

#MODEL SETUP 
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO("yolov8m-seg.pt")
IMAGES_DIR = "images_db"
DB_PATH = "sneakers.db"
SHOW_VISUALIZATION = False  # Setze auf True, wenn du Bounding-Boxes sehen willst

#DB SETUP 
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS sneakers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        description TEXT,
        image_link TEXT UNIQUE,
        light_color TEXT,
        strong_color TEXT,
        embedding_sohle TEXT,
        embedding_schnuersenkel TEXT,
        embedding_farbe TEXT,
        embedding_ganzer_schuh TEXT
    )
''')
conn.commit()

#Hilfsfunktion für Embedding 
def get_clip_embedding(img_pil):
    if img_pil is None:
        return []
    with torch.no_grad():
        image_input = preprocess(img_pil).unsqueeze(0).to(device)
        embedding = clip_model.encode_image(image_input)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten().tolist()

def save_to_db(image_path, embeddings):
    c.execute('SELECT 1 FROM sneakers WHERE image_link = ?', (image_path,))
    if c.fetchone():
        return  # Bild bereits in DB
    c.execute('''
        INSERT INTO sneakers (
            name, description, image_link, light_color, strong_color,
            embedding_sohle, embedding_schnuersenkel, embedding_farbe, embedding_ganzer_schuh
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        "", "", image_path, "", "",
        json.dumps(embeddings["sohle"]),
        json.dumps(embeddings["schnuersenkel"]),
        json.dumps(embeddings["farbe"]),
        json.dumps(embeddings["ganzer_schuh"])
    ))

#Bilddurchlauf 
for img_file in os.listdir(IMAGES_DIR):
    if not img_file.lower().endswith((".jpg", ".png")):
        continue

    image_path = os.path.join(IMAGES_DIR, img_file)

    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠ Fehler beim Laden von {img_file}: {e}")
        continue

    #YOLO Inferenz: Teil-Bounding-Boxes extrahieren 
    results = yolo_model(image_path)

    # Visualisierung der Bounding-Boxes
    if SHOW_VISUALIZATION:
        annotated_frame = results[0].plot()
        plt.imshow(annotated_frame)
        plt.axis('off')  # Achsenbeschriftungen ausblenden
        plt.show()

    #0=sohle, 1=schnürsenkel, 2=farbe, 3=ganzer_schuh
    teil_bilder = {"sohle": None, "schnuersenkel": None, "farbe": None, "ganzer_schuh": None}
    for box in results[0].boxes:
        teil_idx = int(box.cls)
        xyxy = [int(x) for x in box.xyxy[0]]
        crop = pil_image.crop(xyxy)

        if teil_idx == 0:
            teil_bilder["sohle"] = crop
        elif teil_idx == 1:
            teil_bilder["schnuersenkel"] = crop
        elif teil_idx == 2:
            teil_bilder["farbe"] = crop
        elif teil_idx == 3:
            teil_bilder["ganzer_schuh"] = crop
    #Embeddings erzeugen oder als [] speichern 
    embeddings = {teil: get_clip_embedding(img) for teil, img in teil_bilder.items()}
    save_to_db(image_path, embeddings)

conn.commit()
conn.close()