import os
import sqlite3
import json
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import clip
import torch

# --- SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO("yolov8m-seg.pt")
IMAGES_DIR = "static/Shoes"
DB_PATH = "sneakers.db"
META_PATH = "metadata.json"
SHOW_VISUALIZATION = False

# --- Metadaten laden ---
with open(META_PATH, "r") as f:
    metadata_list = json.load(f)

# Mapping mit nur Dateinamen als Key
metadata_dict = {os.path.basename(entry["image"]): entry for entry in metadata_list}

# --- DB SETUP ---
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS sneakers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        description TEXT,
        image TEXT UNIQUE,
        light TEXT,
        strong TEXT,
        embedding_sohle TEXT,
        embedding_schnuersenkel TEXT,
        embedding_farbe TEXT,
        embedding_ganzer_schuh TEXT
    )
''')
conn.commit()

# --- FUNKTIONEN ---
def get_clip_embedding(img_pil):
    if img_pil is None:
        return []
    with torch.no_grad():
        image_input = preprocess(img_pil).unsqueeze(0).to(device)
        embedding = clip_model.encode_image(image_input)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten().tolist()

def save_to_db(image_path, embeddings, meta):
    c.execute('SELECT 1 FROM sneakers WHERE image = ?', (image_path,))
    if c.fetchone():
        return
    c.execute('''
        INSERT INTO sneakers (
            name, description, image, light, strong,
            embedding_sohle, embedding_schnuersenkel, embedding_farbe, embedding_ganzer_schuh
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        meta.get("name", ""), 
        meta.get("description", ""), 
        image_path,
        meta.get("light", ""), 
        meta.get("strong", ""),
        json.dumps(embeddings["sohle"]),
        json.dumps(embeddings["schnuersenkel"]),
        json.dumps(embeddings["farbe"]),
        json.dumps(embeddings["ganzer_schuh"])
    ))

# --- HAUPTSCHLEIFE ---
for img_file in os.listdir(IMAGES_DIR):
    if not img_file.lower().endswith((".jpg", ".png")):
        continue

    image_path = os.path.join(IMAGES_DIR, img_file)
    filename = os.path.basename(image_path)

    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠ Fehler beim Laden von {img_file}: {e}")
        continue

    results = yolo_model(image_path)

    if SHOW_VISUALIZATION:
        annotated_frame = results[0].plot()
        plt.imshow(annotated_frame)
        plt.axis('off')
        plt.show()

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

    embeddings = {teil: get_clip_embedding(img) for teil, img in teil_bilder.items()}
    meta = metadata_dict.get(filename, {
        "name": "", "description": "", "light": "", "strong": ""
    })
    save_to_db(image_path, embeddings, meta)

conn.commit()
conn.close()
