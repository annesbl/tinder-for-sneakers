import os
import sqlite3
import json
import numpy as np
from PIL import Image
from ultralytics import YOLO

import clip  
import torch

#MODEL SETUP 
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

YOLO_MODEL_PATH = "yolov8m_sneakers.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)

IMAGES_DIR = "images_db"

#DB SETUP 
conn = sqlite3.connect('sneakers.db')
c = conn.cursor()
c.execute('DROP TABLE IF EXISTS sneakers')
c.execute('''
    CREATE TABLE IF NOT EXISTS sneakers (
        id TEXT PRIMARY KEY,
        name TEXT,
        description TEXT,
        image_link TEXT,
        light_color TEXT,
        strong_color TEXT,
        embedding_sohle TEXT,
        embedding_schnuersenkel TEXT,
        embedding_farbe TEXT
    )
''')
conn.commit()

#Hilfsfunktion für Embedding 
def get_clip_embedding(img_pil):
    with torch.no_grad():
        image_input = preprocess(img_pil).unsqueeze(0).to(device)
        embedding = clip_model.encode_image(image_input)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten().tolist()  

#Bilddurchlauf 
for img_file in os.listdir(IMAGES_DIR):
    if not img_file.lower().endswith((".jpg", ".png")):
        continue

    img_id = os.path.splitext(img_file)[0]
    image_path = os.path.join(IMAGES_DIR, img_file)
    pil_image = Image.open(image_path).convert("RGB")
    
    #YOLO Inferenz: Teil-Bounding-Boxes extrahieren 
    results = yolo_model(image_path)
    boxes = results[0].boxes
    #0=sohle, 1=schnürsenkel, 2=farbe
    teil_bilder = {"sohle": None, "schnuersenkel": None, "farbe": None}
    for box in boxes:
        teil_idx = int(box.cls)
        xyxy = [int(x) for x in box.xyxy[0]]
        crop = pil_image.crop(xyxy)
        if teil_idx == 0:
            teil_bilder["sohle"] = crop
        elif teil_idx == 1:
            teil_bilder["schnuersenkel"] = crop
        elif teil_idx == 2:
            teil_bilder["farbe"] = crop

    #Embeddings erzeugen oder als [] speichern 
    embedding_sohle = json.dumps(get_clip_embedding(teil_bilder["sohle"])) if teil_bilder["sohle"] is not None else json.dumps([])
    embedding_schnuersenkel = json.dumps(get_clip_embedding(teil_bilder["schnuersenkel"])) if teil_bilder["schnuersenkel"] is not None else json.dumps([])
    embedding_farbe = json.dumps(get_clip_embedding(teil_bilder["farbe"])) if teil_bilder["farbe"] is not None else json.dumps([])

    #DB-Update 
    c.execute('SELECT * FROM sneakers WHERE id=?', (img_id,))
    if not c.fetchone():
        c.execute('''
            INSERT INTO sneakers (
                id, name, description, image_link, light_color, strong_color,
                embedding_sohle, embedding_schnuersenkel, embedding_farbe
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            img_id, "", "", image_path, "", "",
            embedding_sohle, embedding_schnuersenkel, embedding_farbe
        ))

conn.commit()
conn.close()
