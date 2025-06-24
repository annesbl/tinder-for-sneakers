import os
import sys
import sqlite3
import json
from PIL import Image
import matplotlib.pyplot as plt
import clip
import torch

# --- Dynamically add the project root to Python path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from YOLO.yolo_utils import detect_parts

# --- SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
IMAGES_DIR = "UI-DB/static/Shoes"
DB_PATH = "UI-DB/sneakers.db"
META_PATH = "UI-DB/static/metadata.json"
SHOW_VISUALIZATION = False

# --- Load metadata ---
with open(META_PATH, "r") as f:
    metadata_list = json.load(f)

# Create a lookup dictionary for metadata based on image filename
metadata_dict = {os.path.basename(entry["image"]): entry for entry in metadata_list}

# --- Database setup ---
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Create table if it does not exist
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

def get_clip_embedding(img_pil):
    """
    Generate a normalized CLIP embedding for the given PIL image.
    
    Args:
        img_pil (PIL.Image or None): The image to process.
    
    Returns:
        list: A flattened list representing the embedding vector, or empty list if input is None.
    """
    if img_pil is None:
        return []
    with torch.no_grad():
        image_input = preprocess(img_pil).unsqueeze(0).to(device)
        embedding = clip_model.encode_image(image_input)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten().tolist()

def save_to_db(image_path, embeddings, meta):
    """
    Insert a new sneaker entry into the database if it does not already exist.
    
    Args:
        image_path (str): Path to the image file.
        embeddings (dict): Dictionary containing embeddings for parts.
        meta (dict): Dictionary containing metadata (name, description, colors).
    """
    c.execute('SELECT 1 FROM sneakers WHERE image = ?', (image_path,))
    if c.fetchone():
        # Skip if the image already exists in the database
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

# --- Main loop over images ---
for img_file in os.listdir(IMAGES_DIR):
    if not img_file.lower().endswith((".jpg", ".png")):
        continue  # Skip non-image files

    image_path = os.path.join(IMAGES_DIR, img_file)
    filename = os.path.basename(image_path)

    # Run YOLO detection to extract parts
    results, teil_bilder = detect_parts(image_path)
    if results is None:
        continue  # Skip if detection failed

    # Optionally show visualized detections
    if SHOW_VISUALIZATION:
        annotated_frame = results[0].plot()
        plt.imshow(annotated_frame)
        plt.axis('off')
        plt.show()

    # Compute embeddings for each detected part
    embeddings = {teil: get_clip_embedding(img) for teil, img in teil_bilder.items()}

    # Retrieve metadata or use defaults
    meta = metadata_dict.get(filename, {
        "name": "", "description": "", "light": "", "strong": ""
    })

    # Save data to the database
    save_to_db(image_path, embeddings, meta)

# Commit all changes and close the connection
conn.commit()
conn.close()
