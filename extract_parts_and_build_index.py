from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import json
from transformers import CLIPProcessor, CLIPModel
from annoy import AnnoyIndex
import torch
import math
from PIL import ImageDraw

COLOR_WEIGHT = 50.0

# Set Pfade
IMAGE_DIR = "/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers/tinder-for-sneakers/shoes"
COLOR_INDEX_PATH = "color_index.ann"
SOLE_INDEX_PATH = "sole_index.ann"
LACES_INDEX_PATH = "laces_index.ann"
COLOR_MAP = "color_mapping.json"
SOLE_MAP = "sole_mapping.json"
LACES_MAP = "laces_mapping.json"

# Modell laden
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Indexe vorbereiten
EMBED_DIM = 512 + 3  # Embedding + Farbe (RGB)
color_index = AnnoyIndex(EMBED_DIM, "angular")
sole_index = AnnoyIndex(EMBED_DIM, "angular")
laces_index = AnnoyIndex(EMBED_DIM, "angular")
color_map = {}
sole_map = {}
laces_map = {}

# Bounding Box Definitionen
PARTS = {
    "sole": {
        "rel_center": (0.3, 0.76),
        "rel_size":   (0.06, 0.5),
        "angle":      92,
        "color":      "red"
    },
    "laces": {
        "rel_center": (0.57, 0.48),
        "rel_size":   (0.01, 0.04),
        "angle":      0,
        "color":      "green"
    },
    "color": {
        "rel_center": (0.3, 0.5),
        "rel_size":   (0.1, 0.2),
        "angle":      0,
        "color":      "orange"
    }
}

def rotate_point(x, y, angle, cx, cy):
    """Rotiert einen Punkt (x, y) um (cx, cy) mit dem angegebenen Winkel (in Grad)."""
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    x_rot = cos_a * (x - cx) - sin_a * (y - cy) + cx
    y_rot = sin_a * (x - cx) + cos_a * (y - cy) + cy
    return x_rot, y_rot


def get_boxes(part, w, h):
    try:
        cfg = PARTS[part]
        cx, cy = cfg["rel_center"]
        rw, rh = cfg["rel_size"]
        angle = cfg["angle"]
        
        # Berechne die ursprünglichen, ungeflippten Koordinaten
        x1 = (cx - rw / 2) * w
        y1 = (cy - rh / 2) * h
        x2 = (cx + rw / 2) * w
        y2 = (cy + rh / 2) * h
        
        # Mittelpunkte der Box (für Rotation)
        box_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Punkte der Box vor der Rotation
        box_points = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
        
        # Alle Punkte der Box rotieren
        rotated_points = [rotate_point(x, y, angle, box_center[0], box_center[1]) for x, y in box_points]
        
        # Bestimme die neue (rotierte) Box
        min_x = min([p[0] for p in rotated_points])
        min_y = min([p[1] for p in rotated_points])
        max_x = max([p[0] for p in rotated_points])
        max_y = max([p[1] for p in rotated_points])
        
        # Rückgabe der neuen Box-Koordinaten
        return (int(min_x), int(min_y), int(max_x), int(max_y))
    
    except KeyError:
        raise ValueError(f"Unbekannter Part: '{part}'. Verfügbare Teile: {list(PARTS.keys())}")





# Alle Bilder durchgehen
for idx, file in enumerate(tqdm(sorted(os.listdir(IMAGE_DIR)))):
    if not file.lower().endswith(".png"):
        continue
    try:
        img_path = os.path.join(IMAGE_DIR, file)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Durch alle Teile iterieren und Boxen berechnen
        for part in PARTS:
            # Holen der Box für jedes Teil
            box = get_boxes(part, w, h)

            crop = img.crop(box)
            avg_color = np.array(crop).mean(axis=(0, 1)) / 255.0

            # Verarbeitung mit dem Modell
            inputs = processor(images=crop, return_tensors="pt")
            with torch.no_grad():
                embedding = model.get_image_features(**inputs)
                embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

            # Kombinieren der Embeddings und Farbwerte
            combined = np.concatenate([embedding[0].numpy(), avg_color * COLOR_WEIGHT])

            # Je nach Teil, zum entsprechenden Index hinzufügen
            if part == "sole":
                sole_index.add_item(idx, combined)
                sole_map[idx] = file
            elif part == "laces":
                laces_index.add_item(idx, combined)
                laces_map[idx] = file
            elif part == "color":
                color_index.add_item(idx, combined)
                color_map[idx] = file

    except Exception as e:
        print(f"Fehler bei {file}: {e}")

# Indexe speichern
color_index.build(10)
sole_index.build(10)
laces_index.build(10)
color_index.save(COLOR_INDEX_PATH)
sole_index.save(SOLE_INDEX_PATH)
laces_index.save(LACES_INDEX_PATH)

# Mappen speichern
with open(COLOR_MAP, "w") as f:
    json.dump(color_map, f)
with open(SOLE_MAP, "w") as f:
    json.dump(sole_map, f)
with open(LACES_MAP, "w") as f:
    json.dump(laces_map, f)

print("✅ Indexe fertig.")
