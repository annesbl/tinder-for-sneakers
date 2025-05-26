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
from ultralytics import YOLO

yolo_model = YOLO("/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-2/runs/obb/train4/weights/best.pt")  # oder yolov8n-obb.pt


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
#PARTS = {
    #"sole": {
   #     "rel_center": (0.3, 0.76),
    #    "rel_size":   (0.06, 0.5),
     #   "angle":      92,
      #  "color":      "red"
    #},
    #"laces": {
       ## "rel_center": (0.57, 0.48),
        #"rel_size":   (0.01, 0.04),
        #"angle":      0,
        #"color":      "green"
    #},
    #"color": {
     #   "rel_center": (0.3, 0.5),
      #  "rel_size":   (0.1, 0.2),
       # "angle":      0,
        #"color":      "orange"
   # }
#}

def rotate_point(x, y, angle, cx, cy):
    """Rotiert einen Punkt (x, y) um (cx, cy) mit dem angegebenen Winkel (in Grad)."""
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    x_rot = cos_a * (x - cx) - sin_a * (y - cy) + cx
    y_rot = sin_a * (x - cx) + cos_a * (y - cy) + cy
    return x_rot, y_rot


def get_yolo_crop(image_path, part_name):
    results = yolo_model.predict(image_path, device="cpu", conf=0.1)
    print("📦 Erkannt:", [yolo_model.names[int(b.cls)] for r in results if r.boxes for b in r.boxes] if results else "nichts")

    image = Image.open(image_path).convert("RGB")

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls)
            cls_name = yolo_model.names[cls]
            if cls_name == part_name:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                return image.crop(xyxy)

    print(f"❌ Kein '{part_name}' erkannt in {os.path.basename(image_path)}.")
    return None




# Alle Bilder durchgehen
for idx, file in enumerate(tqdm(sorted(os.listdir(IMAGE_DIR)))):
    if not file.lower().endswith(".png"):
        continue
    try:
        img_path = os.path.join(IMAGE_DIR, file)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Durch alle Teile iterieren und Boxen berechnen
        for part in ["sohle", "schnuersenkel", "farbe"]:
            # Holen der Box für jedes Teil
            crop = get_yolo_crop(img_path, part)
            if crop is None:
                print(f"Kein '{part}' erkannt in {file}. Übersprungen.")
                continue

            avg_color = np.array(crop).mean(axis=(0, 1)) / 255.0

            # Verarbeitung mit dem Modell
            inputs = processor(images=crop, return_tensors="pt")
            with torch.no_grad():
                embedding = model.get_image_features(**inputs)
                embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

            # Kombinieren der Embeddings und Farbwerte
            combined = np.concatenate([embedding[0].numpy(), avg_color * COLOR_WEIGHT])

            # Je nach Teil, zum entsprechenden Index hinzufügen
            if part == "sohle":
                sole_index.add_item(idx, combined)
                sole_map[idx] = file
            elif part == "schnuersenkel":
                laces_index.add_item(idx, combined)
                laces_map[idx] = file
            elif part == "farbe":
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
