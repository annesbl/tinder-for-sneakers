from PIL import Image
import os
import numpy as np
import json
from transformers import CLIPProcessor, CLIPModel
from annoy import AnnoyIndex
import torch
import matplotlib.pyplot as plt
import math
from PIL import ImageDraw
from ultralytics import YOLO

yolo_model = YOLO("/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-2/runs/obb/train4/weights/best.pt")  # oder yolov8n-obb.pt


# ==== Einstellungen ====
# Relativer Ordner mit deinen Schuhbildern
IMAGE_DIR = "/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-1/shoes1/"

# Pfad zum Abfragebild (relativ zu Projekt-Root)
EXAMPLE_IMAGE = os.path.join(IMAGE_DIR, "eaba8e75-sneakers_0291.jpg")

# Welchen Teil vergleichen? "sole" oder "laces" oder "color"
PART = "sohle"

# Gewicht für die Farbinformation (muss zum Index passen)
COLOR_WEIGHT = 50.0

# Anzahl der vorgeschlagenen ähnlichen Bilder
TOP_K = 5

# Pfade zu den erzeugten Index- und Mapping-Dateien
COLOR_INDEX_PATH = "color_index.ann"
SOLE_INDEX_PATH = "sole_index.ann"
LACES_INDEX_PATH = "laces_index.ann"
COLOR_MAP = "color_mapping.json"
SOLE_MAP = "sole_mapping.json"
LACES_MAP = "laces_mapping.json"

# ==== Bounding-Box-Funktion ====
#PARTS = {
  #  "sole": {
   #     "rel_center": (0.3, 0.76),
    #    "rel_size":   (0.06, 0.5),
     #   "angle":      92,
      #  "color":      "red"
    #},
    #"laces": {
     #   "rel_center": (0.57, 0.48),
      #  "rel_size":   (0.01, 0.04),
       # "angle":      0,
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


# ==== CLIP-Modell laden ====
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# ==== 1) Abfrage-Vektor berechnen ====
# Bild laden
img = Image.open(EXAMPLE_IMAGE).convert("RGB")
w, h = img.size

# Crop für den gewählten Teil
crop = get_yolo_crop(EXAMPLE_IMAGE, PART)
if crop is None:
    print(f"❌ Kein '{PART}' in Beispielbild erkannt.")
    exit()

# Durchschnittsfarbe extrahieren
avg_color = np.array(crop).mean(axis=(0,1)) / 255.0

# CLIP-Embedding berechnen
inputs = processor(images=crop, return_tensors="pt")
with torch.no_grad():
    embedding = model.get_image_features(**inputs)
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

# Kombinierter Vektor (Embedding + Farbe)
query_vec = np.concatenate([embedding[0].cpu().numpy(), avg_color * COLOR_WEIGHT])

# ==== 2) Index und Mapping laden ====
if PART == "sohle":
    index_file = SOLE_INDEX_PATH
    map_file   = SOLE_MAP
elif PART == "schnuersenkel":
    index_file = LACES_INDEX_PATH
    map_file   = LACES_MAP
elif PART == "farbe":
    index_file = COLOR_INDEX_PATH
    map_file   = COLOR_MAP

# Index laden
dim = query_vec.shape[0]
index = AnnoyIndex(dim, "angular")
index.load(index_file)

# Mapping laden
with open(map_file, "r") as f:
    mapping = json.load(f)

# ==== 3) Suche durchführen ====
ids = index.get_nns_by_vector(query_vec, TOP_K)
results = [mapping[str(i)] for i in ids]

# ==== 4) Ausgabe & Visualisierung ====
print(f"\nÄhnliche {PART}-Bilder für {os.path.basename(EXAMPLE_IMAGE)}:")
for fn in results:
    print(f"  – {fn}")

# Galerie anzeigen
plt.figure(figsize=(12, 3))
for i, fn in enumerate(results):
    path = os.path.join(IMAGE_DIR, fn)
    if not os.path.exists(path):
        continue
    ax = plt.subplot(1, TOP_K, i+1)
    ax.imshow(Image.open(path).convert("RGB"))
    ax.set_title(fn, fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()
