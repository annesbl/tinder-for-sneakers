from PIL import Image
import os
import numpy as np
import json
from transformers import CLIPProcessor, CLIPModel
from annoy import AnnoyIndex
import torch
import matplotlib.pyplot as plt

# ==== Einstellungen ====
# Relativer Ordner mit deinen Schuhbildern
IMAGE_DIR = "/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers/tinder-for-sneakers/shoes"

# Pfad zum Abfragebild (relativ zu Projekt-Root)
EXAMPLE_IMAGE = "shoes/B25R11-U18S13L43-MIN.png"

# Welchen Teil vergleichen? "sole" oder "laces"
PART = "sole"

# Gewicht für die Farbinformation (muss zum Index passen)
COLOR_WEIGHT = 10.0

# Anzahl der vorgeschlagenen ähnlichen Bilder
TOP_K = 5

# Pfade zu den erzeugten Index- und Mapping-Dateien
SOLE_INDEX_PATH = "sole_index.ann"
LACES_INDEX_PATH = "laces_index.ann"
SOLE_MAP = "sole_mapping.json"
LACES_MAP = "laces_mapping.json"

# ==== Bounding-Box-Funktion ====
def get_box(part, w, h):
    if part == "sole":
        return (int(w * 0.05), int(h * 0.82), int(w * 0.95), int(h * 0.97))
    else:  # "laces"
        return (int(w * 0.25), int(h * 0.3), int(w * 0.75), int(h * 0.5))

# ==== CLIP-Modell laden ====
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# ==== 1) Abfrage-Vektor berechnen ====
# Bild laden
img = Image.open(EXAMPLE_IMAGE).convert("RGB")
w, h = img.size

# Crop für den gewählten Teil
box = get_box(PART, w, h)
crop = img.crop(box)

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
if PART == "sole":
    index_file = SOLE_INDEX_PATH
    map_file   = SOLE_MAP
else:
    index_file = LACES_INDEX_PATH
    map_file   = LACES_MAP

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
