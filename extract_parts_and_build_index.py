from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import json
from transformers import CLIPProcessor, CLIPModel
from annoy import AnnoyIndex
import torch

COLOR_WEIGHT = 10.0

# Set Pfade
IMAGE_DIR = "/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers/tinder-for-sneakers/shoes"
SOLE_INDEX_PATH = "sole_index.ann"
LACES_INDEX_PATH = "laces_index.ann"
SOLE_MAP = "sole_mapping.json"
LACES_MAP = "laces_mapping.json"

# Modell laden
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Indexe vorbereiten
EMBED_DIM = 512 + 3  # Embedding + Farbe (RGB)
sole_index = AnnoyIndex(EMBED_DIM, "angular")
laces_index = AnnoyIndex(EMBED_DIM, "angular")
sole_map = {}
laces_map = {}

# Bounding Box Definitionen
def get_boxes(w, h):
    return {
        "sole": (int(w * 0.05), int(h * 0.82), int(w * 0.95), int(h * 0.97)),
        "laces": (int(w * 0.25), int(h * 0.3), int(w * 0.75), int(h * 0.5))
    }

# Alle Bilder durchgehen
for idx, file in enumerate(tqdm(sorted(os.listdir(IMAGE_DIR)))):
    if not file.lower().endswith(".png"):
        continue
    try:
        img_path = os.path.join(IMAGE_DIR, file)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes = get_boxes(w, h)

        for part, box in boxes.items():
            crop = img.crop(box)
            avg_color = np.array(crop).mean(axis=(0, 1)) / 255.0

            inputs = processor(images=crop, return_tensors="pt")
            with torch.no_grad():
                embedding = model.get_image_features(**inputs)
                embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

            
            combined = np.concatenate([embedding[0].numpy(), avg_color * COLOR_WEIGHT])

            if part == "sole":
                sole_index.add_item(idx, combined)
                sole_map[idx] = file
            else:
                laces_index.add_item(idx, combined)
                laces_map[idx] = file

    except Exception as e:
        print(f"Fehler bei {file}: {e}")

# Indexe speichern
sole_index.build(10)
laces_index.build(10)
sole_index.save(SOLE_INDEX_PATH)
laces_index.save(LACES_INDEX_PATH)

with open(SOLE_MAP, "w") as f:
    json.dump(sole_map, f)
with open(LACES_MAP, "w") as f:
    json.dump(laces_map, f)

print("✅ Indexe fertig.")
