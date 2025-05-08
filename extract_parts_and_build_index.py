from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import json
from transformers import CLIPProcessor, CLIPModel
from annoy import AnnoyIndex
import torch

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
def get_boxes(w, h):
    return {
        "sole": (
            int((0.3 - 0.06/2) * w),
            int((0.76 - 0.5/2) * h),
            int((0.3 + 0.06/2) * w),
            int((0.76 + 0.5/2) * h)
        ),
        "laces": (
            int((0.57 - 0.01/2) * w),
            int((0.48 - 0.04/2) * h),
            int((0.57 + 0.01/2) * w),
            int((0.48 + 0.04/2) * h)
        ),
        "color": (
            int((0.3 - 0.1/2) * w),
            int((0.5 - 0.2/2) * h),
            int((0.3 + 0.1/2) * w),
            int((0.5 + 0.2/2) * h)
        )
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

with open(COLOR_MAP, "w") as f:
    json.dump(color_map, f)
with open(SOLE_MAP, "w") as f:
    json.dump(sole_map, f)
with open(LACES_MAP, "w") as f:
    json.dump(laces_map, f)

print("✅ Indexe fertig.")
