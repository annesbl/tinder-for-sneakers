from PIL import Image
import numpy as np
import json
from transformers import CLIPProcessor, CLIPModel
from annoy import AnnoyIndex
import torch
import matplotlib.pyplot as plt

# Parameter
EXAMPLE_IMAGE = "images/B25R11-U12S01L15-MIN_high_res.png"
PART = "sole"  # "laces" oder "sole"
INDEX_FILE = f"{PART}_index.ann"
MAPPING_FILE = f"{PART}_mapping.json"

# Bounding Boxen
def get_box(part, w, h):
    if part == "sole":
        return (int(w * 0.05), int(h * 0.82), int(w * 0.95), int(h * 0.97))
    elif part == "laces":
        return (int(w * 0.25), int(h * 0.3), int(w * 0.75), int(h * 0.5))

# Modell laden
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Beispielbild vorbereiten
img = Image.open(EXAMPLE_IMAGE).convert("RGB")
w, h = img.size
box = get_box(PART, w, h)
crop = img.crop(box)
avg_color = np.array(crop).mean(axis=(0, 1)) / 255.0

# Embedding
inputs = processor(images=crop, return_tensors="pt")
with torch.no_grad():
    embedding = model.get_image_features(**inputs)
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

query_vector = np.concatenate([embedding[0].numpy(), avg_color])

# Index laden
index = AnnoyIndex(515, "angular")
index.load(INDEX_FILE)
with open(MAPPING_FILE, "r") as f:
    mapping = json.load(f)

# Ähnliche Bilder suchen
similar_ids = index.get_nns_by_vector(query_vector, 5)
similar_files = [mapping[str(i)] for i in similar_ids]

# Ergebnisse anzeigen
print(f"Ähnliche {PART}-Bilder:")
for f in similar_files:
    print(f"- {f}")

# Bilder anzeigen
plt.figure(figsize=(12, 3))
for i, fname in enumerate(similar_files):
    img = Image.open(os.path.join("images", fname)).convert("RGB")
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(fname)
    plt.axis("off")
plt.tight_layout()
plt.show()
