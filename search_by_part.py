from annoy import AnnoyIndex
import pickle
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

PART = "sole"  # oder "sole"
QUERY_IMAGE = "shoes/B25R11-U16S06L21-MIN.png"

# Bounding Box wie oben
PARTS = {
    "laces": (0.25, 0.15, 0.75, 0.4),
    "sole": (0.2, 0.75, 0.8, 0.95),
}

# Lade Modell
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# Lade Index und Map
index = AnnoyIndex(512, "angular")
index.load(f"part_embeddings/{PART}.ann")
with open(f"part_embeddings/{PART}_map.pkl", "rb") as f:
    id_map = pickle.load(f)

# Bild vorbereiten
image = Image.open(QUERY_IMAGE).convert("RGB")
w, h = image.size
x1, y1, x2, y2 = [int(w * PARTS[PART][i]) if i % 2 == 0 else int(h * PARTS[PART][i]) for i in range(4)]
crop = image.crop((x1, y1, x2, y2))

# Vektorisieren
inputs = processor(images=crop, return_tensors="pt").to(device)
with torch.no_grad():
    embedding = model.get_image_features(**inputs)
    embedding = embedding.cpu().numpy()[0]

# Ähnliche finden
neighbors = index.get_nns_by_vector(embedding, 10)
print("\nÄhnliche Schuhe (nach Teil: {}):".format(PART))
for i in neighbors:
    print(id_map[i])

