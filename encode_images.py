from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import numpy as np

model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# Verzeichnisse
image_dir = "data"
output_dir = "embeddings"
os.makedirs(output_dir, exist_ok=True)

# Bild → Embedding
def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding[0].numpy()

# Alle Bilder im Ordner verarbeiten
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(image_dir, filename)
        try:
            emb = get_embedding(path)
            np.save(os.path.join(output_dir, filename + ".npy"), emb)
            print(f"[✓] Saved embedding for {filename}")
        except Exception as e:
            print(f"[!] Fehler bei {filename}: {e}")
