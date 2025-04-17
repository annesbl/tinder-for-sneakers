import kagglehub
import os
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

# -----------------------------------------------------------
# 1. Download von Kaggle
# -----------------------------------------------------------

print("Lade Sneaker-Datensatz von Kaggle ...")
path = kagglehub.dataset_download("die9origephit/nike-adidas-and-converse-imaged")

# Zielordner vorbereiten
image_dir = "data"
os.makedirs(image_dir, exist_ok=True)

# Nur gültige Bilddateien
img_extensions = (".jpg", ".jpeg", ".png")
count = 0

print("Konvertiere und speichere Bilder ...")
for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith(img_extensions):
            src = os.path.join(root, file)
            dst = os.path.join(image_dir, file.replace(".png", ".jpg"))

            try:
                img = Image.open(src).convert("RGB")
                img.save(dst, format="JPEG")
                count += 1
                print(f"[{count}] Bild gespeichert: {file}")
            except Exception as e:
                print(f"Fehler bei {file}: {e}")

print(f"Fertig. {count} Bilder im Ordner: {image_dir}/")

# -----------------------------------------------------------
# 2. Embeddings mit FashionCLIP erzeugen
# -----------------------------------------------------------

print("Starte Embedding mit FashionCLIP ...")

# Modell laden
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# Embedding-Ordner
embedding_dir = "embeddings"
os.makedirs(embedding_dir, exist_ok=True)

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding[0].numpy()

# Bilder durchgehen und Embedding erzeugen
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        path = os.path.join(image_dir, filename)
        try:
            emb = get_embedding(path)
            np.save(os.path.join(embedding_dir, filename + ".npy"), emb)
            print(f"Embedding gespeichert für: {filename}")
        except Exception as e:
            print(f"Fehler bei {filename}: {e}")

print("Alle Embeddings erstellt und gespeichert in:", embedding_dir)
