import kagglehub
import os
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

# -----------------------------------------------------------
# 1. Download sneaker dataset from Kaggle
# -----------------------------------------------------------

print("Downloading sneaker dataset from Kaggle...")
path = kagglehub.dataset_download("die9origephit/nike-adidas-and-converse-imaged")

# Prepare output directory
image_dir = "data"
os.makedirs(image_dir, exist_ok=True)

# Valid image extensions
img_extensions = (".jpg", ".jpeg", ".png")
count = 0

print("Converting and saving images...")
for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith(img_extensions):
            src = os.path.join(root, file)
            dst = os.path.join(image_dir, file.replace(".png", ".jpg"))

            try:
                img = Image.open(src).convert("RGB")
                img.save(dst, format="JPEG")
                count += 1
                print(f"[{count}] Saved image: {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

print(f"Done. {count} images saved to: {image_dir}/")

# -----------------------------------------------------------
# 2. Generate embeddings using FashionCLIP
# -----------------------------------------------------------

print("Generating embeddings using FashionCLIP...")

# Load model and processor
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# Prepare embedding output directory
embedding_dir = "embeddings"
os.makedirs(embedding_dir, exist_ok=True)

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding[0].numpy()

# Generate embeddings for all images
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        path = os.path.join(image_dir, filename)
        try:
            emb = get_embedding(path)
            np.save(os.path.join(embedding_dir, filename + ".npy"), emb)
            print(f"Saved embedding for: {filename}")
        except Exception as e:
            print(f"Error embedding {filename}: {e}")

print("All embeddings generated and saved to:", embedding_dir)
