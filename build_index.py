from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import numpy as np
from annoy import AnnoyIndex

# Lade FashionCLIP
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model.eval()

# Bildverzeichnis und Annoy-Index vorbereiten
image_dir = "shoes"
embedding_dim = 512
index = AnnoyIndex(embedding_dim, "angular")
id_to_filename = []

valid_ext = [".jpg", ".jpeg", ".png"]

for idx, filename in enumerate(os.listdir(image_dir)):
    if not any(filename.lower().endswith(ext) for ext in valid_ext):
        continue
    image_path = os.path.join(image_dir, filename)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)[0].cpu().numpy()

    index.add_item(idx, embedding)
    id_to_filename.append(filename)

# Index speichern
index.build(10)
index.save("shoe_index.ann")
np.save("id_to_filename.npy", id_to_filename)