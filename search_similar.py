from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import os
from annoy import AnnoyIndex
import matplotlib.pyplot as plt

# Lade FashionCLIP
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model.eval()

# Lade Suchbild
query_path = "query/test_shoe3.jpg"
image = Image.open(query_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    query_embedding = model.get_image_features(**inputs)[0].cpu().numpy()

# Lade Index und Dateinamenliste
index = AnnoyIndex(512, "angular")
index.load("shoe_index.ann")
id_to_filename = np.load("id_to_filename.npy", allow_pickle=True)

# Ähnliche Bilder finden
top_k = 5
indices = index.get_nns_by_vector(query_embedding, top_k)

# Ergebnisse anzeigen
fig, axs = plt.subplots(1, top_k, figsize=(15, 5))
for i, idx in enumerate(indices):
    img_path = os.path.join("shoes", id_to_filename[idx])
    img = Image.open(img_path)
    axs[i].imshow(img)
    axs[i].axis("off")
plt.show()