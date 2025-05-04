from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import os
from annoy import AnnoyIndex
import matplotlib.pyplot as plt

# Config
query_path = "query/test_shoe1.jpg"  # Set to None or "" to skip image
text_prompt = "white sneaker with thick sole"  # Set to "" to skip text
alpha = 0.5  # Weight: 1.0 = only image, 0.0 = only text, 0.5 = mix
top_k = 5
image_dir = "shoes"

# Load model and processor
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize variables
image_embedding = None
text_embedding = None

# Image embedding
if query_path and os.path.exists(query_path):
    image = Image.open(query_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
        image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)

# Text embedding
if text_prompt.strip():
    inputs = processor(text=[text_prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
        text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)

# Combine embeddings
if image_embedding is not None and text_embedding is not None:
    combined = alpha * image_embedding + (1 - alpha) * text_embedding
elif image_embedding is not None:
    combined = image_embedding
elif text_embedding is not None:
    combined = text_embedding
else:
    raise ValueError("Please provide either an image or a text prompt.")

# Prepare query vector
combined = combined.squeeze(0).cpu().numpy()

# Load index and filename mapping
index = AnnoyIndex(512, "angular")
index.load("shoe_index.ann")
id_to_filename = np.load("id_to_filename.npy", allow_pickle=True)

# Search
indices = index.get_nns_by_vector(combined, top_k)

# Display results
fig, axs = plt.subplots(1, top_k, figsize=(15, 5))
for i, idx in enumerate(indices):
    img_path = os.path.join(image_dir, id_to_filename[idx])
    img = Image.open(img_path)
    axs[i].imshow(img)
    axs[i].axis("off")
    axs[i].set_title(f"Result {i+1}")
query_type = "Image + Text" if image_embedding is not None and text_embedding is not None else "Text" if image_embedding is None else "Image"
plt.suptitle(f"FashionCLIP Search ({query_type})", fontsize=14)
plt.tight_layout()
plt.show()
