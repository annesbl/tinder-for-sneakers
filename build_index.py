from annoy import AnnoyIndex
import os
import numpy as np

# Config
embedding_dim = 512
embedding_dir = "embeddings"
index_dir = "annoy_index"
index_file = os.path.join(index_dir, "fashion.ann")

# Ensure directories exist
os.makedirs(embedding_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)

# Initialize Annoy index
index = AnnoyIndex(embedding_dim, 'angular')
image_filenames = []

# Add all embeddings to the index
for i, file in enumerate(os.listdir(embedding_dir)):
    if file.endswith(".npy"):
        vector = np.load(os.path.join(embedding_dir, file))
        index.add_item(i, vector)
        image_filenames.append(file.replace(".npy", ""))

# Build and save the index
index.build(10)
index.save(index_file)

# Save the ID-to-filename map
with open(os.path.join(index_dir, "id_map.txt"), "w") as f:
    for idx, name in enumerate(image_filenames):
        f.write(f"{idx},{name}\n")

print(f"[✓] Annoy index saved to: {index_file}")
