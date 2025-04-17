from annoy import AnnoyIndex
import numpy as np
import os
import sys
from encode_images import get_embedding

# Configuration
embedding_dim = 512
index_path = "annoy_index/fashion.ann"
id_map_path = "annoy_index/id_map.txt"

# Ensure image path is provided as argument
if len(sys.argv) < 2:
    print("[!] Please provide an image path. Example:")
    print("    python search_similar.py data/your_image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

# Check if file exists
if not os.path.isfile(image_path):
    print(f"[!] Image not found: {image_path}")
    sys.exit(1)

# Load Annoy index
index = AnnoyIndex(embedding_dim, 'angular')
if not os.path.exists(index_path):
    print(f"[!] Annoy index not found at: {index_path}")
    sys.exit(1)
index.load(index_path)

# Generate embedding for input image
query_vector = get_embedding(image_path)

# Find top 5 similar items
ids, dists = index.get_nns_by_vector(query_vector, 5, include_distances=True)

# Load ID map
if not os.path.exists(id_map_path):
    print(f"[!] ID map not found at: {id_map_path}")
    sys.exit(1)

id_map = {}
with open(id_map_path, "r") as f:
    for line in f:
        idx, name = line.strip().split(",")
        id_map[int(idx)] = name

# Display results
print("\nSimilar images:")
for idx, dist in zip(ids, dists):
    print(f"- {id_map[idx]} (distance: {dist:.4f})")
