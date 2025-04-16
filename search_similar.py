from annoy import AnnoyIndex
import numpy as np
import os
import sys
from encode_images import get_embedding  # importiere get_embedding aus encode_images.py

embedding_dim = 512
index_path = "annoy_index/fashion.ann"
id_map_path = "annoy_index/id_map.txt"

# Sicherstellen, dass Bild angegeben wurde
if len(sys.argv) < 2:
    print("[!] Bitte gib ein Bild an. Beispiel:")
    print("    python search_similar.py data/deinbild.jpg")
    sys.exit(1)

image_path = sys.argv[1]

# Datei existiert?
if not os.path.isfile(image_path):
    print(f"[!] Bild nicht gefunden: {image_path}")
    sys.exit(1)

# Index laden
index = AnnoyIndex(embedding_dim, 'angular')
if not os.path.exists(index_path):
    print(f"[!] Annoy-Index nicht gefunden unter: {index_path}")
    sys.exit(1)
index.load(index_path)

# Embedding erzeugen
query_vector = get_embedding(image_path)

# Ähnliche Bilder finden
ids, dists = index.get_nns_by_vector(query_vector, 5, include_distances=True)

# ID-Map laden
if not os.path.exists(id_map_path):
    print(f"[!] ID-Map nicht gefunden unter: {id_map_path}")
    sys.exit(1)

id_map = {}
with open(id_map_path, "r") as f:
    for line in f:
        idx, name = line.strip().split(",")
        id_map[int(idx)] = name

# Ergebnisse anzeigen
print("\n🔍 Ähnliche Bilder:")
for idx, dist in zip(ids, dists):
    print(f"• {id_map[idx]} (Distanz: {dist:.4f})")
