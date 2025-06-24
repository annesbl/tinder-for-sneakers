import os
import sqlite3
import json
import numpy as np
from annoy import AnnoyIndex

# Dynamically set the path to sneakers.db
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "..", "UI-DB", "sneakers.db")
DB_PATH = os.path.abspath(DB_PATH)

# Directory for index files
INDEX_DIR = os.path.join(BASE_DIR, "indices")
os.makedirs(INDEX_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# CLIP embedding dimension
EMBEDDING_DIM = 512

# create annoy index for each component
def create_annoy_index(teil_name, embedding_column):
    print(f"Erstelle Annoy-Index für {teil_name}...")
    
    # annoy index with cosine similarity (angular)
    annoy_index = AnnoyIndex(EMBEDDING_DIM, 'angular') 
    
    # annoy index mapping
    index_to_id = {}
    annoy_idx = 0
    
    # get all embeddings for selected component
    c.execute(f'SELECT id, {embedding_column} FROM sneakers')
    rows = c.fetchall()
    
    for sneaker_id, embedding_json in rows:
        embedding = json.loads(embedding_json)
        
        if embedding:  
            annoy_index.add_item(annoy_idx, embedding)
            index_to_id[annoy_idx] = sneaker_id
            annoy_idx += 1

    # create index with 10 trees
    annoy_index.build(10)
    
    # save annoy index
    index_filename = os.path.join(INDEX_DIR, f"sneakers_{teil_name}.ann")
    annoy_index.save(index_filename)
    
    # save mapping
    mapping_filename = os.path.join(INDEX_DIR, f"sneakers_{teil_name}_mapping.json")
    with open(mapping_filename, 'w') as f:
        json.dump(index_to_id, f, indent=2)
        
    print(f"✔ Saved: {index_filename}, {mapping_filename}")
    return annoy_idx

create_annoy_index("sohle", "embedding_sohle")
create_annoy_index("schnuersenkel", "embedding_schnuersenkel") 
create_annoy_index("farbe", "embedding_farbe")
create_annoy_index("ganzer_schuh", "embedding_ganzer_schuh")

conn.close()