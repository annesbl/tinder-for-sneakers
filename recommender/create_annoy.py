import sqlite3
import json
import numpy as np
from annoy import AnnoyIndex

conn = sqlite3.connect('sneakers.db')
c = conn.cursor()

# CLIP-Embedding dimension 
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
    index_filename = f"sneakers_{teil_name}.ann"
    annoy_index.save(index_filename)
    
    # save mapping
    mapping_filename = f"sneakers_{teil_name}_mapping.json"
    with open(mapping_filename, 'w') as f:
        json.dump(index_to_id, f, indent=2)
        
    
    return annoy_idx

create_annoy_index("sohle", "embedding_sohle")
create_annoy_index("schnuersenkel", "embedding_schnuersenkel") 
create_annoy_index("farbe", "embedding_farbe")
create_annoy_index("ganzer_schuh", "embedding_ganzer_schuh")

conn.close()