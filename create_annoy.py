import sqlite3
import json
import numpy as np
from annoy import AnnoyIndex

# Verbindung zur Datenbank
conn = sqlite3.connect('sneakers.db')
c = conn.cursor()

# CLIP-Embedding Dimension (ViT-B/32 hat 512 Dimensionen)
EMBEDDING_DIM = 512

# Funktion zum Erstellen eines Annoy-Index für ein bestimmtes Teil
def create_annoy_index(teil_name, embedding_column):
    print(f"Erstelle Annoy-Index für {teil_name}...")
    
    # Annoy-Index erstellen
    annoy_index = AnnoyIndex(EMBEDDING_DIM, 'angular')  # 'angular' für cosine similarity
    
    # Mapping von Annoy-Index zu Sneaker-ID
    index_to_id = {}
    annoy_idx = 0
    
    # Alle Einträge mit nicht-leeren Embeddings abrufen
    c.execute(f'SELECT id, {embedding_column} FROM sneakers')
    rows = c.fetchall()
    
    for sneaker_id, embedding_json in rows:
        embedding = json.loads(embedding_json)
        
        # Nur hinzufügen, wenn Embedding nicht leer ist
        if embedding:  # Prüft ob Liste nicht leer ist
            annoy_index.add_item(annoy_idx, embedding)
            index_to_id[annoy_idx] = sneaker_id
            annoy_idx += 1
    
    if annoy_idx > 0:
        # Index mit 10 Bäumen erstellen (mehr Bäume = bessere Genauigkeit, langsamere Suche)
        annoy_index.build(10)
        
        # Index speichern
        index_filename = f"sneakers_{teil_name}.ann"
        annoy_index.save(index_filename)
        
        # Mapping speichern
        mapping_filename = f"sneakers_{teil_name}_mapping.json"
        with open(mapping_filename, 'w') as f:
            json.dump(index_to_id, f, indent=2)
        
        print(f"✓ {teil_name}: {annoy_idx} Embeddings in {index_filename} gespeichert")
        print(f"✓ Mapping in {mapping_filename} gespeichert")
    else:
        print(f"⚠ Keine Embeddings für {teil_name} gefunden")
    
    return annoy_idx

# Indizes für alle vier Teile erstellen
print("Starte Erstellung der Annoy-Indizes...\n")

sohle_count = create_annoy_index("sohle", "embedding_sohle")
print()

schnuersenkel_count = create_annoy_index("schnuersenkel", "embedding_schnuersenkel") 
print()

farbe_count = create_annoy_index("farbe", "embedding_farbe")
print()

ganzer_schuh_count = create_annoy_index("ganzer_schuh", "embedding_ganzer_schuh")
print()

print("=" * 50)
print("ZUSAMMENFASSUNG:")
print(f"Sohle: {sohle_count} Embeddings")
print(f"Schnürsenkel: {schnuersenkel_count} Embeddings")
print(f"Farbe: {farbe_count} Embeddings")
print(f"Ganzer Schuh: {ganzer_schuh_count} Embeddings")
print("\nErstellt Dateien:")
print("- sneakers_sohle.ann + sneakers_sohle_mapping.json")
print("- sneakers_schnuersenkel.ann + sneakers_schnuersenkel_mapping.json") 
print("- sneakers_farbe.ann + sneakers_farbe_mapping.json")
print("- sneakers_ganzer_schuh.ann + sneakers_ganzer_schuh_mapping.json")

conn.close()