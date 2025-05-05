import os
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from annoy import AnnoyIndex
import pickle

#Bounding Box Positionen für Teile (x1, y1, x2, y2) im Verhältnis zum Bild
PARTS = {
    "laces": (0.25, 0.25, 0.85, 0.45),  
    "sole": (0.05, 0.85, 0.95, 0.98),   
}

IMAGE_DIR = "shoes"
EMBEDDINGS_DIR = "part_embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

#Lade CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#Für jeden Teil einen Annoy-Index bauen
part_indexes = {}
part_id_map = {}

for part, box in PARTS.items():
    index = AnnoyIndex(512, "angular")
    id_map = {}
    idx = 0

    for fname in os.listdir(IMAGE_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(IMAGE_DIR, fname)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        x1, y1, x2, y2 = [int(w * box[i]) if i % 2 == 0 else int(h * box[i]) for i in range(4)]
        crop = image.crop((x1, y1, x2, y2))

        inputs = processor(images=crop, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
            embedding = embedding.cpu().numpy()[0]
        
        index.add_item(idx, embedding)
        id_map[idx] = fname
        idx += 1

    index.build(10)
    index.save(f"{EMBEDDINGS_DIR}/{part}.ann")
    with open(f"{EMBEDDINGS_DIR}/{part}_map.pkl", "wb") as f:
        pickle.dump(id_map, f)

    print(f"{part} index fertig mit {idx} Bildern.")

