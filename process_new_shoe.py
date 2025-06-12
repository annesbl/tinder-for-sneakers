"""
Detect labels on a new shoe image, extract colours + 512-D vector,
and store everything in SQLite database
"""

import argparse, sqlite3, uuid, json, cv2, numpy as np
from ultralytics import YOLO
import torch
from torchvision import models, transforms
from PIL import Image

WEIGHTS_FILE  = "models/yolov8m_sneakers.pt"      #aus training
CLASS_MAP     = {0: "Sohle", 1: "Schnuersenkel",
                 2: "Farbe", 3: "ganzer_schuh"}
TARGET_CLASSES = {0, 1, 2}                 #nur diese wollen wir einfärben

#farbe holen
def avg_rgb(img):  
    b, g, r = img.reshape(-1, 3).mean(axis=0)
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"

_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
_resnet = torch.nn.Sequential(*list(_resnet.children())[:-1])  # FC-Schicht weg
_resnet.eval()

_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225])
])

def backbone_feat(_, crop_bgr):
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    x   = _tf(img).unsqueeze(0)       # 1×3×224×224
    with torch.no_grad():
        v = _resnet(x).squeeze()      # 2048-D
    return v.numpy().tolist()

#def backbone_feat(model, crop):
 #   import torch
  #  crop = cv2.resize(crop, (640, 640))[:, :, ::-1]
   # crop = crop.transpose(2, 0, 1)[None] / 255.0
    #with torch.no_grad():
     #   x = torch.from_numpy(crop.astype(np.float32)).to(model.device)
      #  for layer in model.model.model[:-1]:
       #     x = layer(x)
        #v = x.mean([2, 3]).cpu().numpy().squeeze()  # 512-D
   # return v.tolist()

#yolo laden, bild einlesen, detektieren, Farben + Vektor extrahieren, ergebnis 2 dictionarys
def process(image_path, db_path="shoes.db"):
    model = YOLO(WEIGHTS_FILE)
    res = model.predict(image_path, imgsz=640, conf=0.25, device='cpu')[0]
    img   = cv2.imread(image_path)

    colours, vectors = {}, {}
    for xyxy, cls in zip(res.boxes.xyxy.cpu().numpy(),
                         res.boxes.cls.cpu().numpy().astype(int)):
        x1, y1, x2, y2 = map(int, xyxy)
        crop           = img[y1:y2, x1:x2]
        label          = CLASS_MAP[cls]

        if cls in TARGET_CLASSES:
            colours[label] = avg_rgb(crop)
        vectors[label] = backbone_feat(model, crop)

#alles so für die DB vorbereiten und speichern
    shoe_id = str(uuid.uuid4())
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS shoes(
                         id TEXT PRIMARY KEY,
                         name TEXT, description TEXT, img_path TEXT,
                         bg_light TEXT, bg_strong TEXT,
                         colours JSON, vectors JSON );""")
        cur.execute("""INSERT INTO shoes VALUES (?,?,?,?,?,?,?,?)""",
                    (shoe_id, "", "", image_path, "", "",
                     json.dumps(colours), json.dumps(vectors)))
    print(f"[✓] {image_path} ➜ {shoe_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Pfad zum neuen Schuhbild")
    p.add_argument("--db", default="shoes.db", help="SQLite-DB")
    args = p.parse_args()
    process(args.image, args.db)


# ---------------- HOW TO USE ------------------------------------------------------
    """
    1. Voraussetzungen
        • Python ≥3.9  
        • Pakete:  pip install ultralytics==0.6.6 opencv-python numpy

   ⚠️  Die Datei yolov8m_sneakers.pt (Gewichte aus train_yolov8m.py)
      muss im selben Ordner liegen.

    2. Aufruf – neues Schuhbild verarbeiten
        python process_new_shoe.py  <bildpfad>            # schreibt in shoes.db
        python process_new_shoe.py  <bildpfad> --db my.db # andere SQLite-Datei

    3. Was passiert?
        • YOLO erkennt Sohle, Schnürsenkel, Farb-Patch, ganzen Schuh  
        • Für Sohle/Schnürsenkel/Farbe wird die Durchschnittsfarbe berechnet (HEX)  
        • Für jede Box wird ein 512-D-Vektor aus dem Backbone gezogen  
        • Alles (UUID, Bildpfad, Farben, Vektoren) wird als neue Zeile
          in der Tabelle  shoes  der angegebenen SQLite-DB gespeichert

    4. Ausgabe
        [✓] <bildpfad> ➜ <uuid>

        Die UUID ist der Primärschlüssel der neuen DB-Zeile – damit können dann die
        Vektoren in Annoy indexiert werden.

    """