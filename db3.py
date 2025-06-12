#!/usr/bin/env python3
# --------------------------------------
# YOLO-Detektion + CLIP-Embedding + DB
# --------------------------------------
import os, sqlite3, struct, cv2, json, numpy as np, torch, open_clip
from PIL import Image
from ultralytics import YOLO

# ---- Konstanten -----------------------------------------------------------
BASE_DIR   = "data"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
DB_PATH    = os.path.join(BASE_DIR, "sneakers.db")
WEIGHTS    = "models/yolov8m_sneakers.pt"
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
CLASSES    = ["Farbe", "Schnürsenkel", "Sohle", "ganzer_schuh"]

# ---- Modelle laden --------------------------------------------------------
yolo = YOLO(WEIGHTS).to(DEVICE)
clip_model, _, clip_tf = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEVICE
)
clip_model.eval()

# ---- DB vorbereiten -------------------------------------------------------
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur  = conn.cursor()

cur.execute("PRAGMA table_info(sneakers)")
if "description" not in [row[1] for row in cur.fetchall()]:
    cur.execute("ALTER TABLE sneakers ADD COLUMN description TEXT")
    conn.commit()

cur.execute("""CREATE TABLE IF NOT EXISTS sneakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, image_path TEXT, color_light TEXT, color_dark TEXT )""")

cur.execute("""CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sneaker_id INTEGER, class_name TEXT,
    x_min REAL, y_min REAL, x_max REAL, y_max REAL,
    clip_vec BLOB,
    FOREIGN KEY(sneaker_id) REFERENCES sneakers(id) )""")

# °°° falls Spalten fehlen → ALTER TABLE
needed = {"x_min","y_min","x_max","y_max","clip_vec"}
have   = {c["name"] for c in cur.execute("PRAGMA table_info(labels)")}
for col in needed-have:
    cur.execute(f"ALTER TABLE labels ADD COLUMN {col} {'BLOB' if col=='clip_vec' else 'REAL'}")
conn.commit()

# ---- Haupt-Loop -----------------------------------------------------------
for file in os.listdir(LABELS_DIR):
    if not file.endswith(".txt"): continue
    img_id   = os.path.splitext(file)[0]
    img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
    if not os.path.exists(img_path):
        print(f"⚠️  Bild fehlt: {img_id}"); continue

    # Sneaker-Eintrag
    cur.execute("""INSERT INTO sneakers (name, image_path, color_light, color_dark, description)
                   VALUES (?,?,?,?,?)""",
                (f"Sneaker {img_id}", img_path, "#f5f7fa", "#1a1a1a", "—"))
    sk_id = cur.lastrowid

    # YOLO-Inferenz
    res = yolo.predict(img_path, imgsz=640, conf=0.25, device=DEVICE)[0]
    img = cv2.imread(img_path)

    for box, cls in zip(res.boxes.xyxy.cpu().numpy(),
                        res.boxes.cls.cpu().numpy().astype(int)):
        x1,y1,x2,y2 = box.astype(int)
        crop = img[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        clip_img = clip_tf(Image.fromarray(crop_rgb)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            vec = clip_model.encode_image(clip_img).cpu().numpy()[0]
        vec /= np.linalg.norm(vec)

        # Vektor als BLOB serialisieren (float32 → bytes)
        blob = vec.astype(np.float32).tobytes()
        cur.execute("""INSERT INTO labels
            (sneaker_id,class_name,x_min,y_min,x_max,y_max,clip_vec)
            VALUES (?,?,?,?,?,?,?)""",
            (sk_id, CLASSES[cls], x1,y1,x2,y2, sqlite3.Binary(blob)))

    conn.commit()
    print(f"✅ Importiert: {img_id}")

conn.close()
print("fertig")



"""
DESCRIPTION ÄNDERN:
cur.execute("UPDATE sneakers SET description=? WHERE id=?",
            ("Mein Mustertext", sneaker_id))
conn.commit()
"""