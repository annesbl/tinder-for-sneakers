import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Pfade
model = YOLO("runs/obb/train4/weights/best.pt")  # Achte auf OBB-Modell!
img_path = "shoes1/dbe35a7d-sneakers_0166.jpg"

# Laden & Vorhersage
results = model.predict(img_path, conf=0.1)

# Bild laden (BGR -> RGB)
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Zeichne OBBs
for r in results:
    if r.obb is not None:
        for box, obb in zip(r.obb, r.obb.xyxyxyxy):
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id]
            conf = box.conf.item()
            print(f"🟩 Klasse: {cls_name}, Confidence: {conf:.2f}")

            if conf > 0.5 and cls_name == "sohle":  # <- Hier filterst du deine Zielklasse
                points = obb.cpu().numpy().astype(int).reshape(-1, 1, 2)
                cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

# Anzeigen
plt.imshow(img)
plt.axis("off")
plt.title("Geneigte OBBs")
plt.show()
