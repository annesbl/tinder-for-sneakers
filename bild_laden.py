from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-2/runs/obb/train4/weights/best.pt")
image_path = "/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-2/shoes1/dbe35a7d-sneakers_0166.jpg"
image = Image.open(image_path)
results = model.predict(image_path, conf=0.5)
results[0].plot()  # zeichnet die Boxen

results = model.predict(image_path, conf=0.5)

for r in results:
    boxes = r.boxes
    if boxes is not None:
        for b in boxes:
            if b.conf[0] > 0.5:
                print(f"{model.names[int(b.cls)]} → {b.conf[0]:.2f}")


plt.imshow(results[0].plot() if hasattr(results[0], "plot") else image)
plt.axis("off")
plt.show()
