import os
from ultralytics import YOLO
from PIL import Image

# Modell laden
MODEL_PATH = "YOLO/models/yolov8m-seg.pt"
yolo_model = YOLO(MODEL_PATH)

def detect_parts(image_path):
    """
    Führt YOLO-Segmentierung durch und liefert ein dict mit Crops (PIL Images).
    """
    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠ Fehler beim Laden von {image_path}: {e}")
        return None, {}

    results = yolo_model(image_path)

    teil_bilder = {"sohle": None, "schnuersenkel": None, "farbe": None, "ganzer_schuh": None}
    for box in results[0].boxes:
        teil_idx = int(box.cls)
        xyxy = [int(x) for x in box.xyxy[0]]
        crop = pil_image.crop(xyxy)

        if teil_idx == 0:
            teil_bilder["sohle"] = crop
        elif teil_idx == 1:
            teil_bilder["schnuersenkel"] = crop
        elif teil_idx == 2:
            teil_bilder["farbe"] = crop
        elif teil_idx == 3:
            teil_bilder["ganzer_schuh"] = crop

    return results, teil_bilder
