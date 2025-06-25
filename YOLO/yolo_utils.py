import os
from ultralytics import YOLO
from PIL import Image

# Load the YOLO model
MODEL_PATH = "YOLO/models/epoch90.pt"
yolo_model = YOLO(MODEL_PATH)

def detect_parts(image_path):
    """
    Run YOLO segmentation on the given image and return the detection results 
    along with cropped parts as PIL Images.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: (YOLO results object, dict of cropped PIL Images for parts)
               The dict contains keys: 'sohle', 'schnuersenkel', 'farbe', 'ganzer_schuh'.
               If loading fails, returns (None, {}).
    """
    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠ Error loading {image_path}: {e}")
        return None, {}

    # Run YOLO inference
    results = yolo_model(image_path, conf= 0.6)

    # Initialize the dictionary for storing cropped parts
    part_images = {
        "sohle": None,            # Sole
        "schnuersenkel": None,    # Shoelaces
        "farbe": None,            # Color region
        "ganzer_schuh": None      # Whole shoe
    }

    # Process detections and crop parts
    for box in results[0].boxes:
        part_idx = int(box.cls)
        xyxy = [int(x) for x in box.xyxy[0]]
        crop = pil_image.crop(xyxy)

        # Assign the crop to the corresponding part
        if part_idx == 0:
            part_images["sohle"] = crop
        elif part_idx == 1:
            part_images["schnuersenkel"] = crop
        elif part_idx == 2:
            part_images["farbe"] = crop
        elif part_idx == 3:
            part_images["ganzer_schuh"] = crop

    return results, part_images
