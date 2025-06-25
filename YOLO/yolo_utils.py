import os
from ultralytics import YOLO
from PIL import Image

# Load YOLO model from absolute path (robust to execution context)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to this file (yolo_utils.py)
MODEL_PATH = os.path.join(BASE_DIR, "models", "epoch90.pt")
yolo_model = YOLO(MODEL_PATH)

def detect_parts(image_path):
    """
    Run YOLO segmentation on the given image and return detection results 
    along with cropped parts as PIL images.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: (YOLO results object, dict of cropped PIL images)
               The dict contains keys: 'sohle', 'schnuersenkel', 'farbe', 'ganzer_schuh'.
               If detection fails, returns (None, {}).
    """
    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠ Error loading image {image_path}: {e}")
        return None, {}

    # Run YOLO model on the image
    results = yolo_model(image_path, conf=0.6)

    # Prepare a dictionary to store cropped parts
    part_images = {
        "sohle": None,            # Sole
        "schnuersenkel": None,    # Shoelaces
        "farbe": None,            # Color region
        "ganzer_schuh": None      # Whole shoe
    }

    # Loop over detected boxes and crop the corresponding regions
    for box in results[0].boxes:
        part_idx = int(box.cls)                 # Class ID as int
        xyxy = [int(x) for x in box.xyxy[0]]    # Bounding box coordinates
        crop = pil_image.crop(xyxy)             # Crop the region

        # Assign cropped image to the appropriate key based on class ID
        if part_idx == 0:
            part_images["sohle"] = crop
        elif part_idx == 1:
            part_images["schnuersenkel"] = crop
        elif part_idx == 2:
            part_images["farbe"] = crop
        elif part_idx == 3:
            part_images["ganzer_schuh"] = crop

    return results, part_images
