import os
from ultralytics import YOLO

# Achtung: Basismodell muss YOLOv8m-SEGM sein
DATASET_YAML = "dataset_full/dataset.yaml"     
BASE_MODEL   = "yolov8m-seg.pt"  # <- das ist das Segmentation-Model!
OUT_WEIGHTS  = "yolov8m_sneakers.pt"

def main():
    model = YOLO(BASE_MODEL)

    model.train(
        data     = DATASET_YAML,
        epochs   = 150,
        imgsz    = 640,
        batch    = 16,
        patience = 20,          #Early Stopping
        device   = 0            #GPU 0
    )

    best = model.trainer.best   # Pfad zu best.pt
    os.rename(best, OUT_WEIGHTS)
    print(f"[✓] Training done – weights saved as {OUT_WEIGHTS}")

if __name__ == "__main__":
    main()
