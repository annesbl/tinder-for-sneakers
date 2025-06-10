""""
YOLO-v8m auf dataset_full trainieren und stärkste gewichte in
yolov8m_sneakers.pt speichern.
"""
import os
from ultralytics import YOLO

DATASET_YAML = "dataset_full/dataset.yaml"     
BASE_MODEL   = "yolov8m.pt"
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

    best = model.trainer.best   #Pfad zu best.pt
    os.rename(best, OUT_WEIGHTS)
    print(f"[✓] Training done – weights saved as {OUT_WEIGHTS}")

if __name__ == "__main__":
    main()


#für rechner 
"""
pip install ultralytics==0.6.6   opencv-python  numpy
python train_yolov8m.py
"""