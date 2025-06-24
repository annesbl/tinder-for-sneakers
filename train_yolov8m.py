import os
import torch
from ultralytics import YOLO

# Achtung: Basismodell muss YOLOv8m-SEGM sein
DATASET_YAML = "dataset_full/dataset.yaml"     
BASE_MODEL   = "yolov8m-seg.pt"  # <- das ist das Segmentation-Model!
OUT_WEIGHTS  = "yolov8m_sneakers.pt"

def main():
    model = YOLO(BASE_MODEL)

    model.train(
        data       = DATASET_YAML,
        epochs     = 150,
        imgsz      = 640,
        batch      = 16,
        patience   = 20,          #Early Stopping
        device     = "cuda:0" if torch.cuda.is_available() else "cpu",
        cache      = True,
        save_period= 10,
        project    = "runs_sneakers",
        name       = "yolov8m_seg_aug",

        # Augmentierungen
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.1,
        fliplr=0.5
    )

    best = model.trainer.best   # Pfad zu best.pt
    if best and os.path.exists(best):
        os.rename(best, OUT_WEIGHTS)
        print(f"[✓] Training abgeschlossen – Gewichte gespeichert als {OUT_WEIGHTS}")
    else:
        print("[!] Training abgeschlossen – best.pt wurde nicht gefunden!")

    # Val-Ergebnisse loggen
    metrics = model.val()
    print(f"[✓] Val mAP50: {metrics.seg.map50:.4f}, mAP50-95: {metrics.seg.map:.4f}")

if __name__ == "__main__":
    main()
