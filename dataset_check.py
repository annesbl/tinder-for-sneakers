import os
import yaml

# ⚙ Konfiguration
DATASET_PATH = "dataset_full"
IMAGE_DIRS = ["images/train", "images/val", "images/test"]
LABEL_DIRS = ["labels/train", "labels/val", "labels/test"]

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def check_dataset():
    yaml_data = load_yaml(os.path.join(DATASET_PATH, "dataset.yaml"))
    classes = yaml_data.get("names", [])
    num_classes = yaml_data.get("nc", 0)

    print(f"➡ Dataset hat {num_classes} Klassen: {classes}")
    
    for img_dir, lbl_dir in zip(IMAGE_DIRS, LABEL_DIRS):
        img_path = os.path.join(DATASET_PATH, img_dir)
        lbl_path = os.path.join(DATASET_PATH, lbl_dir)

        if not os.path.exists(img_path):
            print(f"[!] Ordner fehlt: {img_path}")
            continue
        if not os.path.exists(lbl_path):
            print(f"[!] Ordner fehlt: {lbl_path}")
            continue

        img_files = set([os.path.splitext(f)[0] for f in os.listdir(img_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        lbl_files = set([os.path.splitext(f)[0] for f in os.listdir(lbl_path) if f.endswith('.txt')])

        # Bild ohne Label
        missing_labels = img_files - lbl_files
        # Label ohne Bild
        missing_images = lbl_files - img_files

        if missing_labels:
            print(f"[!] Bilder ohne Label ({img_dir}): {missing_labels}")
        if missing_images:
            print(f"[!] Labels ohne Bild ({lbl_dir}): {missing_images}")

        # Check Inhalte der Labels
        for lbl_file in lbl_files:
            lbl_fp = os.path.join(lbl_path, f"{lbl_file}.txt")
            with open(lbl_fp, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    if cls_id >= num_classes:
                        print(f"[!] Ungültige Klasse {cls_id} in {lbl_fp}")

    print("✅ Prüfung abgeschlossen.")

if __name__ == "__main__":
    check_dataset()
