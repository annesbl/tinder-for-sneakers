import os

def filter_for_class_0(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        if not fname.endswith(".txt"):
            continue

        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        with open(src_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if parts[0] == "0":  # nur Klasse 0 (sohle)
                parts[0] = "0"  # bleibt 0
                new_lines.append(" ".join(parts))

        if new_lines:
            with open(dst_path, "w") as f:
                f.write("\n".join(new_lines))

# Train und Val bearbeiten
filter_for_class_0("/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-2/yolo/dataset/labels/train", "/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-2/yolo/dataset/labels_sohle/train")
filter_for_class_0("/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-2/yolo/dataset/labels/val", "/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-2/yolo/dataset/labels_sohle/val")
