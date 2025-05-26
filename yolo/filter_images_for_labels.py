import os
import shutil

def copy_images(label_dir, image_src_dir, image_dst_dir):
    os.makedirs(image_dst_dir, exist_ok=True)

    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        image_name = fname.replace(".txt", ".jpg")
        for ext in [".jpg", ".png"]:
            image_path = os.path.join(image_src_dir, image_name.replace(".jpg", ext))
            if os.path.exists(image_path):
                shutil.copy(image_path, os.path.join(image_dst_dir, os.path.basename(image_path)))
                break

# Für train und val
copy_images("/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-2/yolo/dataset/labels_sohle/train", "/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-2/yolo/dataset/images/train", "/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-2/yolo/dataset/images_sohle/train")
copy_images("/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-2/yolo/dataset/labels_sohle/val", "/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-2/yolo/dataset/images/val", "/Users/annesoballa/Documents/semester4/blangblang/tinder-for-sneakers-2/yolo/dataset/images_sohle/val")
