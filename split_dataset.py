import os, shutil, random

SOURCE_DIR = "dataset/dataset/PlantVillage"
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
SPLIT_RATIO = 0.8

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

for category in os.listdir(SOURCE_DIR):
    category_path = os.path.join(SOURCE_DIR, category)
    if not os.path.isdir(category_path):
        continue

    images = os.listdir(category_path)
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, category), exist_ok=True)

    for img in train_images:
        src_path = os.path.join(category_path, img)
        if os.path.isfile(src_path):  # ✅ only copy files
            shutil.copy(src_path, os.path.join(TRAIN_DIR, category, img))

    for img in val_images:
        src_path = os.path.join(category_path, img)
        if os.path.isfile(src_path):  # ✅ only copy files
            shutil.copy(src_path, os.path.join(VAL_DIR, category, img))
print("✅ Dataset split into 'train/' and 'val/' successfully!")

