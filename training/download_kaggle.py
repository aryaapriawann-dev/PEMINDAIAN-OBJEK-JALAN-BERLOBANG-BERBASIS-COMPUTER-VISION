"""
==========================================================
Download Pothole Dataset from Kaggle
==========================================================
Downloads large pothole detection datasets from Kaggle
and converts them to YOLOv8 format.

Usage:
    python training/download_kaggle.py
==========================================================
"""

import os
import sys
import shutil
from pathlib import Path

try:
    import kagglehub
except ImportError:
    kagglehub = None

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"


def download_pothole_dataset():
    """Download pothole dataset using kagglehub."""
    global kagglehub
    if kagglehub is None:
        print("Installing kagglehub...")
        os.system(f"{sys.executable} -m pip install kagglehub --quiet")
        import kagglehub

    print("=" * 55)
    print("  POTHOLE DETECTION - DATASET DOWNLOADER")
    print("  Source: Kaggle")
    print("=" * 55)

    # Download the large pothole dataset from Kaggle
    # This dataset has images + YOLO annotations
    datasets_to_try = [
        ("chitholian/annotated-potholes-dataset", "Annotated Potholes Dataset"),
        ("sovitrath/road-pothole-images-for-pothole-detection", "Road Pothole Images"),
    ]

    downloaded_paths = []

    for dataset_id, name in datasets_to_try:
        print(f"\n  Downloading: {name}")
        print(f"  ID: {dataset_id}")
        try:
            path = kagglehub.dataset_download(dataset_id)
            print(f"  OK! Path: {path}")
            downloaded_paths.append((path, name))
        except Exception as e:
            print(f"  Error: {e}")

    return downloaded_paths


def organize_yolo_format(downloaded_paths: list):
    """
    Organize downloaded data into YOLOv8 format:
    datasets/
      train/images/  train/labels/
      valid/images/  valid/labels/
      test/images/   test/labels/
    """
    print("\n  Organizing into YOLOv8 format...")

    splits = ["train", "valid", "test"]
    for split in splits:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    all_images = []
    all_labels = []

    for dl_path, name in downloaded_paths:
        dl_path = Path(dl_path)
        print(f"\n  Processing: {name} ({dl_path})")

        # Find all image files recursively
        img_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]
        for ext in img_exts:
            for img_file in dl_path.rglob(ext):
                # Check if there's a matching label file
                label_file = None

                # Check same directory
                txt_file = img_file.with_suffix(".txt")
                if txt_file.exists():
                    label_file = txt_file

                # Check labels/ subdirectory parallel to images/
                if label_file is None and "images" in str(img_file):
                    alt_label = Path(str(img_file).replace("images", "labels")).with_suffix(".txt")
                    if alt_label.exists():
                        label_file = alt_label

                # Check labels/ at same level
                if label_file is None:
                    alt_label = img_file.parent.parent / "labels" / (img_file.stem + ".txt")
                    if alt_label.exists():
                        label_file = alt_label

                if label_file is not None:
                    all_images.append(img_file)
                    all_labels.append(label_file)

    print(f"\n  Found {len(all_images)} annotated images total")

    if len(all_images) == 0:
        # Check if there are images without labels (need to create labels)
        print("  No pre-annotated images found.")
        print("  Looking for unannotated images...")

        for dl_path, name in downloaded_paths:
            dl_path = Path(dl_path)
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]:
                for img_file in dl_path.rglob(ext):
                    all_images.append(img_file)

        print(f"  Found {len(all_images)} images (may need annotation)")

        if len(all_images) == 0:
            print("  ERROR: No images found in downloaded data!")
            return 0

    # Split: 80% train, 15% valid, 5% test
    import random
    random.seed(42)
    indices = list(range(len(all_images)))
    random.shuffle(indices)

    n = len(indices)
    train_end = int(n * 0.80)
    valid_end = int(n * 0.95)

    split_map = {
        "train": indices[:train_end],
        "valid": indices[train_end:valid_end],
        "test": indices[valid_end:],
    }

    total_copied = 0
    for split, idxs in split_map.items():
        count = 0
        for idx in idxs:
            img_src = all_images[idx]
            # Unique filename to avoid collisions
            new_name = f"pothole_{total_copied + count:06d}{img_src.suffix.lower()}"

            img_dst = DATASET_DIR / split / "images" / new_name
            shutil.copy2(img_src, img_dst)

            # Copy label if exists
            if idx < len(all_labels) and all_labels[idx] is not None:
                lbl_src = all_labels[idx]
                lbl_dst = DATASET_DIR / split / "labels" / (f"pothole_{total_copied + count:06d}.txt")

                # Read label and normalize class IDs to 0 (pothole)
                with open(lbl_src, 'r') as f:
                    lines = f.readlines()

                with open(lbl_dst, 'w') as f:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Set class to 0 (pothole), keep bbox coordinates
                            parts[0] = "0"
                            f.write(" ".join(parts) + "\n")

            count += 1
        total_copied += count
        print(f"  {split}: {count} images")

    return total_copied


def create_data_yaml():
    """Create data.yaml for YOLOv8 training."""
    yaml_content = f"""# Pothole Detection Dataset
# Auto-generated by download_kaggle.py

path: {DATASET_DIR.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  0: pothole
"""
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\n  data.yaml created: {yaml_path}")


def count_stats():
    """Print dataset statistics."""
    print("\n" + "=" * 55)
    print("  DATASET STATISTICS")
    print("=" * 55)

    total = 0
    for split in ["train", "valid", "test"]:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        img_count = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        lbl_count = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        total += img_count
        print(f"  {split:>8}: {img_count:>6} images, {lbl_count:>6} labels")

    print(f"  {'TOTAL':>8}: {total:>6} images")
    print()

    if total >= 10000:
        print("  TARGET REACHED! (>= 10,000 images)")
    else:
        print(f"  Target: 10,000+ | Current: {total}")
        print("  Tip: Data augmentation during training will help!")

    return total


def main():
    # Download
    downloaded = download_pothole_dataset()

    if not downloaded:
        print("\n  No datasets downloaded successfully.")
        print("  You may need to configure Kaggle credentials.")
        print("  See: https://www.kaggle.com/docs/api")
        sys.exit(1)

    # Organize into YOLO format
    total = organize_yolo_format(downloaded)

    if total > 0:
        # Create data.yaml
        create_data_yaml()

    # Print stats
    count_stats()

    print("\n  Next step: Train the model")
    print("    python training/train.py")
    print()


if __name__ == "__main__":
    main()
