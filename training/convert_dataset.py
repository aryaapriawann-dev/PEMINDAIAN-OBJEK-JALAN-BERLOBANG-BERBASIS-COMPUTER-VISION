"""
Convert Kaggle Pothole Dataset (XML/VOC) to YOLOv8 Format
and organize into train/valid/test splits.
"""
import json
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths
CACHE_DIR = Path(r"C:\Users\aryaz\.cache\kagglehub\datasets\chitholian\annotated-potholes-dataset\versions\1")
DATASET_DIR = Path(r"D:\PROJEK JALAN BERLUBANG\datasets")

def xml_to_yolo(xml_path, img_w, img_h):
    """Convert Pascal VOC XML to YOLO format labels."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    labels = []

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO format (normalized center x, y, width, height)
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        # Clamp values
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return labels


def main():
    print("=" * 55)
    print("  CONVERTING DATASET TO YOLOv8 FORMAT")
    print("=" * 55)

    img_dir = CACHE_DIR / "annotated-images"

    # Load splits if available
    splits_file = CACHE_DIR / "splits.json"
    if splits_file.exists():
        with open(splits_file, "r") as f:
            splits_data = json.load(f)
        print(f"  Found splits.json with keys: {list(splits_data.keys())}")
    else:
        splits_data = None

    # Find all image-XML pairs
    pairs = []
    for img_file in sorted(img_dir.glob("*.jpg")):
        xml_file = img_file.with_suffix(".xml")
        if xml_file.exists():
            pairs.append((img_file, xml_file))

    # Also check for JPEG/PNG
    for ext in ["*.jpeg", "*.png", "*.JPG"]:
        for img_file in sorted(img_dir.glob(ext)):
            xml_file = img_file.with_suffix(".xml")
            if xml_file.exists():
                pairs.append((img_file, xml_file))

    print(f"  Found {len(pairs)} annotated image pairs")

    if len(pairs) == 0:
        print("  ERROR: No annotated pairs found!")
        return

    # Create output directories
    for split in ["train", "valid", "test"]:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Split: 80% train, 15% valid, 5% test
    random.seed(42)
    indices = list(range(len(pairs)))
    random.shuffle(indices)

    n = len(indices)
    train_end = int(n * 0.80)
    valid_end = int(n * 0.95)

    split_assignments = {}
    for i in indices[:train_end]:
        split_assignments[i] = "train"
    for i in indices[train_end:valid_end]:
        split_assignments[i] = "valid"
    for i in indices[valid_end:]:
        split_assignments[i] = "test"

    # Process each pair
    counts = {"train": 0, "valid": 0, "test": 0}
    errors = 0

    for idx, (img_file, xml_file) in enumerate(pairs):
        split = split_assignments[idx]

        try:
            # Get image dimensions from XML
            tree = ET.parse(str(xml_file))
            root = tree.getroot()
            size_elem = root.find("size")
            img_w = float(size_elem.find("width").text)
            img_h = float(size_elem.find("height").text)

            if img_w <= 0 or img_h <= 0:
                # Fallback: read image dimensions
                import cv2
                img = cv2.imread(str(img_file))
                if img is not None:
                    img_h, img_w = img.shape[:2]
                else:
                    errors += 1
                    continue

            # Convert XML to YOLO labels
            labels = xml_to_yolo(xml_file, img_w, img_h)

            if len(labels) == 0:
                continue

            # Copy image
            new_name = f"pothole_{idx:05d}{img_file.suffix.lower()}"
            shutil.copy2(img_file, DATASET_DIR / split / "images" / new_name)

            # Write YOLO label
            label_name = f"pothole_{idx:05d}.txt"
            label_path = DATASET_DIR / split / "labels" / label_name
            with open(label_path, "w") as f:
                f.write("\n".join(labels) + "\n")

            counts[split] += 1

        except Exception as e:
            print(f"  Error processing {img_file.name}: {e}")
            errors += 1

    # Update data.yaml
    yaml_content = f"""# Pothole Detection Dataset
# Converted from Kaggle Annotated Potholes Dataset
# Total: {sum(counts.values())} images

path: {DATASET_DIR.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  0: pothole
"""
    (DATASET_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")

    # Print results
    print()
    print("  " + "=" * 45)
    print("  CONVERSION COMPLETE")
    print("  " + "=" * 45)
    print(f"  Train: {counts['train']} images")
    print(f"  Valid: {counts['valid']} images")
    print(f"  Test:  {counts['test']} images")
    print(f"  Total: {sum(counts.values())} images")
    print(f"  Errors: {errors}")
    print()
    print("  Dataset saved to:", DATASET_DIR)
    print()
    print("  Next: python training/train.py")
    print()


if __name__ == "__main__":
    main()
