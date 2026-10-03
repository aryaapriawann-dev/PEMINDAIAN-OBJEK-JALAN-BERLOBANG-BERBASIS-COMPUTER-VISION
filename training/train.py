"""
==========================================================
🧠 YOLOv8 Training Script - Pothole Detection
==========================================================
Trains a YOLOv8 model on the pothole detection dataset.

Usage:
    python training/train.py
    python training/train.py --epochs 200 --batch 32 --model yolov8m.pt
==========================================================
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"


def validate_dataset():
    """Check if dataset is properly set up."""
    data_yaml = DATASET_DIR / "data.yaml"
    train_dir = DATASET_DIR / "train" / "images"

    if not data_yaml.exists():
        print("❌ Dataset config not found: datasets/data.yaml")
        print("   Run: python training/download_dataset.py --api_key YOUR_KEY")
        return False

    if not train_dir.exists():
        print("❌ Training images not found: datasets/train/images/")
        print("   Run: python training/download_dataset.py --api_key YOUR_KEY")
        return False

    # Count images
    img_count = len(list(train_dir.glob("*")))
    if img_count == 0:
        print("❌ No training images found!")
        return False

    print(f"✅ Dataset found: {img_count} training images")
    return True


def train(args):
    """Run YOLOv8 training."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Validate dataset
    if not validate_dataset():
        sys.exit(1)

    print("\n🧠 YOLOv8 Training - Pothole Detection")
    print("=" * 50)
    print(f"  Model       : {args.model}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch Size  : {args.batch}")
    print(f"  Image Size  : {args.imgsz}")
    print(f"  Device      : {args.device}")
    print(f"  Dataset     : {DATASET_DIR / 'data.yaml'}")
    print("=" * 50)

    # Initialize model
    model = YOLO(args.model)

    # Training run name
    run_name = f"pothole_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Train
    model.train(
        data=str(DATASET_DIR / "data.yaml"),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(PROJECT_ROOT / "runs"),
        name=run_name,
        patience=30,           # Early stopping patience
        save=True,             # Save checkpoints
        save_period=10,        # Save every N epochs
        cache=args.cache,      # Cache images for faster training
        workers=args.workers,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
        deterministic=True,
        plots=True,            # Generate training plots
        # Data Augmentation
        hsv_h=0.015,           # Hue augmentation
        hsv_s=0.7,             # Saturation augmentation
        hsv_v=0.4,             # Value augmentation
        degrees=10.0,          # Rotation
        translate=0.1,         # Translation
        scale=0.5,             # Scale
        shear=2.0,             # Shear
        flipud=0.5,            # Vertical flip
        fliplr=0.5,            # Horizontal flip
        mosaic=1.0,            # Mosaic augmentation
        mixup=0.1,             # Mixup augmentation
    )

    print("\n✅ Training Complete!")
    print("=" * 50)

    # Copy best model to models directory
    MODELS_DIR.mkdir(exist_ok=True)
    best_model_src = PROJECT_ROOT / "runs" / run_name / "weights" / "best.pt"

    if best_model_src.exists():
        import shutil
        dest = MODELS_DIR / "best.pt"
        shutil.copy2(best_model_src, dest)
        print(f"  📦 Best model saved to: {dest}")

        # Validate the model
        print("\n📊 Validating model...")
        model = YOLO(str(dest))
        metrics = model.val(data=str(DATASET_DIR / "data.yaml"))

        print("\n📈 Validation Results:")
        print(f"  mAP50     : {metrics.box.map50:.4f}")
        print(f"  mAP50-95  : {metrics.box.map:.4f}")
        print(f"  Precision : {metrics.box.mp:.4f}")
        print(f"  Recall    : {metrics.box.mr:.4f}")
    else:
        print(f"  ⚠️ Best model not found at {best_model_src}")

    print("\n🚀 Next step: Run the app!")
    print("   python app/main.py")


def export_model(model_path: str, formats: list):
    """Export trained model to different formats."""
    from ultralytics import YOLO

    model = YOLO(model_path)

    for fmt in formats:
        print(f"\n📦 Exporting to {fmt}...")
        model.export(format=fmt)
        print(f"  ✅ Exported to {fmt}")


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for pothole detection"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Pre-trained model to start from (yolov8n/s/m/l/x.pt)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (default: 16, reduce if OOM)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size for training (default: 640)"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="Device to train on (0 for GPU, 'cpu' for CPU)"
    )
    parser.add_argument(
        "--cache", type=str, default="ram",
        help="Cache images in 'ram' or 'disk' for faster training"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of dataloader workers (default: 4)"
    )
    parser.add_argument(
        "--export", type=str, nargs="+", default=None,
        help="Export formats after training (e.g., onnx tflite)"
    )

    args = parser.parse_args()

    train(args)

    # Export if requested
    if args.export:
        model_path = str(MODELS_DIR / "best.pt")
        if Path(model_path).exists():
            export_model(model_path, args.export)


if __name__ == "__main__":
    main()
