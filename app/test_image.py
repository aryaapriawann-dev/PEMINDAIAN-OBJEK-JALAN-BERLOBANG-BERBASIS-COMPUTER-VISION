"""
==========================================================
🖼️ Image Tester - Pothole Detection
==========================================================
Uji coba model langsung pada file gambar/foto.
Bisa mengetes gambar dari dataset ataupun foto jalan kamu sendiri.

Usage:
  python app/test_image.py
  python app/test_image.py --image path/to/foto.jpg
==========================================================
"""

import cv2
import argparse
from pathlib import Path
import sys
import glob

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.detector import PotholeDetector
from app.alarm import AlarmSystem


def test_image(image_path: str = None, model_path: str = None, conf: float = 0.20):
    if model_path is None:
        model_path = str(PROJECT_ROOT / "models" / "best.pt")

    # If no image provided, pick one from test dataset
    if image_path is None:
        test_samples = list(Path(PROJECT_ROOT / "datasets" / "test" / "images").glob("*.jpg"))
        if not test_samples:
            test_samples = list(Path(PROJECT_ROOT / "datasets" / "valid" / "images").glob("*.jpg"))
        if test_samples:
            image_path = str(test_samples[0])
            print(f"ℹ️ Menggunakan contoh gambar dari dataset: {image_path}")
        else:
            print("❌ Tidak ada gambar yang ditemukan. Masukkan path gambar dengan argumen: --image nama_file.jpg")
            return

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Gagal membuka gambar: {image_path}")
        return

    print("\n" + "=" * 50)
    print("  🖼️ UJI COBA DETEKSI GAMBAR")
    print("=" * 50)
    print(f"  📦 Model  : {model_path}")
    print(f"  🖼️ Gambar : {image_path}")
    print(f"  🎯 Conf   : {conf:.2f}")
    print("=" * 50)

    detector = PotholeDetector(model_path=model_path, confidence=conf)
    alarm = AlarmSystem()

    detections = detector.detect(img)
    annotated = detector.draw_detections(img, detections, show_info=False)

    h, w = annotated.shape[:2]
    has_pothole = len(detections) > 0

    if has_pothole:
        cv2.rectangle(annotated, (0, 0), (w, 50), (0, 0, 200), -1)
        status_text = f"BAHAYA! Ditemukan {len(detections)} Lubang Jalan"
        print(f"\n🚨 {status_text}")
        for i, det in enumerate(detections, 1):
            print(f"   [{i}] Confidence: {det.confidence:.1%}, Severity: {det.severity}, Box: {det.bbox}")
        
        # Play alarm beep
        print("🔊 Membunyikan alarm 'TIT TIT TIT'...")
        alarm.start()
    else:
        cv2.rectangle(annotated, (0, 0), (w, 50), (30, 160, 40), -1)
        status_text = "JALAN AMAN - Tidak Ada Lubang Terdeteksi"
        print(f"\n🟢 {status_text}")

    cv2.putText(annotated, status_text, (20, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    output_path = PROJECT_ROOT / "hasil_deteksi.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"\n💾 Hasil visual disimpan ke: {output_path}")

    # Show window
    window_name = "Hasil Deteksi Lubang Jalan (Tekan sembarang tombol untuk keluar)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(w, 900), min(h, 700))
    cv2.imshow(window_name, annotated)
    cv2.waitKey(0)
    
    alarm.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model pada gambar")
    parser.add_argument("--image", type=str, default=None, help="Path ke gambar")
    parser.add_argument("--model", type=str, default=None, help="Path ke model .pt")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    args = parser.parse_args()

    test_image(args.image, args.model, args.conf)
