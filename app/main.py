"""
==========================================================
🚀 Main Entry Point - Pothole Detection System
==========================================================
Menjalankan sistem deteksi jalan berlubang secara real-time.
Mode default: Mode Ringan (Pure OpenCV) tanpa Tkinter GUI,
ringan untuk CPU laptop dan responsif.

Usage:
    python app/main.py
    python app/main.py --conf 0.25 --camera 0
==========================================================
"""

import argparse
import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Sistem Deteksi Jalan Berlubang - Real-time Computer Vision"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path ke model YOLOv8 (.pt file)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Device ID Kamera (default: 0)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Tingkat kepercayaan/sensitivitas deteksi (default: 0.25)"
    )
    parser.add_argument(
        "--persistence", type=int, default=15,
        help="Jumlah frame kotak sensor bertahan/menetap (default: 15 frame)"
    )
    parser.add_argument(
        "--warning-frames", type=int, default=2,
        help="Konfirmasi frame berturut-turut untuk BAHAYA (default: 2)"
    )
    parser.add_argument(
        "--normal-frames", type=int, default=10,
        help="Konfirmasi frame berturut-turut untuk NORMAL (default: 10)"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Jalankan GUI Tkinter dashboard (opsional, lebih berat)"
    )
    parser.add_argument(
        "--no-flip", "--dashcam", dest="flip", action="store_false", default=True,
        help="Mode dashcam jalan normal (tanpa flip)"
    )
    parser.add_argument(
        "--flip", "--mirror", dest="flip", action="store_true", default=True,
        help="Sesuai posisi tubuh seperti foto kedua (default: AKTIF)"
    )
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model
    if model_path is None:
        default_model = PROJECT_ROOT / "models" / "best.pt"
        if default_model.exists():
            model_path = str(default_model)
        else:
            print("=" * 55)
            print("  ⚠️ Model belum ditemukan di models/best.pt")
            print("=" * 55)
            model_path = str(default_model)

    print()
    print("  🚧 ═══════════════════════════════════════════")
    print("  🚧  SISTEM DETEKSI JALAN BERLUBANG")
    print("  🚧  Powered by YOLOv8 + Computer Vision")
    print("  🚧 ═══════════════════════════════════════════")
    print(f"  📦 Model       : {model_path}")
    print(f"  📷 Kamera ID   : {args.camera}")
    print(f"  🎯 Sensitivitas: {args.conf:.2f}")
    print(f"  🧠 Memori Box  : {args.persistence} frame (Anti-Kedap-Kedip)")
    print(f"  📷 Posisi Kamera: {'Sesuai Posisi Tubuh (Seperti Foto Kedua)' if args.flip else 'Mode Dashcam Jalan'}")
    print()

    if args.gui:
        # Hanya jalan jika user minta --gui
        from app.ui.dashboard import PotholeDetectionApp
        app = PotholeDetectionApp(
            model_path=model_path,
            camera_id=args.camera,
            persistence_frames=args.persistence,
            flip_camera=args.flip,
        )
        app.mainloop()
    else:
        # Default: Mode Ringan Pure OpenCV (Tanpa Tkinter GUI)
        from app.test_camera import run_camera
        run_camera(
            model_path=model_path,
            camera_id=args.camera,
            conf=args.conf,
            warning_frames=args.warning_frames,
            normal_frames=args.normal_frames,
            persistence_frames=args.persistence,
            is_mirrored=args.flip,
        )


if __name__ == "__main__":
    main()
