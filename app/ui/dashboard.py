"""
==========================================================
🖥️ Main Dashboard - Pothole Detection System
==========================================================
Assembles all UI components into the main application
window. Handles webcam capture, detection loop, and
alarm coordination.
==========================================================
"""

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from datetime import datetime
import threading
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.detector import PotholeDetector
from app.alarm import AlarmSystem
from app.ui.components import StatusPanel, ControlPanel, DetectionLog


class PotholeDetectionApp(ctk.CTk):
    """
    Main desktop application for real-time pothole detection.
    Features webcam feed, YOLOv8 detection, status display,
    and audio alarm system.
    """

    APP_TITLE = "🚧 Sistem Deteksi Jalan Berlubang"
    APP_SIZE = (1280, 780)
    VIDEO_UPDATE_MS = 30    # ~33 FPS target
    BLINK_INTERVAL_MS = 500  # Status blink interval

    def __init__(
        self,
        model_path: str = None,
        camera_id: int = 0,
        persistence_frames: int = 15,
        flip_camera: bool = True,
    ):
        super().__init__()

        # ---- Window Setup ----
        self.title(self.APP_TITLE)
        self.geometry(f"{self.APP_SIZE[0]}x{self.APP_SIZE[1]}")
        self.minsize(1024, 600)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # ---- State ----
        self._camera_id = camera_id
        self._flip_camera = flip_camera
        self._cap = None
        self._is_running = False
        self._alarm_enabled = True
        self._detection_active = False
        self._last_detection_time = 0
        self._detection_cooldown = 2.0  # Log cooldown in seconds
        self._frame_count = 0

        # ---- Initialize Components ----
        self._init_detector(model_path, persistence_frames)
        self._init_alarm()
        self._build_ui()
        self._start_camera()

    def _init_detector(self, model_path: str = None, persistence_frames: int = 15):
        """Initialize the YOLOv8 detector."""
        self.detector = PotholeDetector(
            model_path=model_path,
            confidence=0.25,
            persistence_frames=persistence_frames,
        )
        model_loaded = self.detector.load_model()

        if not model_loaded:
            self._model_status = "NOT_LOADED"
        else:
            self._model_status = "LOADED"

    def _init_alarm(self):
        """Initialize the alarm system."""
        self.alarm = AlarmSystem()

    def _build_ui(self):
        """Build the complete UI layout."""
        # Configure grid
        self.grid_columnconfigure(0, weight=3)  # Video area
        self.grid_columnconfigure(1, weight=1)  # Side panel
        self.grid_rowconfigure(0, weight=0)     # Header
        self.grid_rowconfigure(1, weight=1)     # Main content
        self.grid_rowconfigure(2, weight=0)     # Log

        # ========== HEADER ==========
        self._build_header()

        # ========== MAIN CONTENT ==========
        # Left: Video Feed
        self._build_video_panel()

        # Right: Status + Controls
        self._build_side_panel()

        # ========== BOTTOM: Detection Log ==========
        self._build_log_panel()

        # Keybindings
        self.bind("<f>", lambda e: self._toggle_flip())
        self.bind("<F>", lambda e: self._toggle_flip())

        # ========== Model Warning ==========
        if self._model_status == "NOT_LOADED":
            self._show_model_warning()

    def _toggle_flip(self):
        """Toggle horizontal camera flip (mirror mode)."""
        self._flip_camera = not self._flip_camera
        state = "ON (Mode Cermin)" if self._flip_camera else "OFF (Mode Normal)"
        self.log_panel.add_entry(f"Mirror Kamera: {state}", "info")

    def _build_header(self):
        """Build the header bar."""
        header = ctk.CTkFrame(self, fg_color="#0d1b2a", height=60, corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)

        # App title
        title = ctk.CTkLabel(
            header,
            text="🚧  SISTEM DETEKSI JALAN BERLUBANG",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#e6f1ff",
        )
        title.pack(side="left", padx=20, pady=15)

        # Timestamp
        self.time_label = ctk.CTkLabel(
            header,
            text="",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color="#8892b0",
        )
        self.time_label.pack(side="right", padx=20)
        self._update_time()

        # YOLOv8 badge
        badge = ctk.CTkLabel(
            header,
            text="YOLOv8",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#0d1b2a",
            fg_color="#64ffda",
            corner_radius=5,
            padx=8,
            pady=2,
        )
        badge.pack(side="right", padx=5)

        # CV badge
        cv_badge = ctk.CTkLabel(
            header,
            text="Computer Vision",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#0d1b2a",
            fg_color="#ffd700",
            corner_radius=5,
            padx=8,
            pady=2,
        )
        cv_badge.pack(side="right", padx=5)

    def _build_video_panel(self):
        """Build the video feed panel."""
        self.video_frame = ctk.CTkFrame(
            self,
            fg_color="#0a0a1a",
            corner_radius=15,
        )
        self.video_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)

        # Video canvas label
        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="📷 Menghubungkan ke Webcam...",
            font=ctk.CTkFont(size=16),
            text_color="#8892b0",
        )
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_side_panel(self):
        """Build the right side panel with status and controls."""
        side_panel = ctk.CTkFrame(self, fg_color="transparent")
        side_panel.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        side_panel.grid_rowconfigure(0, weight=1)
        side_panel.grid_rowconfigure(1, weight=1)

        # Status Panel
        self.status_panel = StatusPanel(side_panel)
        self.status_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        # Control Panel
        self.control_panel = ControlPanel(
            side_panel,
            on_confidence_change=self._on_confidence_change,
            on_alarm_toggle=self._on_alarm_toggle,
        )
        self.control_panel.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        self.control_panel.set_volume_callback(self._on_volume_change)

        if self._model_status == "LOADED":
            self.control_panel.set_model_info("YOLOv8 ✅ Aktif")
        else:
            self.control_panel.set_model_info("❌ Belum dimuat")

    def _build_log_panel(self):
        """Build the bottom detection log."""
        self.log_panel = DetectionLog(self, max_entries=100)
        self.log_panel.grid(
            row=2, column=0, columnspan=2,
            sticky="ew", padx=10, pady=(0, 10),
        )
        self.log_panel.configure(height=140)
        self.log_panel.pack_propagate(False)

        self.log_panel.add_entry("Sistem dimulai", "info")

    def _show_model_warning(self):
        """Show warning overlay when model is not loaded."""
        warning_frame = ctk.CTkFrame(
            self.video_frame,
            fg_color="#1a1a2e",
            corner_radius=12,
        )
        warning_frame.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            warning_frame,
            text="⚠️ Model Belum Dimuat",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#ffd700",
        ).pack(padx=30, pady=(20, 5))

        ctk.CTkLabel(
            warning_frame,
            text="Kamera aktif tapi deteksi tidak berjalan.\n"
                 "Untuk mengaktifkan deteksi, latih model terlebih dahulu:\n\n"
                 "1. python training/download_dataset.py --api_key KEY\n"
                 "2. python training/train.py\n"
                 "3. Restart aplikasi ini",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color="#8892b0",
            justify="left",
        ).pack(padx=30, pady=(5, 20))

        self.log_panel.add_entry(
            "Model belum dimuat! Train model dulu.", "warning"
        )

    # ========================================
    # Camera & Detection Loop
    # ========================================

    def _start_camera(self):
        """Initialize and start webcam capture."""
        try:
            self._cap = cv2.VideoCapture(self._camera_id, cv2.CAP_DSHOW)

            if not self._cap.isOpened():
                # Fallback without DirectShow
                self._cap = cv2.VideoCapture(self._camera_id)

            if self._cap.isOpened():
                # Set camera properties for better performance
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self._cap.set(cv2.CAP_PROP_FPS, 30)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                self._is_running = True
                self.log_panel.add_entry("Webcam terhubung!", "info")
                self._update_frame()
                self._start_blink_animation()
            else:
                self.video_label.configure(
                    text="❌ Webcam tidak terdeteksi!\n\n"
                         "Pastikan webcam terhubung dan tidak\n"
                         "digunakan aplikasi lain.",
                    text_color="#ff4444",
                )
                self.log_panel.add_entry("Webcam tidak terdeteksi!", "danger")

        except Exception as e:
            self.log_panel.add_entry(f"Error kamera: {e}", "danger")

    def _update_frame(self):
        """Main frame update loop - captures, detects, and displays."""
        if not self._is_running:
            return

        ret, frame = self._cap.read()
        if not ret:
            self.after(self.VIDEO_UPDATE_MS, self._update_frame)
            return

        # Mirror mode (flip horizontal)
        if self._flip_camera:
            frame = cv2.flip(frame, 1)

        self._frame_count += 1

        # Run detection and tracking (boxes persist stably across momentary frame drops)
        detections = []
        if self.detector.is_loaded:
            detections = self.detector.detect_and_track(frame)
            frame = self.detector.draw_detections(frame, detections)

        # Update UI based on detections
        self._handle_detections(detections)

        # Convert frame for Tkinter display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Resize to fit the video label
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()
        if label_w > 1 and label_h > 1:
            img = self._resize_keep_aspect(img, label_w, label_h)

        photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=photo, text="")
        self.video_label._image = photo  # Prevent garbage collection

        # Update FPS
        if self.detector.is_loaded:
            self.status_panel.update_fps(self.detector.fps)

        # Schedule next frame
        self.after(self.VIDEO_UPDATE_MS, self._update_frame)

    def _resize_keep_aspect(self, img: Image.Image, max_w: int, max_h: int) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        orig_w, orig_h = img.size
        ratio = min(max_w / orig_w, max_h / orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        return img.resize((new_w, new_h), Image.LANCZOS)

    def _handle_detections(self, detections: list):
        """Handle detection results - update status & alarm."""
        import time

        if len(detections) > 0:
            max_conf = max(d.confidence for d in detections)

            # Update status panel
            self.status_panel.set_danger(len(detections), max_conf)

            # Start alarm
            if self._alarm_enabled and not self.alarm.is_playing:
                self.alarm.start()

            # Log detection (with cooldown to prevent spam)
            now = time.time()
            if now - self._last_detection_time > self._detection_cooldown:
                severities = [d.severity for d in detections]
                self.log_panel.add_entry(
                    f"LUBANG TERDETEKSI! "
                    f"Jumlah: {len(detections)} | "
                    f"Confidence: {max_conf:.1%} | "
                    f"Severity: {', '.join(severities)}",
                    "danger"
                )
                self._last_detection_time = now

            self._detection_active = True

        else:
            if self._detection_active:
                # Transition from danger to safe
                self.log_panel.add_entry("Jalan kembali aman.", "info")
                self._detection_active = False

            self.status_panel.set_safe()

            # Stop alarm
            if self.alarm.is_playing:
                self.alarm.stop()

    def _start_blink_animation(self):
        """Start the status panel blink animation."""
        self.status_panel.blink()
        if self._is_running:
            self.after(self.BLINK_INTERVAL_MS, self._start_blink_animation)

    def _update_time(self):
        """Update the clock display."""
        now = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
        self.time_label.configure(text=now)
        self.after(1000, self._update_time)

    # ========================================
    # Callbacks
    # ========================================

    def _on_confidence_change(self, value: float):
        """Handle confidence threshold change."""
        self.detector.set_confidence(value)
        self.log_panel.add_entry(
            f"Confidence threshold diubah: {value:.0%}", "info"
        )

    def _on_alarm_toggle(self, enabled: bool):
        """Handle alarm toggle."""
        self._alarm_enabled = bool(enabled)
        if not self._alarm_enabled:
            self.alarm.stop()
        status = "aktif" if self._alarm_enabled else "nonaktif"
        self.log_panel.add_entry(f"Alarm suara {status}", "info")

    def _on_volume_change(self, volume: float):
        """Handle volume change."""
        self.alarm.set_volume(volume)

    def _on_close(self):
        """Clean up resources and close the app."""
        self._is_running = False

        # Release camera
        if self._cap and self._cap.isOpened():
            self._cap.release()

        # Stop alarm
        self.alarm.cleanup()

        # Close window
        self.destroy()
