"""
==========================================================
🛡️ ROADGUARD AI - Pothole Detection System (Pure OpenCV)
==========================================================
Real-time road surface hazard & pothole tracking powered by YOLOv8.
High-tech HUD surveillance overlay matching professional vision systems.

Features:
- Native 16:9 Widescreen display (Anti-Gepeng / Aspect Ratio Preserved)
- Semi-transparent glass HUD cards for status and guides
- High-contrast visual alerts and synchronized audio alarm
- Smooth FPS and temporal stabilization
==========================================================
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.detector import PotholeDetector, TemporalStabilizer
from app.alarm import AlarmSystem


def normalize_widescreen(frame: np.ndarray, target_w: int = 1280, target_h: int = 720) -> np.ndarray:
    """
    Guarantees standard 16:9 widescreen aspect ratio without distortion.
    If input camera is 4:3 (e.g. 640x480), center-crops to 16:9 and scales smoothly,
    ensuring the video feed is never squashed or stretched (Anti-Gepeng).
    """
    h, w = frame.shape[:2]
    target_aspect = target_w / target_h
    curr_aspect = w / h

    if abs(curr_aspect - target_aspect) > 0.03:
        if curr_aspect > target_aspect:
            # Wider than 16:9 -> crop left/right evenly
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            cropped = frame[:, start_x:start_x + new_w]
        else:
            # Taller than 16:9 (e.g. 4:3) -> crop top/bottom evenly
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            cropped = frame[start_y:start_y + new_h, :]
    else:
        cropped = frame

    if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
        return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return cropped


def render_hud_dashboard(
    canvas: np.ndarray,
    valid_detections: list,
    is_danger: bool,
    conf_threshold: float,
    fps: float,
    alarm_enabled: bool,
    persistence_frames: int = 15,
    is_mirrored: bool = False,
) -> np.ndarray:
    """
    Renders professional Heads-Up Display (HUD) overlay directly onto the video.
    Layout matches modern high-tech vision tracking systems:
    - Top Bar: ROADGUARD AI title & real-time FPS
    - Outer Framing: Bounding surveillance box
    - Left HUD Card: Status Jalan, Mode, Object count, Action Badge
    - Right HUD Card: Panduan & Controls shortcut badges
    - Center Alert: Bold warning message when danger is detected
    - Bounding Box: Translucent fill + sharp red border + label + corner brackets
    - Bottom Bar: Status summary & hotkey reference
    """
    h, w = canvas.shape[:2]

    # Create overlay layer for translucent alpha blending
    overlay = canvas.copy()

    # Color Palette (BGR)
    YELLOW_GOLD = (0, 215, 255)
    RED_DANGER = (25, 35, 240)
    RED_FILL = (15, 20, 200)
    GREEN_SAFE = (50, 215, 75)
    GREEN_FILL = (15, 75, 25)
    DARK_CARD = (12, 16, 22)
    TEXT_WHITE = (245, 245, 250)
    TEXT_MUTED = (175, 185, 200)

    # ----------------------------------------------------
    # 1. TRANSLUCENT BOUNDING BOX FILLS (Like Reference)
    # ----------------------------------------------------
    for det in valid_detections:
        x1, y1, x2, y2 = det.bbox
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        is_held = getattr(det, 'is_held', False)

        # Translucent fill over detected pothole (slightly darker amber if held via memory)
        box_fill = (15, 20, 160) if is_held else RED_FILL
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_fill, -1)

    # Blend translucent fills into canvas (alpha = 0.28)
    cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)
    overlay = canvas.copy()

    # ----------------------------------------------------
    # 2. BOUNDING BOX BORDERS & LABELS
    # ----------------------------------------------------
    for det in valid_detections:
        x1, y1, x2, y2 = det.bbox
        conf = det.confidence
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        is_held = getattr(det, 'is_held', False)

        # Sharp border: Red if locked/active, Orange-amber if held via memory
        box_border = (0, 140, 255) if is_held else RED_DANGER
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_border, 2)

        # High-tech corner bracket accents on each bounding box
        corner_len = min(18, max(6, (x2 - x1) // 4), max(6, (y2 - y1) // 4))
        corner_col = YELLOW_GOLD if not is_held else (0, 190, 255)
        cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), corner_col, 2)
        cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), corner_col, 2)
        cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), corner_col, 2)
        cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), corner_col, 2)
        cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), corner_col, 2)
        cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), corner_col, 2)
        cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), corner_col, 2)
        cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), corner_col, 2)

        # High-tech tag label with solid badge background
        tag_status = "[TERKUNCI]" if not is_held else "[MEMORI]"
        label_text = f"! POTHOLE {conf:.0%} {tag_status}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.44
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

        tag_y = y1 - 6
        if tag_y < 20:
            tag_y = y1 + th + 6
            cv2.rectangle(canvas, (x1, y1), (x1 + tw + 10, y1 + th + 10), (15, 20, 28), -1)
            cv2.rectangle(canvas, (x1, y1), (x1 + tw + 10, y1 + th + 10), box_border, 1)
            cv2.putText(canvas, label_text, (x1 + 5, tag_y), font, font_scale, corner_col, thickness, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (x1, tag_y - th - 3), (x1 + tw + 10, tag_y + 4), (15, 20, 28), -1)
            cv2.rectangle(canvas, (x1, tag_y - th - 3), (x1 + tw + 10, tag_y + 4), box_border, 1)
            cv2.putText(canvas, label_text, (x1 + 5, tag_y), font, font_scale, corner_col, thickness, cv2.LINE_AA)

    # ----------------------------------------------------
    # 3. OUTER SURVEILLANCE FRAMING BORDER (Like Reference)
    # ----------------------------------------------------
    frame_color = RED_DANGER if is_danger else (0, 180, 220)
    frame_x1, frame_y1 = 18, 54
    frame_x2, frame_y2 = w - 18, h - 36
    cv2.rectangle(canvas, (frame_x1, frame_y1), (frame_x2, frame_y2), frame_color, 2)

    # Corner brackets for high-tech HUD look
    corner_len = 16
    bracket_color = RED_DANGER if is_danger else YELLOW_GOLD
    # Top-left
    cv2.line(canvas, (frame_x1 - 2, frame_y1 - 2), (frame_x1 + corner_len, frame_y1 - 2), bracket_color, 3)
    cv2.line(canvas, (frame_x1 - 2, frame_y1 - 2), (frame_x1 - 2, frame_y1 + corner_len), bracket_color, 3)
    # Top-right
    cv2.line(canvas, (frame_x2 + 2, frame_y1 - 2), (frame_x2 - corner_len, frame_y1 - 2), bracket_color, 3)
    cv2.line(canvas, (frame_x2 + 2, frame_y1 - 2), (frame_x2 + 2, frame_y1 + corner_len), bracket_color, 3)
    # Bottom-left
    cv2.line(canvas, (frame_x1 - 2, frame_y2 + 2), (frame_x1 + corner_len, frame_y2 + 2), bracket_color, 3)
    cv2.line(canvas, (frame_x1 - 2, frame_y2 + 2), (frame_x1 - 2, frame_y2 - corner_len), bracket_color, 3)
    # Bottom-right
    cv2.line(canvas, (frame_x2 + 2, frame_y2 + 2), (frame_x2 - corner_len, frame_y2 + 2), bracket_color, 3)
    cv2.line(canvas, (frame_x2 + 2, frame_y2 + 2), (frame_x2 + 2, frame_y2 - corner_len), bracket_color, 3)

    # ----------------------------------------------------
    # 4. TRANSLUCENT HUD CARDS & BANNERS
    # ----------------------------------------------------
    header_h = 50
    cv2.rectangle(overlay, (0, 0), (w, header_h), DARK_CARD, -1)

    bottom_h = 32
    cv2.rectangle(overlay, (0, h - bottom_h), (w, h), DARK_CARD, -1)

    # Left HUD Card (Status Gerbang / Status Sistem)
    card_l_x, card_l_y = 30, 68
    card_l_w, card_l_h = 270, 235
    cv2.rectangle(overlay, (card_l_x, card_l_y), (card_l_x + card_l_w, card_l_y + card_l_h), DARK_CARD, -1)

    # Right HUD Card (Panduan)
    card_r_w, card_r_h = 250, 235
    card_r_x = w - card_r_w - 30
    card_r_y = 68
    cv2.rectangle(overlay, (card_r_x, card_r_y), (card_r_x + card_r_w, card_r_y + card_r_h), DARK_CARD, -1)

    # Action Badge inside Left HUD Card
    badge_x = card_l_x + 14
    badge_y = card_l_y + 90
    badge_w = card_l_w - 28
    badge_h = 44
    badge_fill = RED_FILL if is_danger else GREEN_FILL
    cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), badge_fill, -1)

    # Blend dark cards with video feed (alpha = 0.70)
    cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, canvas)

    # ----------------------------------------------------
    # 5. DRAW CRISP BORDERS & TEXT ON HUD CARDS
    # ----------------------------------------------------
    # Top bar accent line
    cv2.line(canvas, (0, 0), (w, 0), YELLOW_GOLD, 3)
    cv2.line(canvas, (0, header_h), (w, header_h), (0, 190, 230), 1)

    # Header Title: ROADGUARD AI
    cv2.putText(
        canvas,
        "ROADGUARD AI",
        (30, 34),
        cv2.FONT_HERSHEY_DUPLEX,
        0.82,
        YELLOW_GOLD,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Road Surface Hazard Tracking v3.0",
        (255, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (180, 195, 210),
        1,
        cv2.LINE_AA,
    )

    # Top Right: FPS
    cv2.putText(
        canvas,
        f"FPS {fps:.1f}",
        (w - 150, 34),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        YELLOW_GOLD,
        2,
        cv2.LINE_AA,
    )

    # --- LEFT HUD CARD (STATUS JALAN) ---
    cv2.rectangle(canvas, (card_l_x, card_l_y), (card_l_x + card_l_w, card_l_y + card_l_h), YELLOW_GOLD, 2)

    # Card Title
    cv2.putText(
        canvas,
        "STATUS JALAN",
        (card_l_x + 14, card_l_y + 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.48,
        YELLOW_GOLD,
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (card_l_x + 10, card_l_y + 32), (card_l_x + card_l_w - 10, card_l_y + 32), (0, 180, 220), 1)

    # Mode indicator
    cv2.rectangle(canvas, (card_l_x + 14, card_l_y + 44), (card_l_x + 22, card_l_y + 57), YELLOW_GOLD, -1)
    cv2.putText(
        canvas,
        "MODE: ROAD PATROL",
        (card_l_x + 30, card_l_y + 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        YELLOW_GOLD,
        1,
        cv2.LINE_AA,
    )

    # Object counter
    danger_count = len(valid_detections) if is_danger else 0
    cv2.putText(
        canvas,
        f"Lubang: {danger_count} (Bahaya: {danger_count} | Aman: {0 if is_danger else 1})",
        (card_l_x + 14, card_l_y + 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        TEXT_WHITE,
        1,
        cv2.LINE_AA,
    )

    # Prominent Action Badge (like AKSES DITOLAK in Reference)
    badge_border = RED_DANGER if is_danger else GREEN_SAFE
    cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), badge_border, 2)

    badge_text = "BAHAYA LUBANG" if is_danger else "JALAN AMAN"
    badge_color = (255, 255, 255)
    (btw, bth), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.62, 2)
    bt_x = badge_x + (badge_w - btw) // 2
    bt_y = badge_y + (badge_h + bth) // 2
    cv2.putText(canvas, badge_text, (bt_x, bt_y), cv2.FONT_HERSHEY_DUPLEX, 0.62, badge_color, 2, cv2.LINE_AA)

    # Subtext below badge
    if is_danger:
        subtext = f"Ket: Terdeteksi {danger_count} lubang berisiko"
        sub_color = (130, 150, 255)
    else:
        subtext = "Ket: Permukaan jalan mulus & aman"
        sub_color = (130, 220, 150)

    cv2.putText(canvas, subtext, (card_l_x + 14, card_l_y + 162), cv2.FONT_HERSHEY_SIMPLEX, 0.36, sub_color, 1, cv2.LINE_AA)

    # --- RIGHT HUD CARD (PANDUAN / CONTROLS) ---
    cv2.rectangle(canvas, (card_r_x, card_r_y), (card_r_x + card_r_w, card_r_y + card_r_h), YELLOW_GOLD, 2)

    cv2.putText(
        canvas,
        "PANDUAN",
        (card_r_x + 14, card_r_y + 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.48,
        YELLOW_GOLD,
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (card_r_x + 10, card_r_y + 32), (card_r_x + card_r_w - 10, card_r_y + 32), (0, 180, 220), 1)

    cam_label = "Sesuai Tubuh" if is_mirrored else "Dashcam"
    shortcuts = [
        ("[+]/[-]", f"Sensitivitas ({conf_threshold:.2f})"),
        ("[T]", f"Memori Sensor ({persistence_frames}f)"),
        ("[F]", f"Kamera ({cam_label})"),
        ("[W]", "Fullscreen (Layar Penuh)"),
        ("[M]", f"Audio Alarm ({'ON' if alarm_enabled else 'MUTE'})"),
        ("[S]", "Simpan Screenshot"),
        ("[Q]", "Keluar Sistem"),
    ]

    for i, (key_label, desc) in enumerate(shortcuts):
        sy = card_r_y + 52 + i * 24
        cv2.putText(canvas, key_label, (card_r_x + 14, sy), cv2.FONT_HERSHEY_DUPLEX, 0.38, YELLOW_GOLD, 1, cv2.LINE_AA)
        cv2.putText(canvas, desc, (card_r_x + 65, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_WHITE, 1, cv2.LINE_AA)

    # ----------------------------------------------------
    # 6. CENTER BIG ALERT (When Danger Detected)
    # ----------------------------------------------------
    if is_danger:
        alert_text = "!! WASPADA JALAN BERLUBANG !!"
        afont = cv2.FONT_HERSHEY_DUPLEX
        ascale = 0.95
        athick = 2
        (atw, ath), _ = cv2.getTextSize(alert_text, afont, ascale, athick)
        alert_x = (w - atw) // 2
        alert_y = int(h * 0.50)

        # Black outline for 100% contrast on any road surface
        cv2.putText(canvas, alert_text, (alert_x, alert_y), afont, ascale, (0, 0, 0), athick + 4, cv2.LINE_AA)
        cv2.putText(canvas, alert_text, (alert_x, alert_y), afont, ascale, (25, 35, 255), athick, cv2.LINE_AA)

    # ----------------------------------------------------
    # 7. BOTTOM BAR (Shortcuts & Status)
    # ----------------------------------------------------
    cv2.line(canvas, (0, h - bottom_h), (w, h - bottom_h), (0, 190, 230), 1)
    cv2.line(canvas, (0, h - 1), (w, h - 1), YELLOW_GOLD, 2)

    status_str = "BAHAYA LUBANG" if is_danger else "JALAN AMAN"
    alarm_str = "ON" if alarm_enabled else "MUTE"
    cam_str = "Sesuai Tubuh" if is_mirrored else "Dashcam Jalan"
    bot_text = f"RoadGuard AI | Status: [{status_str}] | Kamera: [{cam_str}] | Memori: [{persistence_frames}f] | Alarm: [{alarm_str}] | Conf: {conf_threshold:.2f} | [W] Fullscreen | [F] Kamera | [Q] Keluar"
    (btw2, _), _ = cv2.getTextSize(bot_text, cv2.FONT_HERSHEY_SIMPLEX, 0.39, 1)
    bx = (w - btw2) // 2
    cv2.putText(canvas, bot_text, (bx, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.39, TEXT_MUTED, 1, cv2.LINE_AA)

    return canvas


def run_camera(
    model_path: str = None,
    camera_id: int = 0,
    conf: float = 0.25,
    warning_frames: int = 2,
    normal_frames: int = 10,
    persistence_frames: int = 15,
    is_mirrored: bool = True,
):
    """
    Main camera loop with widescreen aspect-ratio preservation and HUD overlay.
    """
    if model_path is None:
        model_path = str(PROJECT_ROOT / "models" / "best.pt")

    cam_str = "Sesuai Posisi Tubuh (Seperti Foto Kedua)" if is_mirrored else "Mode Dashcam Jalan"
    print("\n" + "=" * 60)
    print("  🛡️  ROADGUARD AI - SISTEM DETEKSI JALAN BERLUBANG")
    print("=" * 60)
    print(f"  📦 Model               : {model_path}")
    print(f"  📷 Kamera ID           : {camera_id}")
    print(f"  🎯 Confidence Threshold: {conf:.2f}")
    print(f"  📷 Posisi Kamera       : {cam_str}")
    print(f"  ⏱️  Stabilisasi         : Bahaya >= {warning_frames} frame, Aman >= {normal_frames} frame")
    print(f"  🧠 Memori Sensor       : {persistence_frames} frame (Anti-Kedap-Kedip / Tetap Menetap)")
    print("=" * 60)
    print("  ⌨️  KONTROL KEYBOARD:")
    print("     [Q] / [ESC] : Keluar")
    print("     [+] / [-]   : Atur Confidence Threshold")
    print("     [F]         : Ganti Posisi Kamera (Sesuai Tubuh / Dashcam Jalan)")
    print("     [W]         : Toggle Fullscreen (Layar Penuh)")
    print("     [T]         : Atur Durasi Memori Sensor (Kotak Menetap)")
    print("     [M]         : Mute / Nyalakan Alarm 'TIT TIT TIT'")
    print("     [S]         : Simpan Screenshot")
    print("=" * 60 + "\n")

    # Initialize Detector with persistence tracker
    detector = PotholeDetector(
        model_path=model_path,
        confidence=conf,
        persistence_frames=persistence_frames,
    )
    if not detector._is_loaded:
        if not detector.load_model():
            print(f"❌ Gagal memuat model dari: {model_path}")
            return

    # Initialize Temporal Stabilizer
    stabilizer = TemporalStabilizer(
        warning_frames=warning_frames,
        normal_frames=normal_frames,
    )

    # Initialize Audio Alarm
    alarm = AlarmSystem()
    alarm_enabled = True

    # Open Camera with DirectShow on Windows for instant 16:9 capture
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_id)
    else:
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"❌ Tidak dapat membuka kamera ID {camera_id}")
        return

    # Request HD 16:9 widescreen
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "RoadGuard AI \u2014 Multi-Hazard Road Damage Tracking v3.0"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(window_name, 1280, 720)

    is_fullscreen = False
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                print("⚠️ Gagal membaca frame dari webcam.")
                break

            # Calculate FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 0.001))
            prev_time = curr_time

            # ==================================================
            # 1. MIRROR / FLIP CAMERA (REAL-TIME POSISI TUBUH)
            # ==================================================
            # Balikkan horizontal jika mirror aktif agar gerakan sinkron seperti cermin
            if is_mirrored:
                raw_frame = cv2.flip(raw_frame, 1)

            # ==================================================
            # 2. AUTO-RESIZE TO FIT WINDOW (ELIMINATE EMPTY / GREY SPACE)
            # ==================================================
            # Deteksi ukuran jendela aktual secara dinamis agar saat di-maximize (diperbesar)
            # tampilan video otomatis memenuhi 100% layar tanpa menyisakan ruang kosong / abu-abu
            target_w, target_h = 1280, 720
            try:
                rect = cv2.getWindowImageRect(window_name)
                if rect is not None and len(rect) >= 4:
                    rw, rh = int(rect[2]), int(rect[3])
                    if rw > 320 and rh > 240:
                        target_w, target_h = rw, rh
            except Exception:
                target_w, target_h = 1280, 720

            # Crops and scales cleanly without unequal stretching (Anti-Gepeng, Auto-Fill)
            frame = normalize_widescreen(raw_frame, target_w=target_w, target_h=target_h)

            # ==================================================
            # 3. RUN DETECTION + TEMPORAL TRACKER
            # ==================================================
            # Returns tracked bounding boxes that persist smoothly without disappearing rapidly
            tracked_detections = detector.detect_and_track(frame, conf_threshold=detector.confidence)
            has_valid_pothole = len(tracked_detections) > 0

            # ==================================================
            # 4. TEMPORAL STABILIZATION
            # ==================================================
            is_confirmed_danger = stabilizer.update(has_valid_pothole)

            # ==================================================
            # 5. SYNCHRONIZED ALARM TRIGGER
            # ==================================================
            if alarm_enabled and is_confirmed_danger:
                alarm.start()
            else:
                alarm.stop()

            # ==================================================
            # 6. RENDER HUD DASHBOARD
            # ==================================================
            display_canvas = render_hud_dashboard(
                canvas=frame,
                valid_detections=tracked_detections,
                is_danger=is_confirmed_danger,
                conf_threshold=detector.confidence,
                fps=fps,
                alarm_enabled=alarm_enabled,
                persistence_frames=detector.tracker.persistence_frames,
                is_mirrored=is_mirrored,
            )

            cv2.imshow(window_name, display_canvas)

            # Check if user clicked window close button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('+') or key == ord('='):
                detector.confidence = min(0.95, detector.confidence + 0.05)
                print(f"🎯 Confidence Threshold: {detector.confidence:.2f}")
            elif key == ord('-') or key == ord('_'):
                detector.confidence = max(0.05, detector.confidence - 0.05)
                print(f"🎯 Confidence Threshold: {detector.confidence:.2f}")
            elif key == ord('f') or key == ord('F'):
                is_mirrored = not is_mirrored
                state_str = "Sesuai Posisi Tubuh (Seperti Foto Kedua)" if is_mirrored else "Mode Dashcam Jalan (Normal)"
                print(f"🔄 Posisi Kamera diubah: {state_str}")
            elif key == ord('w') or key == ord('W'):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("🖥️ Layar Penuh (Fullscreen): AKTIF")
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    print("🖥️ Layar Penuh (Fullscreen): NONAKTIF")
            elif key == ord('t') or key == ord('T'):
                # Cycle persistence levels: 5 -> 10 -> 15 -> 25 -> 40
                levels = [5, 10, 15, 25, 40]
                curr = detector.tracker.persistence_frames
                next_idx = (levels.index(curr) + 1) % len(levels) if curr in levels else 2
                detector.tracker.persistence_frames = levels[next_idx]
                print(f"⏱️ Memori Sensor (Persistence): {detector.tracker.persistence_frames} frame")
            elif key == ord('m') or key == ord('M'):
                alarm_enabled = not alarm_enabled
                if not alarm_enabled:
                    alarm.stop()
                print(f"🔊 Alarm Audio: {'AKTIF' if alarm_enabled else 'MUTE'}")
            elif key == ord('s') or key == ord('S'):
                filename = f"capture_pothole_{int(time.time())}.jpg"
                cv2.imwrite(filename, display_canvas)
                print(f"📸 Screenshot disimpan: {filename}")

    finally:
        alarm.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Aplikasi ditutup cleanly. Sampai jumpa!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROADGUARD AI - Deteksi Jalan Berlubang")
    parser.add_argument("--model", type=str, default=None, help="Path ke model .pt")
    parser.add_argument("--camera", type=int, default=0, help="ID Kamera")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--persistence", type=int, default=15, help="Jumlah frame kotak sensor bertahan/menetap")
    parser.add_argument("--warning-frames", type=int, default=2, help="Konfirmasi frame bahaya")
    parser.add_argument("--normal-frames", type=int, default=10, help="Konfirmasi frame normal")
    parser.add_argument("--no-flip", "--dashcam", dest="flip", action="store_false", default=True, help="Mode dashcam jalan normal")
    parser.add_argument("--flip", "--mirror", dest="flip", action="store_true", default=True, help="Sesuai posisi tubuh seperti foto kedua (default: AKTIF)")
    args = parser.parse_args()

    run_camera(
        model_path=args.model,
        camera_id=args.camera,
        conf=args.conf,
        warning_frames=args.warning_frames,
        normal_frames=args.normal_frames,
        persistence_frames=args.persistence,
        is_mirrored=args.flip,
    )
