"""
==========================================================
🧠 Pothole Detector - YOLOv8 Inference Engine
==========================================================
Wraps the YOLOv8 model for real-time pothole detection
on webcam frames. Handles model loading, inference,
and detection visualization.
==========================================================
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Set
import time
import math


class Detection:
    """Single detection result."""

    def __init__(self, bbox: tuple, confidence: float, class_id: int, label: str):
        self.bbox = bbox              # (x1, y1, x2, y2)
        self.confidence = confidence  # 0.0 - 1.0
        self.class_id = class_id
        self.label = label
        self.area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def severity(self) -> str:
        """Estimate severity based on area and confidence."""
        score = self.confidence * min(self.area / 10000.0, 1.0)
        if score > 0.6:
            return "KRITIS"
        elif score > 0.3:
            return "SEDANG"
        return "RINGAN"


class TrackedDetection:
    """
    Represents an actively tracked pothole detection with temporal memory and smoothing.
    Keeps bounding boxes stably visible across frames without flickering or instant vanishing.
    """

    def __init__(
        self,
        track_id: int,
        bbox: tuple,
        confidence: float,
        class_id: int,
        label: str,
    ):
        self.track_id = track_id
        self.bbox = bbox              # (x1, y1, x2, y2) smoothed
        self.raw_bbox = bbox
        self.confidence = confidence  # 0.0 - 1.0
        self.class_id = class_id
        self.label = label
        self.area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

        self.frames_seen = 1
        self.frames_missed = 0
        self.first_seen_time = time.time()
        self.last_seen_time = time.time()
        self.is_held = False          # True if retained via persistence buffer during momentary miss

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def severity(self) -> str:
        """Estimate severity based on area and confidence."""
        score = self.confidence * min(self.area / 10000.0, 1.0)
        if score > 0.6:
            return "KRITIS"
        elif score > 0.3:
            return "SEDANG"
        return "RINGAN"

    def update(self, new_bbox: tuple, new_conf: float, alpha: float = 0.65):
        """Smoothly update bbox coordinates and confidence using Exponential Moving Average."""
        x1, y1, x2, y2 = self.bbox
        nx1, ny1, nx2, ny2 = new_bbox

        sx1 = int(round(alpha * nx1 + (1.0 - alpha) * x1))
        sy1 = int(round(alpha * ny1 + (1.0 - alpha) * y1))
        sx2 = int(round(alpha * nx2 + (1.0 - alpha) * x2))
        sy2 = int(round(alpha * ny2 + (1.0 - alpha) * y2))

        self.bbox = (sx1, sy1, sx2, sy2)
        self.raw_bbox = new_bbox
        self.confidence = float(alpha * new_conf + (1.0 - alpha) * self.confidence)
        self.area = max(0, sx2 - sx1) * max(0, sy2 - sy1)

        self.frames_seen += 1
        self.frames_missed = 0
        self.last_seen_time = time.time()
        self.is_held = False

    def mark_missed(self):
        """Called when this track is not detected in the current frame."""
        self.frames_missed += 1
        self.is_held = True


class PotholeTracker:
    """
    Stabilizes detection bounding boxes across frames.
    - Eliminates box flickering
    - Keeps boxes firmly visible (persistence / memory hold)
    - Smooths bounding box coordinates to prevent jitter
    - Uses ByteTrack-style two-stage thresholding (high & low conf matching)
    """

    def __init__(
        self,
        persistence_frames: int = 15,
        iou_threshold: float = 0.20,
        smooth_alpha: float = 0.65,
    ):
        self.persistence_frames = persistence_frames
        self.iou_threshold = iou_threshold
        self.smooth_alpha = smooth_alpha
        self.tracks: List[TrackedDetection] = []
        self._next_id = 1

    def reset(self):
        """Clear all active tracks."""
        self.tracks.clear()
        self._next_id = 1

    @staticmethod
    def _compute_match_score(box_a: tuple, box_b: tuple) -> Tuple[bool, float]:
        """
        Computes spatial match between two bounding boxes using IoU and normalized center distance.
        Returns: (is_match, score)
        """
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b

        ix1 = max(xa1, xb1)
        iy1 = max(ya1, yb1)
        ix2 = min(xa2, xb2)
        iy2 = min(ya2, yb2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter_area = iw * ih

        area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
        area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
        union_area = area_a + area_b - inter_area

        iou = (inter_area / union_area) if union_area > 0 else 0.0

        cx_a = (xa1 + xa2) / 2.0
        cy_a = (ya1 + ya2) / 2.0
        cx_b = (xb1 + xb2) / 2.0
        cy_b = (yb1 + yb2) / 2.0

        diag_a = max(1.0, math.hypot(xa2 - xa1, ya2 - ya1))
        diag_b = max(1.0, math.hypot(xb2 - xb1, yb2 - yb1))
        avg_diag = (diag_a + diag_b) / 2.0

        dist = math.hypot(cx_a - cx_b, cy_a - cy_b)
        rel_dist = dist / avg_diag

        # Match condition:
        # 1. IoU >= 0.20
        # 2. Or IoU >= 0.08 with close centers
        # 3. Or centers very close (rel_dist <= 0.40)
        is_match = (iou >= 0.20) or (iou >= 0.08 and rel_dist <= 0.60) or (rel_dist <= 0.40)
        score = iou - 0.4 * rel_dist
        return is_match, score

    def update(
        self,
        high_detections: List[Detection],
        low_detections: Optional[List[Detection]] = None,
    ) -> List[TrackedDetection]:
        """
        Updates tracks with detections from current frame.
        - high_detections: Detections >= conf_threshold (can create new tracks or update existing)
        - low_detections: Detections between low_thresh and conf_threshold (can only update existing tracks)
        """
        if low_detections is None:
            low_detections = []

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_high_dets = set(range(len(high_detections)))

        # Stage 1: Match existing tracks with HIGH confidence detections
        matches_stage1 = []
        for t_idx in unmatched_tracks:
            for d_idx in unmatched_high_dets:
                is_match, score = self._compute_match_score(
                    self.tracks[t_idx].bbox, high_detections[d_idx].bbox
                )
                if is_match:
                    matches_stage1.append((score, t_idx, d_idx))

        matches_stage1.sort(key=lambda x: x[0], reverse=True)

        matched_tracks_s1: Set[int] = set()
        matched_dets_s1: Set[int] = set()
        for _, t_idx, d_idx in matches_stage1:
            if t_idx not in matched_tracks_s1 and d_idx not in matched_dets_s1:
                matched_tracks_s1.add(t_idx)
                matched_dets_s1.add(d_idx)
                det = high_detections[d_idx]
                self.tracks[t_idx].update(
                    det.bbox, det.confidence, self.smooth_alpha
                )

        unmatched_tracks -= matched_tracks_s1
        unmatched_high_dets -= matched_dets_s1

        # Stage 2: Match remaining tracks with LOW confidence detections (ByteTrack principle)
        if unmatched_tracks and low_detections:
            matches_stage2 = []
            for t_idx in unmatched_tracks:
                for d_idx, low_det in enumerate(low_detections):
                    is_match, score = self._compute_match_score(
                        self.tracks[t_idx].bbox, low_det.bbox
                    )
                    if is_match:
                        matches_stage2.append((score, t_idx, d_idx))

            matches_stage2.sort(key=lambda x: x[0], reverse=True)
            matched_tracks_s2: Set[int] = set()
            matched_dets_s2: Set[int] = set()
            for _, t_idx, d_idx in matches_stage2:
                if t_idx not in matched_tracks_s2 and d_idx not in matched_dets_s2:
                    matched_tracks_s2.add(t_idx)
                    matched_dets_s2.add(d_idx)
                    det = low_detections[d_idx]
                    self.tracks[t_idx].update(
                        det.bbox, det.confidence, self.smooth_alpha
                    )

            unmatched_tracks -= matched_tracks_s2

        # Stage 3: For unmatched tracks, mark as missed (kept alive by persistence window)
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()

        # Stage 4: Create new tracks for unmatched HIGH confidence detections
        for d_idx in unmatched_high_dets:
            det = high_detections[d_idx]
            new_track = TrackedDetection(
                track_id=self._next_id,
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                label=det.label,
            )
            self._next_id += 1
            self.tracks.append(new_track)

        # Stage 5: Remove tracks that exceeded persistence duration
        self.tracks = [
            t for t in self.tracks
            if t.frames_missed <= self.persistence_frames
        ]

        return list(self.tracks)


class TemporalStabilizer:
    """
    Stabilizes detection state across frames to prevent flickering.
    Source of truth: presence of >= 1 valid YOLO pothole bounding box.
    """

    def __init__(self, warning_frames: int = 2, normal_frames: int = 10):
        self.warning_threshold = warning_frames
        self.normal_threshold = normal_frames
        self.pothole_counter = 0
        self.clear_counter = 0
        self.is_confirmed_danger = False

    def update(self, has_valid_pothole: bool) -> bool:
        """
        Updates consecutive frame counters and returns confirmed danger status.
        WARNING_CONFIRMATION_FRAMES (default: 2):
            Must detect valid bounding box for 2 consecutive frames before changing NORMAL -> BAHAYA.
        NORMAL_CONFIRMATION_FRAMES (default: 10):
            Must have 0 valid bounding boxes for 10 consecutive frames before changing BAHAYA -> NORMAL.
        """
        if has_valid_pothole:
            self.pothole_counter += 1
            self.clear_counter = 0
            if self.pothole_counter >= self.warning_threshold:
                self.is_confirmed_danger = True
        else:
            self.clear_counter += 1
            self.pothole_counter = 0
            if self.clear_counter >= self.normal_threshold:
                self.is_confirmed_danger = False

        return self.is_confirmed_danger

    def reset(self):
        self.pothole_counter = 0
        self.clear_counter = 0
        self.is_confirmed_danger = False


class PotholeDetector:
    """
    YOLOv8-based pothole detection engine.
    Loads the trained model and runs inference on video frames.
    """

    # Color scheme for visualizations
    COLORS = {
        'danger_box': (0, 0, 255),       # Red - bounding box
        'danger_fill': (0, 0, 200),       # Dark red - fill overlay
        'danger_text_bg': (0, 0, 180),    # Red - text background
        'safe_text': (0, 255, 100),       # Green
        'text': (255, 255, 255),          # White
        'info_bg': (40, 40, 40),          # Dark gray
    }

    def __init__(
        self,
        model_path: str = None,
        confidence: float = 0.25,
        persistence_frames: int = 15,
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to trained YOLOv8 model (.pt file)
            confidence: Minimum confidence threshold (0.0 - 1.0)
            persistence_frames: Number of frames to hold bounding box before clearing
        """
        if model_path is None:
            model_path = str(Path(__file__).parent.parent / "models" / "best.pt")

        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        self._is_loaded = False
        self._fps_counter = FPSCounter()

        # Initialize persistence tracker
        self.tracker = PotholeTracker(
            persistence_frames=persistence_frames,
            iou_threshold=0.20,
            smooth_alpha=0.65,
        )

        # Class names
        self.class_names = {0: "Lubang Jalan"}

        # Auto-load model if file exists
        if self.model_path and Path(self.model_path).exists():
            self.load_model()

    def load_model(self) -> bool:
        """
        Load the YOLOv8 model.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            from ultralytics import YOLO

            if not Path(self.model_path).exists():
                print(f"❌ Model not found: {self.model_path}")
                print("   Train a model first: python training/train.py")
                return False

            print(f"🧠 Loading model: {self.model_path}")
            self.model = YOLO(self.model_path)
            self._is_loaded = True
            print("✅ Model loaded successfully!")
            return True

        except ImportError:
            print("❌ Ultralytics not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def detect(self, frame: np.ndarray, conf_threshold: float = None) -> List[Detection]:
        """
        Run pothole detection on a single frame.

        Args:
            frame: BGR image from OpenCV (uncropped, original aspect ratio)
            conf_threshold: Optional override for confidence threshold

        Returns:
            List of valid Detection objects meeting the threshold
        """
        if not self._is_loaded:
            return []

        self._fps_counter.tick()
        thresh = conf_threshold if conf_threshold is not None else self.confidence

        # Run YOLOv8 inference directly on full frame
        results = self.model(
            frame,
            conf=thresh,
            verbose=False,
            stream=False,
            imgsz=640,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                conf = float(box.conf[0])
                if conf < thresh:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                label = self.class_names.get(cls, "Lubang Jalan")

                det = Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    class_id=cls,
                    label=label,
                )
                detections.append(det)

        return detections

    def detect_and_track(
        self,
        frame: np.ndarray,
        conf_threshold: float = None,
    ) -> List[TrackedDetection]:
        """
        Run pothole detection AND temporal tracking on a frame.
        Maintains box persistence and smoothing so boxes stay firmly visible.

        Args:
            frame: BGR image from OpenCV
            conf_threshold: Optional override for confidence threshold

        Returns:
            List of actively tracked (and persisted) TrackedDetection objects
        """
        if not self._is_loaded:
            return []

        self._fps_counter.tick()
        thresh = conf_threshold if conf_threshold is not None else self.confidence
        # Low confidence threshold for ByteTrack-style continuation
        low_thresh = max(0.12, thresh * 0.6)

        results = self.model(
            frame,
            conf=low_thresh,
            verbose=False,
            stream=False,
            imgsz=640,
        )

        high_detections = []
        low_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                conf = float(box.conf[0])
                if conf < low_thresh:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                label = self.class_names.get(cls, "Lubang Jalan")

                det = Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    class_id=cls,
                    label=label,
                )
                if conf >= thresh:
                    high_detections.append(det)
                else:
                    low_detections.append(det)

        return self.tracker.update(high_detections, low_detections)

    def draw_detections(self, frame: np.ndarray, detections: list,
                        show_info: bool = True) -> np.ndarray:
        """
        Draw detection results on the frame.
        Supports both Detection and TrackedDetection objects.

        Args:
            frame: Original BGR frame
            detections: List of Detection or TrackedDetection objects
            show_info: Whether to show FPS and info overlay

        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        has_detection = len(detections) > 0

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            conf = det.confidence
            severity = det.severity
            is_held = getattr(det, 'is_held', False)

            # ---- Danger zone overlay (semi-transparent red fill) ----
            overlay = annotated.copy()
            fill_color = (0, 0, 160) if is_held else self.COLORS['danger_fill']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
            cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)

            # ---- Bounding box with thick border ----
            border_color = (0, 140, 255) if is_held else self.COLORS['danger_box']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), border_color, 2)

            # ---- Corner markers for visual style ----
            corner_len = min(25, max(8, (x2 - x1) // 4), max(8, (y2 - y1) // 4))
            corner_col = (0, 215, 255) if not is_held else (0, 165, 255)
            corners = [
                ((x1, y1), (x1 + corner_len, y1), (x1, y1 + corner_len)),
                ((x2, y1), (x2 - corner_len, y1), (x2, y1 + corner_len)),
                ((x1, y2), (x1 + corner_len, y2), (x1, y2 - corner_len)),
                ((x2, y2), (x2 - corner_len, y2), (x2, y2 - corner_len)),
            ]
            for corner, h_end, v_end in corners:
                cv2.line(annotated, corner, h_end, corner_col, 3)
                cv2.line(annotated, corner, v_end, corner_col, 3)

            # ---- Label with background ----
            hold_tag = " [MEMORI]" if is_held else ""
            label = f"{det.label} {conf:.0%} [{severity}]{hold_tag}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            label_y = max(y1 - 10, th + 10)
            text_bg = (15, 20, 28)
            cv2.rectangle(annotated,
                          (x1, label_y - th - 8),
                          (x1 + tw + 12, label_y + 4),
                          text_bg, -1)
            cv2.rectangle(annotated,
                          (x1, label_y - th - 8),
                          (x1 + tw + 12, label_y + 4),
                          border_color, 1)
            cv2.putText(annotated, label,
                        (x1 + 6, label_y - 2),
                        font, font_scale, (0, 215, 255), thickness)

            # ---- Center crosshair ----
            cx, cy = det.center
            cross_size = 10
            cv2.line(annotated, (cx - cross_size, cy), (cx + cross_size, cy),
                     border_color, 2)
            cv2.line(annotated, (cx, cy - cross_size), (cx, cy + cross_size),
                     border_color, 2)

        # ---- Top status bar ----
        if show_info:
            self._draw_status_bar(annotated, detections, has_detection)

        return annotated

    def _draw_status_bar(self, frame: np.ndarray, detections: List[Detection],
                         has_danger: bool):
        """Draw the top status bar overlay."""
        h, w = frame.shape[:2]
        bar_height = 50

        # Semi-transparent background
        overlay = frame.copy()
        color = (0, 0, 180) if has_danger else (0, 120, 0)
        cv2.rectangle(overlay, (0, 0), (w, bar_height), color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Status text
        if has_danger:
            status = f"  BAHAYA! {len(detections)} LUBANG TERDETEKSI"
            icon = "!!"
        else:
            status = "  JALAN AMAN - Tidak Ada Lubang"
            icon = "OK"

        # Icon circle
        cv2.circle(frame, (30, bar_height // 2), 15,
                   (255, 255, 255), -1)
        cv2.putText(frame, icon, (18, bar_height // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Status text
        cv2.putText(frame, status, (55, bar_height // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # FPS counter
        fps_text = f"FPS: {self._fps_counter.fps:.0f}"
        cv2.putText(frame, fps_text, (w - 120, bar_height // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    def set_confidence(self, confidence: float):
        """Update confidence threshold."""
        self.confidence = max(0.1, min(1.0, confidence))

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def fps(self) -> float:
        return self._fps_counter.fps


class FPSCounter:
    """Tracks frames per second."""

    def __init__(self, avg_count: int = 30):
        self._times = []
        self._avg_count = avg_count
        self.fps = 0.0

    def tick(self):
        now = time.time()
        self._times.append(now)

        # Keep only recent timestamps
        if len(self._times) > self._avg_count:
            self._times = self._times[-self._avg_count:]

        if len(self._times) >= 2:
            elapsed = self._times[-1] - self._times[0]
            if elapsed > 0:
                self.fps = (len(self._times) - 1) / elapsed
