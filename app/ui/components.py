"""
==========================================================
🎨 UI Components - Pothole Detection Dashboard
==========================================================
Custom widgets for the desktop application using
CustomTkinter for a modern dark-themed interface.
==========================================================
"""

import customtkinter as ctk
from datetime import datetime
from typing import Callable, Optional


class StatusPanel(ctk.CTkFrame):
    """
    Large status indicator showing AMAN (safe) or BAHAYA (danger).
    Features animated color transitions and detection count.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.configure(
            fg_color="#1a1a2e",
            corner_radius=15,
        )

        # Status icon / indicator
        self.status_frame = ctk.CTkFrame(
            self,
            fg_color="#0f3460",
            corner_radius=12,
            height=120,
        )
        self.status_frame.pack(fill="x", padx=15, pady=(15, 5))
        self.status_frame.pack_propagate(False)

        self.status_icon = ctk.CTkLabel(
            self.status_frame,
            text="✅",
            font=ctk.CTkFont(size=40),
            text_color="#00ff88",
        )
        self.status_icon.pack(pady=(10, 0))

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="JALAN AMAN",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#00ff88",
        )
        self.status_label.pack(pady=(0, 10))

        # Detection count
        self.count_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.count_frame.pack(fill="x", padx=15, pady=10)

        self.count_label = ctk.CTkLabel(
            self.count_frame,
            text="Lubang Terdeteksi",
            font=ctk.CTkFont(size=13),
            text_color="#8892b0",
        )
        self.count_label.pack()

        self.count_value = ctk.CTkLabel(
            self.count_frame,
            text="0",
            font=ctk.CTkFont(family="Consolas", size=48, weight="bold"),
            text_color="#ccd6f6",
        )
        self.count_value.pack()

        # Confidence display
        self.conf_label = ctk.CTkLabel(
            self,
            text="Confidence: -",
            font=ctk.CTkFont(family="Consolas", size=14),
            text_color="#8892b0",
        )
        self.conf_label.pack(pady=(0, 5))

        # FPS display
        self.fps_label = ctk.CTkLabel(
            self,
            text="FPS: 0",
            font=ctk.CTkFont(family="Consolas", size=14),
            text_color="#8892b0",
        )
        self.fps_label.pack(pady=(0, 15))

        # Blinking state
        self._blink_state = False
        self._is_danger = False

    def set_safe(self):
        """Set status to SAFE (green)."""
        self._is_danger = False
        self.status_frame.configure(fg_color="#0a4a2e")
        self.status_icon.configure(text="✅", text_color="#00ff88")
        self.status_label.configure(text="JALAN AMAN", text_color="#00ff88")
        self.count_value.configure(text="0", text_color="#00ff88")

    def set_danger(self, count: int, max_confidence: float):
        """Set status to DANGER (red, blinking)."""
        self._is_danger = True
        self.status_frame.configure(fg_color="#5c1a1a")
        self.status_icon.configure(text="⚠️", text_color="#ff4444")
        self.status_label.configure(text="BAHAYA!", text_color="#ff4444")
        self.count_value.configure(text=str(count), text_color="#ff4444")
        self.conf_label.configure(text=f"Confidence: {max_confidence:.1%}")

    def update_fps(self, fps: float):
        """Update FPS counter."""
        self.fps_label.configure(text=f"FPS: {fps:.0f}")

    def blink(self):
        """Toggle blink state for danger animation."""
        if self._is_danger:
            self._blink_state = not self._blink_state
            if self._blink_state:
                self.status_frame.configure(fg_color="#8b0000")
                self.status_label.configure(text="⚠️ BAHAYA! ⚠️")
            else:
                self.status_frame.configure(fg_color="#5c1a1a")
                self.status_label.configure(text="BAHAYA!")


class ControlPanel(ctk.CTkFrame):
    """
    Settings and control panel with confidence slider,
    alarm toggle, and action buttons.
    """

    def __init__(self, parent, on_confidence_change: Callable = None,
                 on_alarm_toggle: Callable = None, **kwargs):
        super().__init__(parent, **kwargs)

        self.configure(
            fg_color="#1a1a2e",
            corner_radius=15,
        )

        self._on_confidence_change = on_confidence_change
        self._on_alarm_toggle = on_alarm_toggle

        # Title
        title = ctk.CTkLabel(
            self,
            text="⚙️ Pengaturan",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#ccd6f6",
        )
        title.pack(pady=(15, 10), padx=15, anchor="w")

        # Confidence Threshold Slider
        conf_frame = ctk.CTkFrame(self, fg_color="transparent")
        conf_frame.pack(fill="x", padx=15, pady=5)

        conf_label = ctk.CTkLabel(
            conf_frame,
            text="Confidence Threshold",
            font=ctk.CTkFont(size=12),
            text_color="#8892b0",
        )
        conf_label.pack(anchor="w")

        self.conf_value_label = ctk.CTkLabel(
            conf_frame,
            text="45%",
            font=ctk.CTkFont(family="Consolas", size=14, weight="bold"),
            text_color="#64ffda",
        )
        self.conf_value_label.pack(anchor="e")

        self.conf_slider = ctk.CTkSlider(
            conf_frame,
            from_=10,
            to=95,
            number_of_steps=17,
            command=self._on_conf_slider,
            progress_color="#64ffda",
            button_color="#64ffda",
            button_hover_color="#4dd4b0",
        )
        self.conf_slider.set(45)
        self.conf_slider.pack(fill="x", pady=(0, 5))

        # Alarm Toggle
        alarm_frame = ctk.CTkFrame(self, fg_color="transparent")
        alarm_frame.pack(fill="x", padx=15, pady=10)

        alarm_label = ctk.CTkLabel(
            alarm_frame,
            text="🔊 Alarm Suara",
            font=ctk.CTkFont(size=13),
            text_color="#ccd6f6",
        )
        alarm_label.pack(side="left")

        self.alarm_switch = ctk.CTkSwitch(
            alarm_frame,
            text="",
            command=self._on_alarm_switch,
            progress_color="#64ffda",
            button_color="#ccd6f6",
            button_hover_color="#64ffda",
        )
        self.alarm_switch.select()  # ON by default
        self.alarm_switch.pack(side="right")

        # Volume Slider
        vol_frame = ctk.CTkFrame(self, fg_color="transparent")
        vol_frame.pack(fill="x", padx=15, pady=(0, 5))

        vol_label = ctk.CTkLabel(
            vol_frame,
            text="Volume",
            font=ctk.CTkFont(size=12),
            text_color="#8892b0",
        )
        vol_label.pack(anchor="w")

        self.vol_slider = ctk.CTkSlider(
            vol_frame,
            from_=0,
            to=100,
            command=self._on_vol_slider,
            progress_color="#64ffda",
            button_color="#64ffda",
            button_hover_color="#4dd4b0",
        )
        self.vol_slider.set(70)
        self.vol_slider.pack(fill="x")

        self._vol_callback = None

        # Separator
        sep = ctk.CTkFrame(self, fg_color="#233554", height=2)
        sep.pack(fill="x", padx=15, pady=10)

        # Model info
        self.model_label = ctk.CTkLabel(
            self,
            text="📦 Model: Tidak dimuat",
            font=ctk.CTkFont(size=12),
            text_color="#8892b0",
        )
        self.model_label.pack(padx=15, anchor="w", pady=(0, 15))

    def _on_conf_slider(self, value):
        val = int(value)
        self.conf_value_label.configure(text=f"{val}%")
        if self._on_confidence_change:
            self._on_confidence_change(val / 100.0)

    def _on_alarm_switch(self):
        if self._on_alarm_toggle:
            self._on_alarm_toggle(self.alarm_switch.get())

    def _on_vol_slider(self, value):
        if self._vol_callback:
            self._vol_callback(value / 100.0)

    def set_volume_callback(self, callback: Callable):
        self._vol_callback = callback

    def set_model_info(self, info: str):
        self.model_label.configure(text=f"📦 Model: {info}")

    @property
    def alarm_enabled(self) -> bool:
        return bool(self.alarm_switch.get())


class DetectionLog(ctk.CTkFrame):
    """
    Scrollable log showing recent detection events with timestamps.
    """

    def __init__(self, parent, max_entries: int = 50, **kwargs):
        super().__init__(parent, **kwargs)

        self.configure(
            fg_color="#1a1a2e",
            corner_radius=15,
        )

        self.max_entries = max_entries

        # Title
        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.pack(fill="x", padx=15, pady=(10, 5))

        title = ctk.CTkLabel(
            title_frame,
            text="📋 Log Deteksi",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#ccd6f6",
        )
        title.pack(side="left")

        clear_btn = ctk.CTkButton(
            title_frame,
            text="Hapus",
            width=60,
            height=25,
            font=ctk.CTkFont(size=11),
            fg_color="#233554",
            hover_color="#2d4a6f",
            command=self.clear,
        )
        clear_btn.pack(side="right")

        # Log text area
        self.log_text = ctk.CTkTextbox(
            self,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#0a0a1a",
            text_color="#8892b0",
            corner_radius=8,
            wrap="word",
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self._entry_count = 0

    def add_entry(self, message: str, level: str = "info"):
        """Add a log entry with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if level == "danger":
            prefix = "🔴"
        elif level == "warning":
            prefix = "🟡"
        else:
            prefix = "🟢"

        entry = f"[{timestamp}] {prefix} {message}\n"

        self.log_text.configure(state="normal")
        self.log_text.insert("end", entry)
        self.log_text.see("end")  # Auto scroll
        self.log_text.configure(state="disabled")

        self._entry_count += 1

        # Trim old entries
        if self._entry_count > self.max_entries:
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "2.0")
            self.log_text.configure(state="disabled")
            self._entry_count -= 1

    def clear(self):
        """Clear all log entries."""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self._entry_count = 0
