"""
==========================================================
🔊 Alarm System - Pothole Detection
==========================================================
Generates and plays "TIT TIT TIT TIT" alarm sound
when a pothole is detected. Uses pygame for audio.
==========================================================
"""

import wave
import struct
import math
import threading
import os
from pathlib import Path


class AlarmSystem:
    """
    Audio alarm system that plays a repeating "TIT TIT TIT TIT"
    beep pattern when potholes are detected.
    """

    def __init__(self, assets_dir: str = None):
        if assets_dir is None:
            assets_dir = str(Path(__file__).parent / "assets")

        self.assets_dir = assets_dir
        self.alarm_file = os.path.join(assets_dir, "alarm_beep.wav")
        self.is_playing = False
        self._stop_event = threading.Event()
        self._thread = None
        self._mixer_initialized = False

        # Generate alarm sound file if it doesn't exist
        os.makedirs(assets_dir, exist_ok=True)
        if not os.path.exists(self.alarm_file):
            self._generate_alarm_sound()

        # Initialize pygame mixer
        self._init_mixer()

    def _init_mixer(self):
        """Initialize pygame mixer for audio playback."""
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self._sound = pygame.mixer.Sound(self.alarm_file)
            self._sound.set_volume(0.7)
            self._mixer_initialized = True
        except Exception as e:
            print(f"⚠️ Audio system error: {e}")
            print("   Alarm will be silent. Install pygame: pip install pygame")
            self._mixer_initialized = False

    def _generate_alarm_sound(self):
        """
        Generate a "TIT TIT TIT TIT" alarm WAV file.
        Pattern: 4 short high-pitched beeps with pauses.
        """
        sample_rate = 44100
        frequency = 2800       # Hz - high pitched "tit" sound
        beep_duration = 0.08   # 80ms per beep
        pause_duration = 0.08  # 80ms between beeps
        group_pause = 0.35     # 350ms pause between groups
        num_beeps = 4          # TIT TIT TIT TIT
        amplitude = 0.75

        samples = []

        for i in range(num_beeps):
            # Generate beep with fade-in/fade-out to avoid clicks
            beep_samples = int(sample_rate * beep_duration)
            fade_samples = int(beep_samples * 0.1)  # 10% fade

            for j in range(beep_samples):
                t = j / sample_rate

                # Sine wave
                value = amplitude * math.sin(2 * math.pi * frequency * t)

                # Add slight harmonic for more "electronic" sound
                value += 0.2 * amplitude * math.sin(2 * math.pi * frequency * 2 * t)

                # Fade in/out envelope
                if j < fade_samples:
                    value *= j / fade_samples
                elif j > beep_samples - fade_samples:
                    value *= (beep_samples - j) / fade_samples

                # Clamp and convert to 16-bit
                value = max(-1.0, min(1.0, value))
                samples.append(struct.pack('<h', int(value * 32767)))

            # Pause between beeps (silence)
            if i < num_beeps - 1:
                pause_samples = int(sample_rate * pause_duration)
                for _ in range(pause_samples):
                    samples.append(struct.pack('<h', 0))

        # Group pause at the end (before repeat)
        group_samples = int(sample_rate * group_pause)
        for _ in range(group_samples):
            samples.append(struct.pack('<h', 0))

        # Write WAV file
        with wave.open(self.alarm_file, 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b''.join(samples))

        print(f"  🔊 Alarm sound generated: {self.alarm_file}")

    def start(self):
        """Start playing the alarm in a loop."""
        if self.is_playing or not self._mixer_initialized:
            return

        self.is_playing = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the alarm."""
        if not self.is_playing:
            return

        self.is_playing = False
        self._stop_event.set()

        if self._mixer_initialized:
            try:
                self._sound.stop()
            except Exception:
                pass

    def update(self, has_detection: bool):
        """Update alarm state based on whether potholes are detected."""
        if has_detection:
            self.start()
        else:
            self.stop()

    def _play_loop(self):
        """Background thread that loops the alarm sound."""
        while not self._stop_event.is_set():
            try:
                self._sound.play()
                # Wait for sound duration + small buffer, or stop event
                self._stop_event.wait(timeout=0.9)
            except Exception:
                break

    def set_volume(self, volume: float):
        """Set alarm volume (0.0 to 1.0)."""
        if self._mixer_initialized:
            self._sound.set_volume(max(0.0, min(1.0, volume)))

    def cleanup(self):
        """Clean up audio resources."""
        self.stop()
        try:
            import pygame
            pygame.mixer.quit()
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        """Check if audio system is available."""
        return self._mixer_initialized
