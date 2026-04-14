#!/usr/bin/env python3
"""Voice recording, transcription, and TTS management.

Extracted from HermesCLI to isolate voice-mode state and audio I/O
from the main CLI interaction loop.
"""

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class VoiceManager:
    """Manages voice recording, transcription, and text-to-speech.

    Encapsulates all voice-related state and operations that were previously
    scattered across the HermesCLI class.  The CLI creates one VoiceManager
    and delegates voice commands to it.

    Parameters
    ----------
    print_fn : callable
        Function to use for status output (CLI uses its _cprint).
    hermes_home : Path
        Root directory for audio cache and temp files.
    whisper_model : str
        Whisper model name for transcription (default: ``"base"``).
    tts_enabled : bool
        Whether TTS playback is active (default: ``False``).
    """

    def __init__(
        self,
        print_fn: Callable = print,
        hermes_home: Optional[Path] = None,
        whisper_model: str = "base",
        tts_enabled: bool = False,
    ):
        self._print = print_fn
        self._hermes_home = hermes_home or Path.home() / ".hermes"
        self._whisper_model = whisper_model

        # Voice state
        self.recording = False
        self.tts_enabled = tts_enabled
        self.voice_mode_enabled = False
        self._recording_process = None
        self._audio_file = None

    # ── Recording ──────────────────────────────────────────────────

    def start_recording(self) -> bool:
        """Start capturing audio from the default microphone.

        Uses ``arecord`` (ALSA) on Linux.  Writes a WAV file to the
        ``audio_cache/`` directory under ``~/.hermes``.

        Returns True if recording started successfully.
        """
        if self.recording:
            self._print("Already recording.")
            return False

        audio_dir = self._hermes_home / "audio_cache"
        audio_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._audio_file = audio_dir / f"recording_{timestamp}.wav"

        try:
            self._recording_process = subprocess.Popen(
                [
                    "arecord",
                    "-f", "S16_LE",
                    "-r", "16000",
                    "-c", "1",
                    str(self._audio_file),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.recording = True
            self._print("Recording started... Press Enter or /voice to stop.")
            return True
        except FileNotFoundError:
            self._print("arecord not found. Install ALSA utilities: sudo apt install alsa-utils")
            return False
        except Exception as e:
            self._print(f"Failed to start recording: {e}")
            return False

    def stop_and_transcribe(self) -> Optional[str]:
        """Stop recording and transcribe the captured audio.

        Uses the Whisper model specified at construction time.
        Returns the transcribed text, or None on failure.
        """
        if not self.recording or self._recording_process is None:
            return None

        try:
            self._recording_process.terminate()
            self._recording_process.wait(timeout=5)
        except Exception:
            pass
        finally:
            self.recording = False
            self._recording_process = None

        if not self._audio_file or not self._audio_file.exists():
            self._print("No audio file to transcribe.")
            return None

        try:
            import whisper
            model = whisper.load_model(self._whisper_model)
            result = model.transcribe(str(self._audio_file))
            text = result.get("text", "").strip()
            if text:
                self._print(f"Transcribed: {text[:100]}{'...' if len(text) > 100 else ''}")
            return text or None
        except ImportError:
            self._print("Whisper not installed. Install: pip install openai-whisper")
            return None
        except Exception as e:
            self._print(f"Transcription failed: {e}")
            return None

    # ── TTS ────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Speak text aloud using TTS.

        Currently uses ``espeak`` on Linux.  Runs in a background thread
        to avoid blocking the CLI.
        """
        if not self.tts_enabled or not text:
            return

        def _speak():
            try:
                subprocess.run(
                    ["espeak", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            except FileNotFoundError:
                logger.debug("espeak not found, skipping TTS")
            except Exception as e:
                logger.debug("TTS playback failed: %s", e)

        threading.Thread(target=_speak, daemon=True).start()

    # ── Toggle helpers ─────────────────────────────────────────────

    def toggle_tts(self) -> bool:
        """Toggle TTS on/off.  Returns the new state."""
        self.tts_enabled = not self.tts_enabled
        state = "enabled" if self.tts_enabled else "disabled"
        self._print(f"TTS {state}.")
        return self.tts_enabled

    def enable_voice_mode(self) -> None:
        """Enable full voice mode (recording + TTS)."""
        self.voice_mode_enabled = True
        self.tts_enabled = True
        self._print("Voice mode enabled. Use /voice to record, /voice off to disable.")

    def disable_voice_mode(self) -> None:
        """Disable voice mode."""
        if self.recording:
            self.stop_and_transcribe()
        self.voice_mode_enabled = False
        self.tts_enabled = False
        self._print("Voice mode disabled.")

    def status(self) -> dict[str, Any]:
        """Return current voice state for status display."""
        return {
            "voice_mode": self.voice_mode_enabled,
            "recording": self.recording,
            "tts": self.tts_enabled,
            "whisper_model": self._whisper_model,
        }
