from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging


@dataclass
class VoiceConfig:
    rate: int = 150
    volume: float = 1.0
    energy_threshold: int = 4000


class VoiceIO:
    """
    Text-to-speech and speech-to-text wrapper.

    Notes:
    - Keeps the same underlying libraries as your current implementation.
    - Lazily initialises engines so importing the module is safe in GUI apps.
    """

    def __init__(self, config: Optional[VoiceConfig] = None, enabled: bool = True) -> None:
        self.config = config or VoiceConfig()
        self.enabled = enabled
        self.muted = False
        self._tts_engine = None
        self._recognizer = None
        self._log = logging.getLogger(__name__)

    def _ensure_tts(self) -> None:
        if self._tts_engine is not None:
            return
        # Local imports and initialization can produce noisy startup logs
        # (notably from comtypes on Windows). Temporarily raise log levels
        # and redirect stdout/stderr while importing/initialising pyttsx3.
        import os
        import logging
        import contextlib

        prev_levels: dict[str, int] = {}
        noisy_loggers = ("comtypes", "pyttsx3")
        for name in noisy_loggers:
            prev_levels[name] = logging.getLogger(name).level
            logging.getLogger(name).setLevel(logging.WARNING)

        devnull = open(os.devnull, "w")
        try:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                import pyttsx3  # local import for startup safety

                engine = pyttsx3.init()
                engine.setProperty("rate", self.config.rate)
                engine.setProperty("volume", self.config.volume)
                self._tts_engine = engine
        finally:
            devnull.close()
            for name, lvl in prev_levels.items():
                logging.getLogger(name).setLevel(lvl)

    def _ensure_stt(self) -> None:
        if self._recognizer is not None:
            return
        import speech_recognition as sr  # local import for startup safety

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = self.config.energy_threshold
        self._recognizer = recognizer

    def speak(self, text: str) -> None:
        if not self.enabled or self.muted:
            return
        self._ensure_tts()
        assert self._tts_engine is not None
        self._tts_engine.say(text)
        self._tts_engine.runAndWait()

    def listen(self) -> str:
        """
        Listen via microphone and return recognised text.
        Returns empty string on failure (keeps old behaviour).
        """
        if not self.enabled:
            return ""

        import speech_recognition as sr

        self._ensure_stt()
        assert self._recognizer is not None

        with sr.Microphone() as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=1)
            self._log.info("Listening from microphone")
            audio = self._recognizer.listen(source)
        try:
            return self._recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            self._log.warning("Speech recognition error: %s", e)
            return ""

