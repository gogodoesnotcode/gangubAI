"""Wake word detector for the GangubAI voice pipeline.

Phase 4 deliverables in this module:
- WakeWordDetector: wraps OpenWakeWord model loading and listening loop.
- listen_for_trigger() -> Literal["WAKE", "PTT"]
- Enter-key push-to-talk fallback in a background thread.
- interrupt()/shutdown() APIs to break out of blocking listen loops.

Usage:
    python3 -m ai.voice.wake_word
    python3 -m ai.voice.wake_word --timeout 20
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.signal
import sounddevice as sd

from ai.voice import config


class WakeWordDetector:
    """Detect wake-word events from microphone audio with PTT fallback."""

    def __init__(
        self,
        *,
        wake_word_model: str = config.WAKE_WORD_MODEL,
        feature_models_dir: str = config.WAKE_WORD_ASSETS_DIR,
        threshold: float = config.WAKE_WORD_THRESHOLD,
        sample_rate: int = config.OWW_SAMPLE_RATE,
        chunk_size: int = config.OWW_CHUNK_SIZE,
        input_device: str | int | None = config.INPUT_DEVICE_NAME,
        enable_keyboard_ptt: bool = config.ENABLE_KEYBOARD_PTT,
    ) -> None:
        self.wake_word_model = wake_word_model
        self.feature_models_dir = feature_models_dir
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.input_device = input_device
        self.enable_keyboard_ptt = enable_keyboard_ptt

        self._oww_model = None
        self._interrupt_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._ptt_event = threading.Event()
        self._keyboard_unavailable_event = threading.Event()

        self._stdin_thread: threading.Thread | None = None

        self._load_wake_word_model()
        if self.enable_keyboard_ptt:
            self._start_keyboard_thread()

    def _load_wake_word_model(self) -> None:
        """Load the OpenWakeWord model if package/model are available."""
        model_path = Path(self.wake_word_model)
        if not model_path.exists():
            print(
                f"[WakeWord] Model not found at {model_path}. "
                "Falling back to keyboard PTT.",
                flush=True,
            )
            return

        try:
            from openwakeword.model import Model  # type: ignore
        except Exception as exc:
            print(
                f"[WakeWord] openwakeword import failed: {exc}. "
                "Falling back to keyboard PTT.",
                flush=True,
            )
            return

        errors: list[str] = []

        model_kwargs: dict[str, str] = {}
        assets_dir = Path(self.feature_models_dir)
        melspec_path = assets_dir / "melspectrogram.onnx"
        embedding_path = assets_dir / "embedding_model.onnx"
        if melspec_path.exists() and embedding_path.exists():
            model_kwargs["melspec_model_path"] = str(melspec_path)
            model_kwargs["embedding_model_path"] = str(embedding_path)
        else:
            print(
                "[WakeWord] Feature ONNX models not found in assets dir; "
                "using OpenWakeWord defaults.",
                flush=True,
            )

        # Current API (preferred): wakeword_models + explicit ONNX inference.
        try:
            self._oww_model = Model(
                wakeword_models=[str(model_path)],
                inference_framework="onnx",
                **model_kwargs,
            )
            print(
                f"[WakeWord] Loaded wake-word model: {model_path} (wakeword_models, onnx).",
                flush=True,
            )
            return
        except Exception as exc:
            errors.append(f"wakeword_models API failed: {exc}")

        # Backward compatibility API.
        try:
            self._oww_model = Model(
                wakeword_model_paths=[str(model_path)],
                **model_kwargs,
            )
            print(
                f"[WakeWord] Loaded wake-word model: {model_path} (wakeword_model_paths).",
                flush=True,
            )
            return
        except Exception as exc:
            errors.append(f"wakeword_model_paths API failed: {exc}")

        # Older OpenWakeWord API variants (e.g. versions that don't accept
        # feature-model kwargs) can still work with minimal arguments.
        try:
            self._oww_model = Model(
                wakeword_model_paths=[str(model_path)],
            )
            print(
                f"[WakeWord] Loaded wake-word model: {model_path} (wakeword_model_paths minimal).",
                flush=True,
            )
            return
        except Exception as exc:
            errors.append(f"wakeword_model_paths minimal API failed: {exc}")

        try:
            self._oww_model = Model(
                wakeword_models=[str(model_path)],
            )
            print(
                f"[WakeWord] Loaded wake-word model: {model_path} (wakeword_models minimal).",
                flush=True,
            )
            return
        except Exception as exc:
            errors.append(f"wakeword_models minimal API failed: {exc}")

        print(
            "[WakeWord] Failed loading model with available APIs; "
            f"details: {' | '.join(errors)}. Falling back to keyboard PTT.",
            flush=True,
        )
        self._oww_model = None

    def _start_keyboard_thread(self) -> None:
        """Start a daemon thread that sets PTT event when Enter is pressed."""
        if self._stdin_thread and self._stdin_thread.is_alive():
            return

        self._stdin_thread = threading.Thread(
            target=self._keyboard_listener_loop,
            daemon=True,
            name="wakeword-stdin-listener",
        )
        self._stdin_thread.start()

    def _keyboard_listener_loop(self) -> None:
        """Block on stdin and set PTT event whenever user hits Enter."""
        while not self._shutdown_event.is_set():
            try:
                _ = input()
            except EOFError:
                # Non-interactive stdin; keyboard PTT is unavailable.
                self._keyboard_unavailable_event.set()
                return
            except Exception:
                self._keyboard_unavailable_event.set()
                return

            if self._shutdown_event.is_set():
                return

            self._ptt_event.set()

    def interrupt(self) -> None:
        """Interrupt listen loop promptly (safe from other threads)."""
        self._interrupt_event.set()

    def clear_interrupt(self) -> None:
        """Clear pending interrupt signal."""
        self._interrupt_event.clear()

    def shutdown(self) -> None:
        """Signal shutdown and release waiting loops."""
        self._shutdown_event.set()
        self._interrupt_event.set()
        self._ptt_event.set()

        if self._stdin_thread and self._stdin_thread.is_alive():
            self._stdin_thread.join(timeout=0.2)

    def _reset_oww(self) -> None:
        if self._oww_model is None:
            return
        try:
            self._oww_model.reset()
        except Exception:
            pass

    def _get_input_samplerate(self) -> int:
        try:
            info = sd.query_devices(device=self.input_device, kind="input")
            return int(info["default_samplerate"])
        except Exception:
            return 48000

    def _predict_wake_word(self, audio_chunk: np.ndarray) -> bool:
        """Return True if any model score crosses the wake-word threshold."""
        if self._oww_model is None:
            return False

        try:
            predictions = self._oww_model.predict(audio_chunk)
        except Exception:
            return False

        # Preferred path: read latest score from OpenWakeWord internal buffers.
        try:
            for model_name in self._oww_model.prediction_buffer.keys():
                values = list(self._oww_model.prediction_buffer[model_name])
                if values and float(values[-1]) >= self.threshold:
                    return True
        except Exception:
            pass

        # Fallback path: direct dictionary return from predict().
        if isinstance(predictions, dict):
            for score in predictions.values():
                try:
                    if float(score) >= self.threshold:
                        return True
                except Exception:
                    continue

        return False

    def _wait_for_ptt(self, timeout_s: float | None) -> Literal["PTT"]:
        """Wait until PTT event, interrupt, or timeout occurs."""
        start = time.monotonic()
        while True:
            if self._shutdown_event.is_set() or self._interrupt_event.is_set():
                self._interrupt_event.clear()
                self._ptt_event.clear()
                return "PTT"

            # Avoid infinite blocking if keyboard fallback is unavailable.
            if timeout_s is None and (
                not self.enable_keyboard_ptt or self._keyboard_unavailable_event.is_set()
            ):
                return "PTT"

            if self._ptt_event.is_set():
                self._ptt_event.clear()
                return "PTT"

            if timeout_s is not None and (time.monotonic() - start) >= timeout_s:
                return "PTT"

            time.sleep(0.05)

    def listen_for_trigger(self, timeout_s: float | None = None) -> Literal["WAKE", "PTT"]:
        """Listen continuously until wake word or PTT fallback is triggered."""
        self._interrupt_event.clear()
        self._ptt_event.clear()
        self._reset_oww()

        # If wake-word model is unavailable, fallback to keyboard PTT only.
        if self._oww_model is None:
            if self.enable_keyboard_ptt and not self._keyboard_unavailable_event.is_set():
                print("[WakeWord] Waiting for Enter key (PTT fallback)...", flush=True)
            else:
                print(
                    "[WakeWord] No wake-word model and keyboard PTT unavailable; "
                    "returning PTT fallback.",
                    flush=True,
                )
            return self._wait_for_ptt(timeout_s)

        native_rate = self._get_input_samplerate()
        use_resampling = native_rate != self.sample_rate

        input_rate = native_rate if use_resampling else self.sample_rate
        input_chunk_size = (
            int(self.chunk_size * (input_rate / self.sample_rate))
            if use_resampling
            else self.chunk_size
        )

        start = time.monotonic()
        if self.enable_keyboard_ptt and not self._keyboard_unavailable_event.is_set():
            print("[WakeWord] Listening for wake word (Enter = PTT fallback)...", flush=True)
        else:
            print("[WakeWord] Listening for wake word...", flush=True)

        try:
            with sd.InputStream(
                samplerate=input_rate,
                channels=1,
                dtype="int16",
                blocksize=input_chunk_size,
                device=self.input_device,
            ) as stream:
                while True:
                    if self._shutdown_event.is_set() or self._interrupt_event.is_set():
                        self._interrupt_event.clear()
                        self._ptt_event.clear()
                        return "PTT"

                    if self._ptt_event.is_set():
                        self._ptt_event.clear()
                        return "PTT"

                    if timeout_s is not None and (time.monotonic() - start) >= timeout_s:
                        return "PTT"

                    data, _ = stream.read(input_chunk_size)
                    if isinstance(data, np.ndarray):
                        # InputStream returns shape (frames, channels)
                        audio = data[:, 0].astype(np.int16).flatten()
                    else:
                        audio = np.frombuffer(data, dtype=np.int16)

                    if use_resampling:
                        audio = scipy.signal.resample(audio, self.chunk_size).astype(np.int16)

                    if self._predict_wake_word(audio):
                        self._reset_oww()
                        return "WAKE"

        except Exception as exc:
            print(f"[WakeWord] Audio stream error: {exc}", flush=True)
            print("[WakeWord] Falling back to keyboard PTT.", flush=True)
            return self._wait_for_ptt(timeout_s)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wake word detector test runner")
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds before returning PTT fallback (default: 30)",
    )
    parser.add_argument(
        "--no-keyboard-ptt",
        action="store_true",
        help="Disable Enter-key fallback thread",
    )
    parser.add_argument(
        "--model",
        default=config.WAKE_WORD_MODEL,
        help="Path to wake-word .onnx model",
    )
    parser.add_argument(
        "--assets-dir",
        default=config.WAKE_WORD_ASSETS_DIR,
        help="Directory with OpenWakeWord feature ONNX models",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.WAKE_WORD_THRESHOLD,
        help="Wake word threshold",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    detector = WakeWordDetector(
        wake_word_model=args.model,
        feature_models_dir=args.assets_dir,
        threshold=args.threshold,
        enable_keyboard_ptt=not args.no_keyboard_ptt,
    )

    try:
        result = detector.listen_for_trigger(timeout_s=args.timeout)
        print(f"[WakeWord] Trigger source: {result}")
        return 0
    finally:
        detector.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
