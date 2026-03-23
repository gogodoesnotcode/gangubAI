"""Piper TTS module for the GangubAI voice pipeline.

Phase 3 deliverables in this module:
- speak(text, interrupted_flag): synthesize and play speech with Piper.
- TTSWorker: daemon worker that drains a queue of text chunks.
- split_into_sentences(text): low-latency response chunking.

Usage:
    python3 -m ai.voice.tts
    python3 -m ai.voice.tts --text "Hello there. I am GangubAI."
    python3 -m ai.voice.tts --dry-run --self-test
"""

from __future__ import annotations

import argparse
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import scipy.signal
import sounddevice as sd

from ai.voice import config


_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_SANITIZE_RE = re.compile(r"[^\w\s,.!?:;'\-]")
_MAX_SENTENCE_CHARS = 220


def _sanitize_text_for_tts(text: str) -> str:
    """Normalize and strip unsupported characters before TTS."""
    cleaned = _SANITIZE_RE.sub("", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _chunk_long_sentence(sentence: str, max_chars: int = _MAX_SENTENCE_CHARS) -> list[str]:
    """Split very long sentence strings into word-safe chunks."""
    sentence = sentence.strip()
    if not sentence:
        return []

    if len(sentence) <= max_chars:
        return [sentence]

    words = sentence.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join(current + [word]).strip()
        if not current or len(candidate) <= max_chars:
            current.append(word)
            continue

        chunks.append(" ".join(current))
        current = [word]

    if current:
        chunks.append(" ".join(current))

    return chunks


def split_into_sentences(text: str) -> list[str]:
    """Split model output into short sentence chunks for low-latency TTS."""
    if not text or not text.strip():
        return []

    parts = [part.strip() for part in _SENTENCE_BREAK_RE.split(text) if part.strip()]
    chunks: list[str] = []
    for part in parts:
        chunks.extend(_chunk_long_sentence(part, max_chars=_MAX_SENTENCE_CHARS))
    return chunks


def _resolve_output_device_rate(output_device: str | int | None) -> int:
    """Return output device native sample rate, or a safe fallback."""
    try:
        info = sd.query_devices(device=output_device, kind="output")
        return int(info["default_samplerate"])
    except Exception:
        return 48000


def speak(
    text: str,
    interrupted_flag: threading.Event | None = None,
    *,
    piper_bin: str = config.PIPER_BIN,
    voice_model: str = config.PIPER_VOICE_MODEL,
    output_device: str | int | None = config.OUTPUT_DEVICE_NAME,
    piper_rate: int = config.PIPER_SAMPLE_RATE,
    read_chunk_bytes: int = config.TTS_READ_CHUNK_BYTES,
    stream_blocksize: int = config.TTS_STREAM_BLOCKSIZE,
    tail_silence_s: float = config.TTS_TAIL_SILENCE_S,
) -> None:
    """Synthesize and play *text* with Piper + sounddevice.

    Raises:
        FileNotFoundError: if the Piper binary or model is missing.
        RuntimeError: if Piper fails or audio playback fails.
    """
    utterance = _sanitize_text_for_tts(text)
    if not utterance:
        return

    bin_path = Path(piper_bin)
    if not bin_path.exists():
        raise FileNotFoundError(
            f"Piper binary not found: {bin_path}. "
            "Set ai.voice.config.PIPER_BIN to your piper executable path."
        )

    model_path = Path(voice_model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Piper voice model not found: {model_path}. "
            "Set ai.voice.config.PIPER_VOICE_MODEL to a valid .onnx model path."
        )

    proc = subprocess.Popen(
        [str(bin_path), "--model", str(model_path), "--output-raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        assert proc.stdin is not None
        proc.stdin.write(utterance.encode("utf-8") + b"\n")
        proc.stdin.close()

        native_rate = _resolve_output_device_rate(output_device)
        use_native_rate = False
        try:
            sd.check_output_settings(
                device=output_device,
                samplerate=piper_rate,
                channels=1,
                dtype="int16",
            )
        except Exception:
            use_native_rate = True

        playback_rate = native_rate if use_native_rate else piper_rate

        assert proc.stdout is not None
        with sd.RawOutputStream(
            samplerate=playback_rate,
            channels=1,
            dtype="int16",
            device=output_device,
            latency="high",
            blocksize=stream_blocksize,
        ) as stream:
            while True:
                if interrupted_flag and interrupted_flag.is_set():
                    break

                raw = proc.stdout.read(read_chunk_bytes)
                if not raw:
                    break

                if use_native_rate:
                    audio = np.frombuffer(raw, dtype=np.int16)
                    if audio.size == 0:
                        continue
                    resampled_samples = max(1, int(len(audio) * (native_rate / piper_rate)))
                    audio = scipy.signal.resample(audio, resampled_samples).astype(np.int16)
                    stream.write(audio.tobytes())
                else:
                    stream.write(raw)

            # Add a short tail pad to avoid clipping the final phoneme.
            if not (interrupted_flag and interrupted_flag.is_set()):
                silence_samples = max(0, int(playback_rate * tail_silence_s))
                if silence_samples > 0:
                    stream.write(np.zeros(silence_samples, dtype=np.int16).tobytes())

        if proc.poll() is None:
            proc.wait(timeout=2)

        if proc.returncode not in (0, None):
            stderr = b""
            if proc.stderr is not None:
                stderr = proc.stderr.read()
            msg = stderr.decode("utf-8", errors="ignore").strip() or "no stderr"
            raise RuntimeError(f"Piper exited with code {proc.returncode}: {msg}")

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while waiting for Piper process to exit.") from exc
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()


class TTSWorker:
    """Background TTS worker that drains a queue of text chunks."""

    def __init__(
        self,
        *,
        piper_bin: str = config.PIPER_BIN,
        voice_model: str = config.PIPER_VOICE_MODEL,
        output_device: str | int | None = config.OUTPUT_DEVICE_NAME,
        interrupted_flag: threading.Event | None = None,
    ) -> None:
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._speaking_event = threading.Event()
        self._interrupted_flag = interrupted_flag
        self._last_error: Exception | None = None

        self._piper_bin = piper_bin
        self._voice_model = voice_model
        self._output_device = output_device

    def start(self) -> None:
        """Start the daemon worker thread."""
        if self._thread and self._thread.is_alive():
            return

        self._last_error = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tts-worker")
        self._thread.start()

    def stop(self, wait: bool = True) -> None:
        """Signal worker shutdown and optionally wait for thread exit."""
        self._stop_event.set()
        self._queue.put(None)

        if wait and self._thread:
            self._thread.join(timeout=3)

    def enqueue(self, text: str) -> int:
        """Chunk and enqueue text. Returns number of queued chunks."""
        cleaned = _sanitize_text_for_tts(text)
        if not cleaned:
            return 0

        # Use larger chunks to avoid pauses from process startup between sentences.
        chunks = _chunk_long_sentence(cleaned, max_chars=config.TTS_QUEUE_CHUNK_CHARS)
        for chunk in chunks:
            self._queue.put(chunk)
        return len(chunks)

    def enqueue_sentence(self, sentence: str) -> None:
        """Enqueue one pre-split sentence chunk as-is."""
        sentence = sentence.strip()
        if sentence:
            self._queue.put(sentence)

    def clear_pending(self) -> int:
        """Drop pending queue items (does not interrupt current speech)."""
        removed = 0
        while True:
            try:
                item = self._queue.get_nowait()
                self._queue.task_done()
                if item is not None:
                    removed += 1
            except queue.Empty:
                break
        return removed

    def wait_until_done(self, timeout_s: float | None = None) -> bool:
        """Wait until queue is drained and speech is idle."""
        deadline = None if timeout_s is None else (time.monotonic() + timeout_s)
        while True:
            idle = self._queue.unfinished_tasks == 0 and not self._speaking_event.is_set()
            if idle:
                return True
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.05)

    @property
    def last_error(self) -> Exception | None:
        """Most recent worker exception, if any."""
        return self._last_error

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if item is None:
                self._queue.task_done()
                break

            self._speaking_event.set()
            try:
                speak(
                    item,
                    interrupted_flag=self._interrupted_flag,
                    piper_bin=self._piper_bin,
                    voice_model=self._voice_model,
                    output_device=self._output_device,
                )
            except Exception as exc:
                self._last_error = exc
                print(f"[TTSWorker] Error: {exc}")
            finally:
                self._speaking_event.clear()
                self._queue.task_done()


def _parse_output_device(raw: str | None) -> str | int | None:
    if raw is None or raw.strip() == "":
        return None
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    return raw


def _run_self_test() -> None:
    print("[TTS self-test] split_into_sentences() edge-case demo")
    cases = [
        "",
        "Hello!",
        "Hi there. How are you? I am fine!",
        "This has unicode: cafe, naive, bonjour. 123.",
        "This is a very long sentence " + ("word " * 120),
    ]
    for idx, case in enumerate(cases, start=1):
        chunks = split_into_sentences(case)
        print(f"Case {idx}: {len(chunks)} chunk(s)")
        for chunk in chunks:
            print(f"  - {chunk}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speak text via Piper TTS")
    parser.add_argument(
        "--text",
        default="Hello! I am GangubAI. This is a Piper text to speech test.",
        help="Text to speak",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print sentence chunks")
    parser.add_argument("--self-test", action="store_true", help="Run sentence splitting edge-case test")
    parser.add_argument("--bin", dest="piper_bin", default=config.PIPER_BIN, help="Path to piper binary")
    parser.add_argument(
        "--model",
        dest="voice_model",
        default=config.PIPER_VOICE_MODEL,
        help="Path to piper voice .onnx model",
    )
    parser.add_argument(
        "--device",
        dest="output_device",
        default=None,
        help="Output device index or name (default: system default)",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    if args.self_test:
        _run_self_test()

    chunks = split_into_sentences(args.text)
    if args.dry_run:
        print("[TTS dry-run] sentence chunks:")
        for idx, chunk in enumerate(chunks, start=1):
            print(f"{idx}. {chunk}")
        return 0

    worker = TTSWorker(
        piper_bin=args.piper_bin,
        voice_model=args.voice_model,
        output_device=_parse_output_device(args.output_device),
    )
    worker.start()
    queued = worker.enqueue(args.text)

    if queued == 0:
        print("[TTS] Nothing to speak (empty text after sanitization).")
        worker.stop(wait=True)
        return 0

    ok = worker.wait_until_done(timeout_s=60)
    worker.stop(wait=True)

    if worker.last_error is not None:
        print(f"[TTS] Failed: {worker.last_error}")
        return 1

    if not ok:
        print("[TTS] Timed out waiting for playback to finish.")
        return 1

    print("[TTS] Playback finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
