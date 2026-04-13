"""Whisper.cpp transcription module for the GangubAI voice pipeline.

Usage:
    python3 -m ai.voice.transcriber /path/to/audio.wav

This module intentionally supports whisper.cpp only.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from ai.voice import config


_SEGMENT_RE = re.compile(r"^\[[0-9:.]+\s*-->\s*[0-9:.]+\]\s*(.*)$")


def _extract_text_from_whisper_output(stdout: str) -> str:
    """Parse whisper.cpp stdout and return a clean transcription string."""
    segments: list[str] = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _SEGMENT_RE.match(line)
        if match:
            text = match.group(1).strip()
            if text:
                segments.append(text)

    if segments:
        return " ".join(segments).strip()

    # Fallback: use the last non-empty line if segment lines were not found.
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def transcribe(
    wav_path: str,
    whisper_bin: str = config.WHISPER_CPP_BIN,
    whisper_model: str = config.WHISPER_CPP_MODEL,
    language: str = config.WHISPER_LANGUAGE,
    threads: int = config.WHISPER_CPP_THREADS,
    timeout_s: int = config.WHISPER_TIMEOUT_S,
) -> str:
    """Transcribe a WAV file using whisper.cpp.

    Returns the transcribed text.
    Raises FileNotFoundError or RuntimeError on failures.
    """
    audio_file = Path(wav_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    bin_path = Path(whisper_bin)
    if not bin_path.exists():
        raise FileNotFoundError(
            f"whisper.cpp binary not found: {bin_path}. "
            "Build whisper.cpp and update ai.voice.config.WHISPER_CPP_BIN."
        )

    model_path = Path(whisper_model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"whisper.cpp model not found: {model_path}. "
            "Download a GGML model and update ai.voice.config.WHISPER_CPP_MODEL."
        )

    cmd = [
        str(bin_path),
        "-m",
        str(model_path),
        "-l",
        language,
        "-t",
        str(threads),
        "-f",
        str(audio_file),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"whisper.cpp timed out after {timeout_s}s while transcribing {audio_file}."
        ) from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"whisper.cpp failed (exit {result.returncode}): {stderr or 'no stderr'}"
        )

    text = _extract_text_from_whisper_output(result.stdout)
    return text.strip()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe WAV audio with whisper.cpp")
    parser.add_argument("wav_path", help="Path to a WAV file")
    parser.add_argument("--bin", dest="whisper_bin", default=config.WHISPER_CPP_BIN)
    parser.add_argument("--model", dest="whisper_model", default=config.WHISPER_CPP_MODEL)
    parser.add_argument("--lang", dest="language", default=config.WHISPER_LANGUAGE)
    parser.add_argument("--threads", dest="threads", type=int, default=config.WHISPER_CPP_THREADS)
    parser.add_argument("--timeout", dest="timeout_s", type=int, default=config.WHISPER_TIMEOUT_S)
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        text = transcribe(
            wav_path=args.wav_path,
            whisper_bin=args.whisper_bin,
            whisper_model=args.whisper_model,
            language=args.language,
            threads=args.threads,
            timeout_s=args.timeout_s,
        )
    except Exception as exc:
        print(f"[Transcriber] Error: {exc}")
        return 1

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
