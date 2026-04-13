"""Run frontend + backend together so the display can stay in focus."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from pathlib import Path

from ai.frontend.app import FrontendApp
from ai.frontend.config import ASSETS_ROOT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GangubAI frontend and backend voice loop together",
    )
    parser.add_argument("--fullscreen", action="store_true", help="Start frontend in fullscreen")
    parser.add_argument("--assets-dir", default="", help="Optional frontend assets directory override")
    parser.add_argument("--listen-host", default="127.0.0.1", help="Frontend UDP listen host")
    parser.add_argument("--listen-port", type=int, default=8765, help="Frontend UDP listen port")
    parser.add_argument("--ros-bridge", action="store_true", help="Enable ROS emotion subscription in frontend")
    parser.add_argument("--ros-emotion-topic", default="/robot_emotion", help="ROS emotion topic name")
    parser.add_argument(
        "--backend-mode",
        choices=("wakeword", "ptt", "wav"),
        default="wav",
        help="Backend input mode. Use wav to run a pre-recorded input file.",
    )
    parser.add_argument(
        "--backend-wav-file",
        default=str((Path(__file__).resolve().parents[1] / "gangubai_input.wav")),
        help="WAV path used when --backend-mode wav",
    )
    parser.add_argument(
        "--backend-log",
        default="voice_backend.log",
        help="File path for backend logs",
    )
    parser.add_argument("--no-tts", action="store_true", help="Run backend voice loop without TTS")
    return parser


def _build_backend_cmd(mode: str, no_tts: bool, wav_file: str) -> list[str]:
    app_module = "ai.voice.app" if mode == "wav" else "ai.voice.app"
    cmd = ["python3", "-m", app_module]

    if mode == "ptt":
        cmd.append("--ptt")
    elif mode == "wav":
        cmd.extend(["--wav-input", "--wav-file", wav_file])
    else:
        # Keep backend fully hands-free so terminal focus is not required.
        cmd.append("--no-keyboard-ptt")

    if no_tts:
        cmd.append("--no-tts")

    return cmd


def _stream_backend_output(proc: subprocess.Popen, log_file) -> None:
    """Mirror backend stdout/stderr to both log file and launcher terminal."""
    if proc.stdout is None:
        return

    for line in proc.stdout:
        log_file.write(line)
        log_file.flush()
        sys.stdout.write(f"[Backend] {line}")
        sys.stdout.flush()


def main() -> int:
    args = _build_parser().parse_args()

    backend_cmd = _build_backend_cmd(args.backend_mode, args.no_tts, args.backend_wav_file)
    log_path = Path(args.backend_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    backend_proc = None
    backend_log_thread: threading.Thread | None = None
    log_file = None

    try:
        log_file = log_path.open("a", encoding="utf-8")
        backend_proc = subprocess.Popen(
            backend_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        backend_log_thread = threading.Thread(
            target=_stream_backend_output,
            args=(backend_proc, log_file),
            daemon=True,
            name="backend-log-tee",
        )
        backend_log_thread.start()

        print(f"[Launcher] Backend started (pid={backend_proc.pid}).", flush=True)
        print(f"[Launcher] Logging backend output to {log_path}", flush=True)

        assets_dir = args.assets_dir if args.assets_dir else str(ASSETS_ROOT)
        app = FrontendApp(
            fullscreen=args.fullscreen,
            listen_host=args.listen_host,
            listen_port=args.listen_port,
            assets_dir=assets_dir,
            ros_bridge=args.ros_bridge,
            ros_emotion_topic=args.ros_emotion_topic,
        )
        return app.run()
    finally:
        if backend_proc is not None and backend_proc.poll() is None:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                backend_proc.kill()
                backend_proc.wait(timeout=4)

        if backend_log_thread is not None and backend_log_thread.is_alive():
            backend_log_thread.join(timeout=1)

        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
