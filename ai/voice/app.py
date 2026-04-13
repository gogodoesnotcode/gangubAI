"""Full voice-loop app for GangubAI.

Phase 5 integration:
- Wake word (or PTT mode)
- Recorder
- Whisper transcription
- LangGraph chatbot
- Emotion emission hook
- Piper TTS playback via sentence queue

Usage:
    python3 -m ai.voice.app
    python3 -m ai.voice.app --ptt
    python3 -m ai.voice.app --max-turns 1 --no-tts
    python3 -m ai.voice.app --wav-input --wav-file ~/gangubai_input.wav --no-tts
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from ai.frontend.bridge.local_queue import EmotionPublisher
from ai.voice import config
from ai.voice.recorder import record_adaptive, record_ptt
from ai.voice.transcriber import transcribe
from ai.voice.tts import TTSWorker
from ai.voice.wake_word import WakeWordDetector


RAG_LOOKUP_MESSAGE = "Let me look into your slides and get back to you."
_EMOTION_PUBLISHER = EmotionPublisher()
_ROS_EMOTION_TOPIC = "/robot_emotion"


def _extract_duration_seconds(user_text: str, default_s: float) -> float:
    """Extract duration in seconds from user text, with safe bounds."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", user_text.lower())
    if not match:
        return default_s
    try:
        return max(0.2, min(10.0, float(match.group(1))))
    except ValueError:
        return default_s


def _ros_topic_has_subscribers(topic: str) -> bool:
    """Return True when at least one subscriber is present for the ROS topic."""
    try:
        result = subprocess.run(
            ["ros2", "topic", "info", topic],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return False

    if result.returncode != 0:
        return False

    match = re.search(r"Subscription count:\s*(\d+)", result.stdout)
    return bool(match and int(match.group(1)) > 0)


def _publish_once(topic: str, message_type: str, payload: str) -> None:
    subprocess.Popen(
        ["ros2", "topic", "pub", "--once", topic, message_type, payload],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _direct_move(direction: str, duration: float) -> str:
    """Publish motor command directly, repeatedly, then auto-stop."""
    if not _ros_topic_has_subscribers("/motor_command"):
        return (
            "Error: No subscribers on /motor_command. Start ROS launch first: "
            "ros2 launch gangubai_control motor_control.launch.py simulate:=true"
        )

    if direction == "stop":
        _publish_once("/motor_command", "std_msgs/msg/String", "{data: 'stop'}")
        return "Robot stopped."

    duration = max(0.2, float(duration))
    payload = direction
    publisher_script = f"""
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

direction = {payload!r}
duration_s = {duration!r}


class PublisherNode(Node):
    def __init__(self):
        super().__init__('direct_motor_publisher')
        self.publisher = self.create_publisher(String, '/motor_command', 10)

    def publish_direction(self, value: str) -> None:
        message = String()
        message.data = value
        self.publisher.publish(message)


rclpy.init()
node = PublisherNode()
deadline = time.time() + duration_s

try:
    while time.time() < deadline:
        node.publish_direction(direction)
        time.sleep(0.1)
finally:
    node.publish_direction('stop')
    rclpy.shutdown()
"""
    subprocess.Popen(
        ["/usr/bin/python3", "-c", publisher_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return f"Robot moving {direction} for {duration:.1f}s then auto-stopping."


def _direct_wander(action: str) -> str:
    """Publish wander mode command directly."""
    if not _ros_topic_has_subscribers("/wander_mode"):
        return (
            "Error: No subscribers on /wander_mode. Start ROS launch with wander controller first."
        )

    _publish_once("/wander_mode", "std_msgs/msg/String", f"{{data: '{action}'}}")
    if action == "start":
        return "Wander mode enabled. Robot will move autonomously until stopped."
    return "Wander mode stopped. Robot returned to idle."


def _handle_direct_robot_intent(user_text: str) -> tuple[str, str] | None:
    """Handle movement/wander intents directly (bypassing LLM/tool routing)."""
    text = user_text.strip().lower()
    if not text:
        return None

    # Wander mode intents
    if "wander" in text:
        action = "stop" if any(k in text for k in ("stop", "off", "disable")) else "start"
        result = _direct_wander(action)
        emotion = "excited" if action == "start" else "neutral"
        return str(result), emotion

    movement_cues = (
        "move", "go", "forward", "backward", "back", "left", "right", "stop", "spin", "rotate", "360"
    )
    if not any(cue in text for cue in movement_cues):
        return None

    direction = None
    if any(k in text for k in ("spin", "rotate", "360", "turn around")):
        direction = "360"
    elif "forward" in text or "ahead" in text:
        direction = "forward"
    elif "backward" in text or "reverse" in text or re.search(r"\bback\b", text):
        direction = "backward"
    elif re.search(r"\bleft\b", text):
        direction = "left"
    elif re.search(r"\bright\b", text):
        direction = "right"
    elif re.search(r"\bstop\b", text):
        direction = "stop"

    if direction is None:
        return None

    default_duration = 10.0 if direction == "360" else 2.0
    duration = 0.0 if direction == "stop" else _extract_duration_seconds(text, default_duration)
    result = _direct_move(direction, duration)
    emotion = "excited" if direction != "stop" else "neutral"
    return str(result), emotion


def _safe_input(prompt: str) -> bool:
    """Read user input when stdin is interactive; return False on EOF."""
    try:
        input(prompt)
        return True
    except EOFError:
        return False


def _coerce_message_content_to_text(content: Any) -> str:
    """Extract text from LangChain message content variants."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        fragments: list[str] = []
        for part in content:
            if isinstance(part, str) and part.strip():
                fragments.append(part.strip())
                continue
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    fragments.append(text.strip())
        return " ".join(fragments).strip()

    return str(content).strip()


def _extract_reply_and_emotion(final_state: dict[str, Any] | None) -> tuple[str, str]:
    """Return assistant text and emotion from final graph state."""
    if not final_state:
        return "", "neutral"

    emotion = str(final_state.get("current_emotion", "neutral"))
    messages = final_state.get("messages", [])
    if not messages:
        return "", emotion

    last_message = messages[-1]
    content = getattr(last_message, "content", "")
    text = _coerce_message_content_to_text(content)
    return text, emotion


def _latest_ai_has_retrieve_tool_call(state: dict[str, Any]) -> bool:
    """Detect whether the latest AI message requested retrieve_context."""
    messages = state.get("messages", [])
    if not messages:
        return False

    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue

        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            if isinstance(call, dict) and call.get("name") == "retrieve_context":
                return True

        # Fallback for providers that keep tool calls in additional_kwargs.
        additional = getattr(message, "additional_kwargs", {})
        raw_calls = additional.get("tool_calls", [])
        for call in raw_calls:
            function_block = call.get("function", {}) if isinstance(call, dict) else {}
            if function_block.get("name") == "retrieve_context":
                return True

        return False

    return False


def emit_emotion(
    emotion: str,
    emotion_events: "queue.Queue[str] | None" = None,
    event: str = "emotion",
) -> None:
    """Emit an emotion event for UI integration."""
    if emotion_events is not None:
        emotion_events.put(emotion)
    normalized = str(emotion).strip().lower() or "neutral"
    payload = json.dumps({"event": event}, ensure_ascii=True)
    _EMOTION_PUBLISHER.publish(normalized, source="voice", payload=payload)
    _publish_emotion_to_ros(normalized, source="voice", payload=payload)
    print(f"[Emotion] {normalized} ({event})", flush=True)


def _publish_emotion_to_ros(emotion: str, source: str = "voice", payload: str = "") -> None:
    """Best-effort publish of emotion payload to ROS for UI subscribers."""
    message_json = json.dumps(
        {
            "emotion": emotion,
            "source": source,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "payload": payload,
        },
        ensure_ascii=True,
    )
    ros_payload = "{data: '" + message_json + "'}"

    try:
        subprocess.Popen(
            [
                "ros2",
                "topic",
                "pub",
                "--once",
                _ROS_EMOTION_TOPIC,
                "std_msgs/msg/String",
                ros_payload,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return


def _publish_wander_stop() -> bool:
    """Request ROS wander controller to return to idle (safe/idempotent)."""
    try:
        subprocess.Popen(
            [
                "ros2", "topic", "pub", "--once",
                "/wander_mode", "std_msgs/msg/String",
                "{data: 'stop'}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


def _build_chatbot(history_db: str):
    """Compile chatbot graph with in-memory checkpointing and return graph."""
    # Lazy import avoids model initialization when only parsing CLI args/help.
    from ai.chatbot.graph import graph

    checkpointer = MemorySaver()
    chatbot = graph.compile(checkpointer=checkpointer)
    return chatbot


def _chatbot_turn(
    chatbot,
    user_text: str,
    thread_id: str,
    on_retrieve_context_start: Callable[[], None] | None = None,
) -> tuple[str, str]:
    """Run one chatbot turn and return (assistant_text, emotion)."""
    run_config = {"configurable": {"thread_id": thread_id}}
    final_state: dict[str, Any] | None = None
    retrieve_notice_sent = False

    for event in chatbot.stream(
        {"messages": [HumanMessage(content=user_text)]},
        config=run_config,
        stream_mode="values",
    ):
        if not retrieve_notice_sent and _latest_ai_has_retrieve_tool_call(event):
            retrieve_notice_sent = True
            if on_retrieve_context_start is not None:
                on_retrieve_context_start()

        final_state = event

    return _extract_reply_and_emotion(final_state)


def _record_once_ptt(filename: str) -> str | None:
    """Record one utterance in PTT mode (Enter to start/stop)."""
    print("[PTT] Press Enter to start recording.", flush=True)
    _safe_input("")

    stop_event = threading.Event()

    def _stopper() -> None:
        interactive = _safe_input("[PTT] Recording... press Enter to stop.\n")
        if not interactive:
            # Non-interactive stdin: auto-stop after max configured duration.
            time.sleep(config.MAX_RECORD_TIME_S)
        stop_event.set()

    threading.Thread(target=_stopper, daemon=True, name="ptt-stop-listener").start()
    return record_ptt(stop_event=stop_event, filename=filename)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GangubAI full voice-loop app")

    parser.add_argument(
        "--ptt",
        action="store_true",
        help="Disable wake-word listening and use push-to-talk mode only",
    )

    # ── WAV-input mode ──────────────────────────────────────────────────────
    parser.add_argument(
        "--wav-input",
        action="store_true",
        help="Skip all recording; feed a pre-recorded WAV file directly into the pipeline",
    )
    parser.add_argument(
        "--wav-file",
        default="~/gangubai_input.wav",
        help="Path to WAV file used in --wav-input mode (default: ~/gangubai_input.wav)",
    )
    # ────────────────────────────────────────────────────────────────────────

    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Stop after N turns (0 = run forever)",
    )
    parser.add_argument(
        "--thread-id",
        default="voice-session",
        help="LangGraph thread_id used for chat memory",
    )
    parser.add_argument(
        "--history-db",
        default="gangubai_voice_history.db",
        help="SQLite file used by LangGraph checkpointer",
    )
    parser.add_argument(
        "--record-file",
        default=config.RECORDING_OUTPUT_FILE,
        help="Temporary WAV path used by recorder",
    )

    parser.add_argument(
        "--listen-timeout",
        type=float,
        default=None,
        help="Wake listen timeout in seconds (default: no timeout)",
    )
    parser.add_argument(
        "--wake-model",
        default=config.WAKE_WORD_MODEL,
        help="Path to wake-word ONNX model",
    )
    parser.add_argument(
        "--wake-assets-dir",
        default=config.WAKE_WORD_ASSETS_DIR,
        help="Directory containing OpenWakeWord feature ONNX models",
    )
    parser.add_argument(
        "--wake-threshold",
        type=float,
        default=config.WAKE_WORD_THRESHOLD,
        help="Wake-word confidence threshold",
    )
    parser.add_argument(
        "--no-keyboard-ptt",
        action="store_true",
        help="Disable Enter-key fallback in wake-word mode",
    )

    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Run full loop but skip spoken output",
    )
    parser.add_argument(
        "--tts-timeout",
        type=float,
        default=180.0,
        help="Max seconds to wait for TTS queue drain per turn",
    )

    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    # Resolve wav-file path early so we can fail fast with a clear message.
    wav_input_path: Path | None = None
    if args.wav_input:
        wav_input_path = Path(args.wav_file).expanduser().resolve()
        if not wav_input_path.exists():
            print(
                f"[Error] --wav-input specified but file not found: {wav_input_path}",
                flush=True,
            )
            return 1

    print("=" * 64)
    print("GangubAI Voice App")
    print("=" * 64)

    chatbot = None
    detector: WakeWordDetector | None = None
    tts_worker: TTSWorker | None = None
    emotion_events: "queue.Queue[str]" = queue.Queue()

    try:
        print("[Init] Chatbot will initialize on first non-robot query.", flush=True)

        if not args.no_tts:
            tts_worker = TTSWorker()
            tts_worker.start()
            print("[Init] TTS worker started.", flush=True)
        else:
            print("[Init] TTS disabled (--no-tts).", flush=True)

        if args.wav_input:
            print(f"[Init] WAV-input mode — using file: {wav_input_path}", flush=True)
        elif not args.ptt:
            detector = WakeWordDetector(
                wake_word_model=args.wake_model,
                feature_models_dir=args.wake_assets_dir,
                threshold=args.wake_threshold,
                enable_keyboard_ptt=not args.no_keyboard_ptt,
            )
            print("[Init] Wake-word detector ready.", flush=True)
        else:
            print("[Init] Running in PTT-only mode.", flush=True)

        turns = 0
        while True:
            if args.max_turns > 0 and turns >= args.max_turns:
                print(f"[Main] Reached max turns: {args.max_turns}", flush=True)
                break

            # ── Determine WAV path for this turn ──────────────────────────
            if args.wav_input:
                # Use the provided file directly; exit after one turn.
                wav_path = str(wav_input_path)
                print(f"[WAV] Using pre-recorded file: {wav_path}", flush=True)
            elif args.ptt:
                wav_path = _record_once_ptt(filename=args.record_file)
            else:
                assert detector is not None
                trigger = detector.listen_for_trigger(timeout_s=args.listen_timeout)
                print(f"[WakeWord] Trigger source: {trigger}", flush=True)

                # Always force wander stop when user calls the bot.
                if _publish_wander_stop():
                    print("[WakeWord] Wander stop requested.", flush=True)

                wav_path = record_adaptive(filename=args.record_file)
            # ──────────────────────────────────────────────────────────────

            if not wav_path:
                print("[Main] No audio captured; waiting for next turn.", flush=True)
                if args.wav_input:
                    break  # Nothing to retry in wav-input mode.
                continue

            try:
                user_text = transcribe(wav_path).strip()
            except Exception as exc:
                print(f"[STT] Transcription error: {exc}", flush=True)
                if args.wav_input:
                    return 1
                continue

            if not user_text:
                print("[STT] Empty transcription; waiting for next turn.", flush=True)
                if args.wav_input:
                    return 1
                continue

            print(f"[User] {user_text}", flush=True)

            direct_robot = _handle_direct_robot_intent(user_text)
            if direct_robot is not None:
                reply_text, emotion = direct_robot
                print(f"[Assistant] {reply_text}", flush=True)

                if tts_worker is not None:
                    emit_emotion(emotion, emotion_events=emotion_events, event="speech_start")
                    queued = tts_worker.enqueue(reply_text)
                    if queued == 0:
                        print("[TTS] Nothing queued (empty/unsupported text).", flush=True)
                    else:
                        completed = tts_worker.wait_until_done(timeout_s=args.tts_timeout)
                        if not completed:
                            print(
                                f"[TTS] Timeout after {args.tts_timeout:.1f}s; continuing.",
                                flush=True,
                            )
                        if tts_worker.last_error is not None:
                            print(f"[TTS] Last error: {tts_worker.last_error}", flush=True)
                    emit_emotion("happy", emotion_events=emotion_events, event="speech_end")
                else:
                    emit_emotion("happy", emotion_events=emotion_events, event="speech_end")

                turns += 1
                if args.wav_input:
                    break  # Single-shot: done.
                continue

            def _on_retrieve_context_start() -> None:
                print(f"[Assistant] {RAG_LOOKUP_MESSAGE}", flush=True)
                if tts_worker is not None:
                    emit_emotion("thinking", emotion_events=emotion_events, event="speech_start")
                    tts_worker.enqueue_sentence(RAG_LOOKUP_MESSAGE)

            if chatbot is None:
                chatbot = _build_chatbot(history_db=args.history_db)
                print("[Init] Chat memory: in-memory (use --history-db for persistence)", flush=True)

            try:
                reply_text, emotion = _chatbot_turn(
                    chatbot=chatbot,
                    user_text=user_text,
                    thread_id=args.thread_id,
                    on_retrieve_context_start=_on_retrieve_context_start,
                )
            except Exception as exc:
                print(f"[Chatbot] Error: {exc}", flush=True)
                if args.wav_input:
                    return 1
                continue

            if not reply_text:
                print("[Chatbot] Empty response; waiting for next turn.", flush=True)
                if args.wav_input:
                    return 1
                continue

            print(f"[Assistant] {reply_text}", flush=True)

            if tts_worker is not None:
                emit_emotion(emotion, emotion_events=emotion_events, event="speech_start")
                queued = tts_worker.enqueue(reply_text)
                if queued == 0:
                    print("[TTS] Nothing queued (empty/unsupported text).", flush=True)
                else:
                    completed = tts_worker.wait_until_done(timeout_s=args.tts_timeout)
                    if not completed:
                        print(
                            f"[TTS] Timeout after {args.tts_timeout:.1f}s; continuing.",
                            flush=True,
                        )
                    if tts_worker.last_error is not None:
                        print(f"[TTS] Last error: {tts_worker.last_error}", flush=True)
                emit_emotion("happy", emotion_events=emotion_events, event="speech_end")
            else:
                emit_emotion("happy", emotion_events=emotion_events, event="speech_end")

            turns += 1
            if args.wav_input:
                break  # Single-shot: done.

        return 0

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user. Shutting down...", flush=True)
        return 0
    except Exception as exc:
        print(f"[Main] Fatal error: {exc}", flush=True)
        return 1
    finally:
        if detector is not None:
            detector.shutdown()
        if tts_worker is not None:
            tts_worker.stop(wait=True)


if __name__ == "__main__":
    raise SystemExit(main())
