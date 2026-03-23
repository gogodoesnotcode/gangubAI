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
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver, sqlite3

from ai.voice import config
from ai.voice.recorder import record_adaptive, record_ptt
from ai.voice.transcriber import transcribe
from ai.voice.tts import TTSWorker
from ai.voice.wake_word import WakeWordDetector


RAG_LOOKUP_MESSAGE = "Let me look into your slides and get back to you."


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


def emit_emotion(emotion: str, emotion_events: "queue.Queue[str] | None" = None) -> None:
    """Emit an emotion event for future UI integration."""
    if emotion_events is not None:
        emotion_events.put(emotion)
    print(f"[Emotion] {emotion}", flush=True)


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
    """Compile chatbot graph with SQLite memory and return (graph, connection)."""
    # Lazy import avoids model initialization when only parsing CLI args/help.
    from ai.chatbot.graph import graph

    db_path = Path(history_db)
    conn = sqlite3.connect(database=str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    chatbot = graph.compile(checkpointer=checkpointer)
    return chatbot, conn


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

    print("=" * 64)
    print("GangubAI Voice App")
    print("=" * 64)

    chatbot = None
    conn: sqlite3.Connection | None = None
    detector: WakeWordDetector | None = None
    tts_worker: TTSWorker | None = None
    emotion_events: "queue.Queue[str]" = queue.Queue()

    try:
        chatbot, conn = _build_chatbot(history_db=args.history_db)
        print(f"[Init] Chat memory DB: {args.history_db}", flush=True)

        if not args.no_tts:
            tts_worker = TTSWorker()
            tts_worker.start()
            print("[Init] TTS worker started.", flush=True)
        else:
            print("[Init] TTS disabled (--no-tts).", flush=True)

        if not args.ptt:
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

            if args.ptt:
                wav_path = _record_once_ptt(filename=args.record_file)
            else:
                assert detector is not None
                trigger = detector.listen_for_trigger(timeout_s=args.listen_timeout)
                print(f"[WakeWord] Trigger source: {trigger}", flush=True)

                # Always force wander stop when user calls the bot.
                if _publish_wander_stop():
                    print("[WakeWord] Wander stop requested.", flush=True)

                wav_path = record_adaptive(filename=args.record_file)

            if not wav_path:
                print("[Main] No audio captured; waiting for next turn.", flush=True)
                continue

            try:
                user_text = transcribe(wav_path).strip()
            except Exception as exc:
                print(f"[STT] Transcription error: {exc}", flush=True)
                continue

            if not user_text:
                print("[STT] Empty transcription; waiting for next turn.", flush=True)
                continue

            print(f"[User] {user_text}", flush=True)

            def _on_retrieve_context_start() -> None:
                print(f"[Assistant] {RAG_LOOKUP_MESSAGE}", flush=True)
                emit_emotion("thinking", emotion_events=emotion_events)
                if tts_worker is not None:
                    tts_worker.enqueue_sentence(RAG_LOOKUP_MESSAGE)

            try:
                reply_text, emotion = _chatbot_turn(
                    chatbot=chatbot,
                    user_text=user_text,
                    thread_id=args.thread_id,
                    on_retrieve_context_start=_on_retrieve_context_start,
                )
            except Exception as exc:
                print(f"[Chatbot] Error: {exc}", flush=True)
                continue

            if not reply_text:
                print("[Chatbot] Empty response; waiting for next turn.", flush=True)
                continue

            print(f"[Assistant] {reply_text}", flush=True)
            emit_emotion(emotion, emotion_events=emotion_events)

            if tts_worker is not None:
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

            turns += 1

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
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
