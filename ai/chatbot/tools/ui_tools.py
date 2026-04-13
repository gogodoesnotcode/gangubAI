"""Tools that drive frontend overlays for timers and pomodoro sessions."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from ai.frontend.bridge.local_queue import EmotionPublisher


_EMOTION_PUBLISHER = EmotionPublisher()
_ROS_EMOTION_TOPIC = "/robot_emotion"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _publish_to_ros(message_json: str) -> None:
    safe_json = message_json.replace("'", "''")
    ros_payload = "{data: '" + safe_json + "'}"

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


def _publish_overlay_event(event: str, payload: dict[str, Any], source: str) -> None:
    event_payload = json.dumps(
        {
            "event": event,
            "source": source,
            "timestamp": _utc_timestamp(),
            "payload": payload,
        },
        ensure_ascii=True,
    )
    _EMOTION_PUBLISHER.publish("happy", source=source, payload=event_payload)
    _publish_to_ros(
        json.dumps(
            {
                "emotion": "happy",
                "source": source,
                "timestamp": _utc_timestamp(),
                "payload": event_payload,
            },
            ensure_ascii=True,
        )
    )


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, remainder = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {remainder:02d}s"
    if minutes > 0:
        return f"{minutes}m {remainder:02d}s"
    return f"{remainder}s"


@tool
def timer(duration_seconds: float) -> str:
    """Start a countdown timer overlay.

    Use this tool only when the user gave an explicit duration.
    If the duration is missing, ask the user for the time instead of calling the tool.

    Args:
        duration_seconds: Required countdown length in seconds.

    Returns:
        A confirmation message for the active timer overlay.
    """

    duration_seconds = max(1.0, float(duration_seconds))
    _publish_overlay_event(
        event="timer_start",
        source="timer",
        payload={
            "duration_seconds": duration_seconds,
            "label": "timer",
        },
    )
    return f"Timer started for {_format_duration(duration_seconds)}."


@tool
def pomodoro(
    work_minutes: float = 25.0,
    break_minutes: float = 5.0,
    cycles: int = 4,
) -> str:
    """Start a pomodoro overlay.

    Use the standard 25 minute focus and 5 minute break split unless the user
    asks for different values. If the user does not specify cycles, use 4.

    Args:
        work_minutes: Focus length for each cycle.
        break_minutes: Break length between focus cycles.
        cycles: Number of focus cycles to run.

    Returns:
        A confirmation message for the active pomodoro overlay.
    """

    work_minutes = max(1.0, float(work_minutes))
    break_minutes = max(0.0, float(break_minutes))
    cycles = max(1, int(cycles))
    _publish_overlay_event(
        event="pomodoro_start",
        source="pomodoro",
        payload={
            "work_seconds": work_minutes * 60.0,
            "break_seconds": break_minutes * 60.0,
            "cycles": cycles,
            "label": "pomodoro",
        },
    )
    return (
        f"Pomodoro started with {cycles} cycles of {int(work_minutes)} minutes focus "
        f"and {int(break_minutes)} minutes break."
    )