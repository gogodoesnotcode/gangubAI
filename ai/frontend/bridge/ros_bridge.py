"""ROS 2 emotion bridge for future robot integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from ai.frontend.state import Emotion, EmotionEvent, normalize_emotion

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover - optional ROS dependency
    rclpy = None
    Node = object  # type: ignore[assignment]
    String = object  # type: ignore[assignment]


@dataclass(slots=True)
class RosEmotionConfig:
    topic: str = "/robot_emotion"
    debounce_seconds: float = 0.15


class RosEmotionBridge:
    """Subscribe to a ROS emotion topic and relay updates to a callback."""

    def __init__(self, on_emotion: Callable[[EmotionEvent], None], config: RosEmotionConfig | None = None) -> None:
        if rclpy is None:
            raise RuntimeError("rclpy is not available in this environment")
        self.on_emotion = on_emotion
        self.config = config or RosEmotionConfig()
        self._node = None
        self._executor = None
        self._thread = None
        self._last_event_s = 0.0

    def start(self) -> None:
        if self._node is not None:
            return
        rclpy.init(args=None)
        self._node = rclpy.create_node("gangubai_emotion_bridge")
        self._node.create_subscription(String, self.config.topic, self._handle_message, 10)

        from rclpy.executors import SingleThreadedExecutor
        import threading

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True, name="ros-emotion-bridge")
        self._thread.start()

    def _handle_message(self, message: String) -> None:  # type: ignore[valid-type]
        import json
        import time

        now_s = time.time()
        if now_s - self._last_event_s < self.config.debounce_seconds:
            return
        self._last_event_s = now_s

        raw = getattr(message, "data", "")
        try:
            decoded = json.loads(raw)
            emotion = normalize_emotion(decoded.get("emotion"))
            source = str(decoded.get("source", "ros"))
            timestamp = _parse_timestamp(str(decoded.get("timestamp", "")))
            payload = str(decoded.get("payload", ""))
        except Exception:
            emotion = normalize_emotion(raw)
            source = "ros"
            timestamp = now_s
            payload = raw

        self.on_emotion(EmotionEvent(emotion=emotion, source=source, timestamp=timestamp, payload=payload))

    def stop(self) -> None:
        if self._executor is not None and self._node is not None:
            self._executor.remove_node(self._node)
        if self._node is not None:
            self._node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        self._node = None
        self._executor = None
        self._thread = None


def _parse_timestamp(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0
