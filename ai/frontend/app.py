"""Runnable Pygame frontend for the GangubAI robot face."""

from __future__ import annotations

import argparse
import json
import os
import queue
import time
from pathlib import Path

import pygame

from ai.frontend.animation import FaceAnimator, load_face_clips
from ai.frontend.bridge.local_queue import EmotionListener
from ai.frontend.bridge.ros_bridge import RosEmotionBridge, RosEmotionConfig
from ai.frontend.config import (
    ASSETS_ROOT,
    DEFAULT_FACE_CANVAS_SIZE,
    DEFAULT_LISTEN_HOST,
    DEFAULT_LISTEN_PORT,
    DEFAULT_WINDOW_SIZE,
    FRONTEND_FPS,
)
from ai.frontend.overlays import draw_pomodoro_overlay, draw_timer_overlay, load_fonts
from ai.frontend.state import Emotion, FrontendState, normalize_emotion
from ai.frontend.timers import CountdownTimer, PomodoroSession


# Disable Pygame's audio driver to avoid exclusive ALSA device access.
# This allows the backend TTS (aplay) to use the audio device simultaneously.
os.environ['SDL_AUDIODRIVER'] = 'dummy'


KEY_TO_EMOTION: dict[int, Emotion] = {
    pygame.K_1: Emotion.HAPPY,
    pygame.K_2: Emotion.HAPPY,
    pygame.K_3: Emotion.SAD,
    pygame.K_4: Emotion.ANGRY,
    pygame.K_5: Emotion.CURIOUS,
    pygame.K_6: Emotion.EXCITED,
    pygame.K_7: Emotion.CONFUSED,
    pygame.K_8: Emotion.THINKING,
}


class FrontendApp:
    """Owns the Pygame window, animation state, and event loop."""

    def __init__(
        self,
        fullscreen: bool = False,
        listen_host: str = DEFAULT_LISTEN_HOST,
        listen_port: int = DEFAULT_LISTEN_PORT,
        assets_dir: str | Path = ASSETS_ROOT,
        ros_bridge: bool = False,
        ros_emotion_topic: str = "/robot_emotion",
    ) -> None:
        pygame.init()
        pygame.display.set_caption("GangubAI Face")
        pygame.key.set_repeat(250, 40)

        self.fonts = load_fonts()
        self.state = FrontendState(fullscreen=fullscreen)
        self.clock = pygame.time.Clock()
        self.windowed_size = DEFAULT_WINDOW_SIZE
        self.screen = self._create_display(fullscreen)
        self.canvas_size = DEFAULT_FACE_CANVAS_SIZE
        self.assets_dir = Path(assets_dir)
        self.animator = FaceAnimator(load_face_clips(self.assets_dir, self.canvas_size))
        self.animator.set_emotion(Emotion.HAPPY)
        self.animator.set_playing(False)
        self.listener = self._create_listener(listen_host, listen_port)
        self._pending_events: "queue.Queue[tuple[Emotion, str, str]]" = queue.Queue()
        self.ros_bridge = self._create_ros_bridge(enabled=ros_bridge, topic=ros_emotion_topic)
        self.start_time_s = time.time()
        self.timer_overlay: CountdownTimer | None = None
        self.pomodoro_overlay: PomodoroSession | None = None

    def _create_listener(self, host: str, port: int) -> EmotionListener | None:
        try:
            return EmotionListener(host=host, port=port)
        except OSError:
            return None

    def _create_ros_bridge(self, enabled: bool, topic: str) -> RosEmotionBridge | None:
        if not enabled:
            return None

        try:
            bridge = RosEmotionBridge(
                on_emotion=lambda event: self._pending_events.put(
                    (event.emotion, event.source, event.payload)
                ),
                config=RosEmotionConfig(topic=topic),
            )
            bridge.start()
            return bridge
        except Exception:
            return None

    def _create_display(self, fullscreen: bool) -> pygame.Surface:
        flags = pygame.FULLSCREEN if fullscreen else 0
        size = pygame.display.get_desktop_sizes()[0] if fullscreen else self.windowed_size
        return pygame.display.set_mode(size, flags)

    def _toggle_fullscreen(self) -> None:
        self.state.fullscreen = not self.state.fullscreen
        self.screen = self._create_display(self.state.fullscreen)

    def _apply_emotion(self, emotion: str | Emotion, source: str, payload: str = "") -> None:
        normalized = normalize_emotion(emotion)
        self.state.emotion = normalized
        self.state.source = source
        self.state.last_update_s = time.time()
        self.state.last_payload = payload
        self.animator.set_emotion(normalized)

    def _set_idle_happy(self, source: str = "idle") -> None:
        self.state.speaking = False
        self._apply_emotion(Emotion.HAPPY, source=source)
        self.animator.set_playing(False)

    def _handle_bridge_event(self, emotion: Emotion, source: str, payload: str) -> None:
        event_type = "emotion"
        event_data: dict[str, object] = {}
        if payload:
            try:
                decoded = json.loads(payload)
                if isinstance(decoded, dict):
                    event_type = str(decoded.get("event", "emotion"))
                    raw_payload = decoded.get("payload", {})
                    if isinstance(raw_payload, dict):
                        event_data = raw_payload
            except Exception:
                event_type = payload.strip() or "emotion"

        if event_type == "timer_start":
            duration_seconds = float(event_data.get("duration_seconds", 0.0) or 0.0)
            if duration_seconds > 0.0:
                self.timer_overlay = CountdownTimer(duration_seconds)
                self.timer_overlay.start()
                self.pomodoro_overlay = None
            return

        if event_type == "pomodoro_start":
            work_seconds = float(event_data.get("work_seconds", 25.0 * 60.0) or 25.0 * 60.0)
            break_seconds = float(event_data.get("break_seconds", 5.0 * 60.0) or 5.0 * 60.0)
            cycles = int(event_data.get("cycles", 4) or 4)
            self.pomodoro_overlay = PomodoroSession(
                work_seconds=work_seconds,
                break_seconds=break_seconds,
                cycles=cycles,
            )
            self.pomodoro_overlay.start()
            self.timer_overlay = None
            return

        if event_type in {"timer_end", "pomodoro_end", "overlay_clear"}:
            self.timer_overlay = None
            self.pomodoro_overlay = None
            return

        if event_type == "speech_start":
            self.state.speaking = True
            self._apply_emotion(emotion, source=source, payload=payload)
            self.animator.set_playing(True)
            return

        if event_type == "speech_end":
            self._set_idle_happy(source=source)
            return

        # While speaking, allow emotion updates; otherwise stay idle happy.
        if self.state.speaking:
            self._apply_emotion(emotion, source=source, payload=payload)

    def _handle_key(self, event: pygame.event.Event) -> None:
        if event.key in KEY_TO_EMOTION:
            self._apply_emotion(KEY_TO_EMOTION[event.key], source="keyboard")
            return

        if event.key == pygame.K_f:
            self._toggle_fullscreen()
            return

        if event.key == pygame.K_d:
            self.state.debug_visible = not self.state.debug_visible
            return

        if event.key in {pygame.K_ESCAPE, pygame.K_q}:
            self.state.running = False

    def _drain_listener(self) -> None:
        if self.listener is None:
            return
        event = self.listener.poll_latest()
        if event is None:
            return
        if event.timestamp and (time.time() - event.timestamp) > 60.0:
            return
        self._handle_bridge_event(event.emotion, source=event.source, payload=event.payload)

    def _drain_pending_events(self) -> None:
        latest: tuple[Emotion, str, str] | None = None
        while True:
            try:
                latest = self._pending_events.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            return

        emotion, source, payload = latest
        self._handle_bridge_event(emotion, source=source, payload=payload)

    def run(self) -> int:
        self._set_idle_happy(source="startup")
        while self.state.running:
            dt_ms = self.clock.tick(FRONTEND_FPS)
            dt_s = dt_ms / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event)

            self._drain_listener()
            self._drain_pending_events()
            self.animator.update(dt_ms)

            if self.timer_overlay is not None:
                self.timer_overlay.update(dt_s)
                if self.timer_overlay.remaining_s <= 0.0:
                    self.timer_overlay = None

            if self.pomodoro_overlay is not None:
                self.pomodoro_overlay.update(dt_s)
                if self.pomodoro_overlay.phase == "done":
                    self.pomodoro_overlay = None

            self.screen.fill((0, 0, 0))
            self._draw_face()
            self._draw_overlay()
            pygame.display.flip()

        self.shutdown()
        return 0

    def _draw_face(self) -> None:
        frame = self.animator.current_frame()
        screen_rect = self.screen.get_rect()
        scaled = pygame.transform.smoothscale(frame, screen_rect.size)
        self.screen.blit(scaled, (0, 0))

    def _draw_overlay(self) -> None:
        accent = (106, 199, 255)
        if self.timer_overlay is not None:
            draw_timer_overlay(self.screen, self.fonts, self.timer_overlay, accent)
            return

        if self.pomodoro_overlay is not None:
            draw_pomodoro_overlay(self.screen, self.fonts, self.pomodoro_overlay, accent)

    def shutdown(self) -> None:
        if self.listener is not None:
            self.listener.close()
        if self.ros_bridge is not None:
            self.ros_bridge.stop()
        pygame.quit()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GangubAI Pygame frontend")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    parser.add_argument("--assets-dir", default=str(ASSETS_ROOT), help="Path to the frontend assets directory")
    parser.add_argument("--listen-host", default=DEFAULT_LISTEN_HOST, help="Local host used for emotion updates")
    parser.add_argument("--listen-port", type=int, default=DEFAULT_LISTEN_PORT, help="Local port used for emotion updates")
    parser.add_argument("--ros-bridge", action="store_true", help="Subscribe to ROS emotion topic updates")
    parser.add_argument("--ros-emotion-topic", default="/robot_emotion", help="ROS topic used for emotion events")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    app = FrontendApp(
        fullscreen=args.fullscreen,
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        assets_dir=args.assets_dir,
        ros_bridge=args.ros_bridge,
        ros_emotion_topic=args.ros_emotion_topic,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
