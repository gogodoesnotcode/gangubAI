"""Configuration for the GangubAI Pygame frontend."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = Path(__file__).resolve().parent
ASSETS_ROOT = FRONTEND_ROOT / "assets"
FACE_ASSETS_ROOT = ASSETS_ROOT / "faces"
LEGACY_FACE_SPLIT_ROOT = FRONTEND_ROOT / "gangubai_face_split"

DEFAULT_WINDOW_SIZE: tuple[int, int] = (800, 480)
DEFAULT_FACE_CANVAS_SIZE: tuple[int, int] = (420, 320)
FRONTEND_FPS = 60

DEFAULT_TIMER_SECONDS = 60.0
DEFAULT_LISTEN_HOST = "127.0.0.1"
DEFAULT_LISTEN_PORT = 8765
DEFAULT_LISTEN_TIMEOUT_SECONDS = 0.0
EMOTION_STALE_THRESHOLD_SECONDS = 10.0

EMOTION_SEQUENCE: tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "curious",
    "excited",
    "confused",
    "thinking",
)

EMOTION_FRAME_DURATIONS_MS: dict[str, int] = {
    "neutral": 180,
    "happy": 140,
    "sad": 220,
    "angry": 160,
    "curious": 180,
    "excited": 110,
    "confused": 180,
    "thinking": 150,
}

EMOTION_THEME: dict[str, dict[str, tuple[int, int, int]]] = {
    "neutral": {
        "accent": (101, 174, 255),
        "glow": (120, 196, 255),
        "backdrop_top": (10, 16, 30),
        "backdrop_bottom": (22, 30, 50),
    },
    "happy": {
        "accent": (255, 191, 90),
        "glow": (255, 224, 144),
        "backdrop_top": (42, 22, 12),
        "backdrop_bottom": (78, 46, 18),
    },
    "sad": {
        "accent": (106, 154, 255),
        "glow": (163, 190, 255),
        "backdrop_top": (8, 16, 34),
        "backdrop_bottom": (16, 27, 54),
    },
    "angry": {
        "accent": (255, 103, 96),
        "glow": (255, 157, 145),
        "backdrop_top": (42, 11, 14),
        "backdrop_bottom": (74, 18, 26),
    },
    "curious": {
        "accent": (120, 255, 214),
        "glow": (168, 255, 236),
        "backdrop_top": (8, 28, 34),
        "backdrop_bottom": (14, 52, 52),
    },
    "excited": {
        "accent": (255, 136, 214),
        "glow": (255, 187, 238),
        "backdrop_top": (36, 14, 38),
        "backdrop_bottom": (72, 24, 78),
    },
    "confused": {
        "accent": (255, 202, 106),
        "glow": (255, 231, 153),
        "backdrop_top": (30, 24, 10),
        "backdrop_bottom": (56, 44, 18),
    },
    "thinking": {
        "accent": (159, 128, 255),
        "glow": (198, 184, 255),
        "backdrop_top": (18, 14, 42),
        "backdrop_bottom": (40, 30, 78),
    },
}

HELP_LINES: tuple[str, ...] = (
    "1-8 emotions   SPACE timer   R reset   F fullscreen   ESC quit",
    "Voice app updates this face over localhost UDP on port 8765.",
)
