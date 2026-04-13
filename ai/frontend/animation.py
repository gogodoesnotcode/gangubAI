"""Emotion animation loader and frame stepping logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pygame

from ai.frontend.config import DEFAULT_FACE_CANVAS_SIZE, EMOTION_FRAME_DURATIONS_MS
from ai.frontend.state import Emotion, normalize_emotion


@dataclass(slots=True)
class AnimationClip:
    """Frames and frame timing for a single emotion."""

    frames: list[pygame.Surface]
    frame_duration_ms: int


class FaceAnimator:
    """Stateful emotion animator with per-emotion clip playback."""

    def __init__(self, clips: dict[Emotion, AnimationClip]) -> None:
        self._clips = clips
        self._emotion = Emotion.HAPPY
        self._frame_index = 0
        self._elapsed_ms = 0
        self._playing = False

    @property
    def emotion(self) -> Emotion:
        return self._emotion

    def set_emotion(self, emotion: str | Emotion) -> None:
        normalized = normalize_emotion(emotion)
        if normalized == self._emotion:
            return
        self._emotion = normalized
        self._frame_index = 0
        self._elapsed_ms = 0

    def set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        if not self._playing:
            self._frame_index = 0
            self._elapsed_ms = 0

    def update(self, dt_ms: int) -> None:
        if not self._playing:
            return
        clip = self._clips[self._emotion]
        if len(clip.frames) <= 1:
            return
        self._elapsed_ms += max(0, int(dt_ms))
        while self._elapsed_ms >= clip.frame_duration_ms:
            self._elapsed_ms -= clip.frame_duration_ms
            self._frame_index = (self._frame_index + 1) % len(clip.frames)

    def current_frame(self) -> pygame.Surface:
        clip = self._clips[self._emotion]
        if not self._playing:
            return clip.frames[0]
        return clip.frames[self._frame_index % len(clip.frames)]


def load_face_clips(asset_root: Path, canvas_size: tuple[int, int] = DEFAULT_FACE_CANVAS_SIZE) -> dict[Emotion, AnimationClip]:
    """Load PNG sequences per emotion or synthesize a fallback animation."""

    clips: dict[Emotion, AnimationClip] = {}
    for emotion in Emotion:
        frames = _load_frames_for_emotion(asset_root, emotion, canvas_size)
        frame_duration_ms = EMOTION_FRAME_DURATIONS_MS.get(emotion.value, 180)
        clips[emotion] = AnimationClip(frames=frames, frame_duration_ms=frame_duration_ms)
    return clips


def _load_frames_for_emotion(asset_root: Path, emotion: Emotion, canvas_size: tuple[int, int]) -> list[pygame.Surface]:
    search_roots = [asset_root]
    try:
        from ai.frontend.config import LEGACY_FACE_SPLIT_ROOT

        if LEGACY_FACE_SPLIT_ROOT != asset_root:
            search_roots.append(LEGACY_FACE_SPLIT_ROOT)
    except Exception:
        pass

    for root in search_roots:
        for emotion_dir in _candidate_emotion_dirs(root, emotion):
            if not emotion_dir.exists():
                continue
            png_paths = sorted(emotion_dir.glob("*.png"))
            if not png_paths:
                continue

            loaded = []
            for path in png_paths:
                try:
                    image = pygame.image.load(str(path)).convert_alpha()
                    loaded.append(_scale_to_canvas(image, canvas_size))
                except Exception:
                    continue
            if loaded:
                return loaded

    return _build_generated_frames(emotion, canvas_size)


def _candidate_emotion_dirs(asset_root: Path, emotion: Emotion) -> list[Path]:
    """Return candidate emotion directories for common asset layouts."""

    candidates = [
        asset_root / "faces" / emotion.value,
        asset_root / emotion.value,
        asset_root / "ezgif-split" / emotion.value,
    ]

    if asset_root.exists():
        for child in asset_root.iterdir():
            if not child.is_dir():
                continue
            candidates.append(child / emotion.value)

    # Preserve order while removing duplicates.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _scale_to_canvas(surface: pygame.Surface, canvas_size: tuple[int, int]) -> pygame.Surface:
    if surface.get_size() == canvas_size:
        return surface
    return pygame.transform.smoothscale(surface, canvas_size)


def _build_generated_frames(emotion: Emotion, canvas_size: tuple[int, int], frame_count: int = 4) -> list[pygame.Surface]:
    width, height = canvas_size
    frames: list[pygame.Surface] = []
    for index in range(frame_count):
        surface = pygame.Surface(canvas_size, pygame.SRCALPHA)
        _draw_generated_face(surface, emotion, index, width, height)
        frames.append(surface)
    return frames


def _draw_generated_face(surface: pygame.Surface, emotion: Emotion, frame_index: int, width: int, height: int) -> None:
    from ai.frontend.config import EMOTION_THEME

    theme = EMOTION_THEME[emotion.value]
    accent = theme["accent"]
    glow = theme["glow"]
    background = theme["backdrop_bottom"]

    surface.fill((0, 0, 0, 0))

    center_x = width // 2
    center_y = height // 2 + (frame_index - 1) * 2
    face_radius_x = min(width, height) * 0.34
    face_radius_y = min(width, height) * 0.28

    # Soft aura
    for alpha, scale in ((22, 1.35), (16, 1.18), (10, 1.0)):
        aura_rect = pygame.Rect(0, 0, int(face_radius_x * 2 * scale), int(face_radius_y * 2 * scale))
        aura_rect.center = (center_x, center_y)
        pygame.draw.ellipse(surface, (*glow, alpha), aura_rect)

    face_rect = pygame.Rect(0, 0, int(face_radius_x * 2), int(face_radius_y * 2))
    face_rect.center = (center_x, center_y)
    pygame.draw.ellipse(surface, (*background, 220), face_rect)
    pygame.draw.ellipse(surface, (*accent, 240), face_rect, width=4)

    eye_y = center_y - int(face_radius_y * 0.22)
    left_eye_x = center_x - int(face_radius_x * 0.42)
    right_eye_x = center_x + int(face_radius_x * 0.42)
    blink = 0 if emotion in {Emotion.SAD, Emotion.THINKING} and frame_index % 4 == 0 else 1

    _draw_eye(surface, left_eye_x, eye_y, emotion, frame_index, blink, accent, mirrored=False)
    _draw_eye(surface, right_eye_x, eye_y, emotion, frame_index, blink, accent, mirrored=True)
    _draw_mouth(surface, center_x, center_y, face_radius_x, face_radius_y, emotion, frame_index, accent)
    _draw_brows(surface, center_x, center_y, face_radius_x, face_radius_y, emotion, frame_index, accent)


def _draw_eye(
    surface: pygame.Surface,
    eye_x: int,
    eye_y: int,
    emotion: Emotion,
    frame_index: int,
    blink: int,
    accent: tuple[int, int, int],
    mirrored: bool,
) -> None:
    if blink == 0:
        pygame.draw.line(surface, accent, (eye_x - 18, eye_y), (eye_x + 18, eye_y), 6)
        return

    eye_width = 28
    eye_height = 20
    if emotion in {Emotion.EXCITED, Emotion.HAPPY}:
        eye_height = 24
    if emotion == Emotion.CONFUSED and mirrored:
        eye_height = 14
        eye_width = 22
    if emotion == Emotion.THINKING and not mirrored:
        eye_height = 14
        eye_width = 24

    eye_rect = pygame.Rect(0, 0, eye_width, eye_height)
    eye_rect.center = (eye_x, eye_y)
    pygame.draw.ellipse(surface, (245, 245, 250), eye_rect)
    pupil_size = 9 if emotion != Emotion.EXCITED else 11
    pupil_offset = (frame_index % 2 - 0.5) * 3
    pupil_x = eye_x + int(pupil_offset if not mirrored else -pupil_offset)
    pupil_y = eye_y + (1 if emotion in {Emotion.CURIOUS, Emotion.THINKING} else 0)
    pygame.draw.circle(surface, accent, (pupil_x, pupil_y), pupil_size)
    pygame.draw.circle(surface, (20, 20, 24), (pupil_x, pupil_y), max(2, pupil_size // 3))


def _draw_brows(
    surface: pygame.Surface,
    center_x: int,
    center_y: int,
    face_radius_x: float,
    face_radius_y: float,
    emotion: Emotion,
    frame_index: int,
    accent: tuple[int, int, int],
) -> None:
    brow_y = center_y - int(face_radius_y * 0.42)
    left_start = (center_x - int(face_radius_x * 0.57), brow_y)
    right_start = (center_x + int(face_radius_x * 0.18), brow_y)
    brow_width = 34

    if emotion == Emotion.ANGRY:
        pygame.draw.line(surface, accent, left_start, (left_start[0] + brow_width, brow_y + 10), 5)
        pygame.draw.line(surface, accent, right_start, (right_start[0] + brow_width, brow_y - 10), 5)
    elif emotion == Emotion.SAD:
        pygame.draw.line(surface, accent, (left_start[0], brow_y - 8), (left_start[0] + brow_width, brow_y - 2), 4)
        pygame.draw.line(surface, accent, (right_start[0], brow_y - 2), (right_start[0] + brow_width, brow_y - 8), 4)
    elif emotion == Emotion.CONFUSED:
        pygame.draw.line(surface, accent, (left_start[0], brow_y - 10), (left_start[0] + brow_width, brow_y - 3), 4)
        pygame.draw.line(surface, accent, (right_start[0], brow_y - 18), (right_start[0] + brow_width, brow_y - 11), 4)
    elif emotion == Emotion.THINKING:
        pygame.draw.line(surface, accent, (left_start[0], brow_y - 4), (left_start[0] + brow_width, brow_y - 4), 4)
        pygame.draw.line(surface, accent, (right_start[0], brow_y - 12), (right_start[0] + brow_width, brow_y - 10), 4)
    elif emotion == Emotion.CURIOUS:
        pygame.draw.line(surface, accent, (left_start[0], brow_y - 6), (left_start[0] + brow_width, brow_y - 3), 4)
        pygame.draw.line(surface, accent, (right_start[0], brow_y - 16), (right_start[0] + brow_width, brow_y - 6), 4)
    elif emotion == Emotion.EXCITED:
        wobble = 2 if frame_index % 2 == 0 else -2
        pygame.draw.line(surface, accent, (left_start[0], brow_y - 12 + wobble), (left_start[0] + brow_width, brow_y - 6 + wobble), 4)
        pygame.draw.line(surface, accent, (right_start[0], brow_y - 12 - wobble), (right_start[0] + brow_width, brow_y - 6 - wobble), 4)


def _draw_mouth(
    surface: pygame.Surface,
    center_x: int,
    center_y: int,
    face_radius_x: float,
    face_radius_y: float,
    emotion: Emotion,
    frame_index: int,
    accent: tuple[int, int, int],
) -> None:
    mouth_y = center_y + int(face_radius_y * 0.28)
    mouth_w = int(face_radius_x * 0.42)
    mouth_h = int(face_radius_y * 0.18)

    if emotion == Emotion.HAPPY:
        rect = pygame.Rect(0, 0, mouth_w, mouth_h)
        rect.center = (center_x, mouth_y)
        pygame.draw.arc(surface, accent, rect, 0.1, 3.05, 5)
    elif emotion == Emotion.EXCITED:
        rect = pygame.Rect(0, 0, int(mouth_w * 1.1), int(mouth_h * 1.1))
        rect.center = (center_x, mouth_y)
        pygame.draw.arc(surface, accent, rect, 0.15, 3.0, 6)
        pygame.draw.line(surface, accent, (center_x - mouth_w // 2, mouth_y), (center_x + mouth_w // 2, mouth_y), 3)
    elif emotion == Emotion.SAD:
        rect = pygame.Rect(0, 0, mouth_w, mouth_h)
        rect.center = (center_x, mouth_y + 8)
        pygame.draw.arc(surface, accent, rect, 3.3, 6.1, 4)
    elif emotion == Emotion.ANGRY:
        pygame.draw.line(surface, accent, (center_x - mouth_w // 2, mouth_y), (center_x + mouth_w // 2, mouth_y), 5)
    elif emotion == Emotion.CURIOUS:
        pygame.draw.arc(surface, accent, (center_x - 24, mouth_y - 6, 48, 30), 0.2, 3.0, 4)
    elif emotion == Emotion.CONFUSED:
        pygame.draw.lines(
            surface,
            accent,
            False,
            [
                (center_x - mouth_w // 2, mouth_y),
                (center_x - mouth_w // 6, mouth_y + 10),
                (center_x + mouth_w // 8, mouth_y - 4),
                (center_x + mouth_w // 2, mouth_y + 8),
            ],
            4,
        )
    elif emotion == Emotion.THINKING:
        wobble = 2 if frame_index % 2 else 0
        pygame.draw.line(surface, accent, (center_x - mouth_w // 4, mouth_y + wobble), (center_x + mouth_w // 4, mouth_y + wobble), 4)
    else:
        pygame.draw.line(surface, accent, (center_x - mouth_w // 4, mouth_y), (center_x + mouth_w // 4, mouth_y), 4)
