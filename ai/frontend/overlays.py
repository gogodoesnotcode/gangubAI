"""Overlay rendering for the GangubAI frontend."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import pygame

from ai.frontend.config import EMOTION_THEME, HELP_LINES
from ai.frontend.state import Emotion, FrontendState
from ai.frontend.timers import CountdownTimer, PomodoroSession, format_mmss


@dataclass(slots=True)
class FontPack:
    title: pygame.font.Font
    body: pygame.font.Font
    small: pygame.font.Font


def load_fonts() -> FontPack:
    pygame.font.init()
    font_name = pygame.font.match_font("dejavusans") or pygame.font.get_default_font()
    title = pygame.font.Font(font_name, 34)
    body = pygame.font.Font(font_name, 22)
    small = pygame.font.Font(font_name, 16)
    return FontPack(title=title, body=body, small=small)


def draw_background(surface: pygame.Surface, emotion: Emotion, time_s: float) -> None:
    width, height = surface.get_size()
    theme = EMOTION_THEME[emotion.value]
    top = theme["backdrop_top"]
    bottom = theme["backdrop_bottom"]

    # Vertical gradient.
    for y in range(height):
        blend = y / max(1, height - 1)
        color = (
            int(top[0] + (bottom[0] - top[0]) * blend),
            int(top[1] + (bottom[1] - top[1]) * blend),
            int(top[2] + (bottom[2] - top[2]) * blend),
        )
        pygame.draw.line(surface, color, (0, y), (width, y))

    # Subtle rings for depth.
    pulse = 0.5 + 0.5 * math.sin(time_s * 0.8)
    accent = theme["accent"]
    for index, radius in enumerate((width * 0.22, width * 0.35, width * 0.48)):
        alpha = max(8, int(24 - index * 5 + pulse * 6))
        ring = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.circle(ring, (*accent, alpha), (int(width * 0.82), int(height * 0.18)), int(radius), 3)
        surface.blit(ring, (0, 0))


def draw_status_chip(surface: pygame.Surface, fonts: FontPack, state: FrontendState) -> None:
    theme = EMOTION_THEME[state.emotion.value]
    label = f"{state.emotion.value.upper()}"
    text = fonts.body.render(label, True, (248, 248, 252))
    source = fonts.small.render(f"source: {state.source}", True, (214, 219, 230))

    chip_width = max(220, text.get_width() + 40)
    chip_height = 74
    chip = pygame.Surface((chip_width, chip_height), pygame.SRCALPHA)
    pygame.draw.rect(chip, (10, 12, 18, 170), chip.get_rect(), border_radius=20)
    pygame.draw.rect(chip, (*theme["accent"], 220), chip.get_rect(), width=2, border_radius=20)
    chip.blit(text, (18, 10))
    chip.blit(source, (18, 42))
    surface.blit(chip, (24, 20))


def draw_timer_panel(surface: pygame.Surface, fonts: FontPack, timer: CountdownTimer) -> None:
    width = 230
    height = 96
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(panel, (8, 10, 16, 165), panel.get_rect(), border_radius=20)
    pygame.draw.rect(panel, (255, 255, 255, 36), panel.get_rect(), width=1, border_radius=20)

    heading = fonts.small.render("timer", True, (190, 198, 214))
    value = fonts.title.render(format_mmss(timer.remaining_s), True, (248, 248, 252))
    panel.blit(heading, (18, 10))
    panel.blit(value, (16, 30))

    bar_rect = pygame.Rect(18, 76, 194, 10)
    pygame.draw.rect(panel, (255, 255, 255, 28), bar_rect, border_radius=6)
    progress_width = int(bar_rect.width * max(0.0, min(1.0, timer.progress)))
    if progress_width > 0:
        pygame.draw.rect(panel, (106, 199, 255, 220), (bar_rect.x, bar_rect.y, progress_width, bar_rect.height), border_radius=6)
    surface.blit(panel, (surface.get_width() - width - 24, 20))


def draw_debug_panel(surface: pygame.Surface, fonts: FontPack, fps: float, state: FrontendState, timer: CountdownTimer) -> None:
    width = 320
    height = 142
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(panel, (10, 12, 18, 150), panel.get_rect(), border_radius=20)
    pygame.draw.rect(panel, (255, 255, 255, 28), panel.get_rect(), width=1, border_radius=20)

    lines = [
        f"fps: {fps:5.1f}",
        f"emotion: {state.emotion.value}",
        f"updated: {max(0.0, time.time() - state.last_update_s):.1f}s ago",
        f"timer: {'running' if timer.running else 'paused'}",
    ]

    y = 14
    for line in lines:
        panel.blit(fonts.small.render(line, True, (232, 236, 244)), (16, y))
        y += 24

    surface.blit(panel, (24, surface.get_height() - height - 76))


def draw_overlay_card(
    surface: pygame.Surface,
    fonts: FontPack,
    title: str,
    subtitle_lines: list[str],
    progress: float,
    accent: tuple[int, int, int],
) -> None:
    width = min(720, surface.get_width() - 80)
    height = 176 if len(subtitle_lines) <= 2 else 200
    x = (surface.get_width() - width) // 2
    y = surface.get_height() - height - 28

    card = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(card, (8, 10, 16, 188), card.get_rect(), border_radius=24)
    pygame.draw.rect(card, (*accent, 220), card.get_rect(), width=2, border_radius=24)

    card.blit(fonts.title.render(title, True, (248, 248, 252)), (22, 18))
    text_y = 62
    for line in subtitle_lines:
        card.blit(fonts.body.render(line, True, (224, 230, 240)), (24, text_y))
        text_y += 26

    bar_rect = pygame.Rect(24, height - 28, width - 48, 10)
    pygame.draw.rect(card, (255, 255, 255, 28), bar_rect, border_radius=6)
    progress_width = int(bar_rect.width * max(0.0, min(1.0, progress)))
    if progress_width > 0:
        pygame.draw.rect(card, (*accent, 230), (bar_rect.x, bar_rect.y, progress_width, bar_rect.height), border_radius=6)

    surface.blit(card, (x, y))


def draw_timer_overlay(surface: pygame.Surface, fonts: FontPack, timer: CountdownTimer, accent: tuple[int, int, int]) -> None:
    draw_overlay_card(
        surface=surface,
        fonts=fonts,
        title="Timer",
        subtitle_lines=[f"Remaining {format_mmss(timer.remaining_s)}"],
        progress=timer.progress,
        accent=accent,
    )


def draw_pomodoro_overlay(surface: pygame.Surface, fonts: FontPack, session: PomodoroSession, accent: tuple[int, int, int]) -> None:
    phase_line = f"{session.phase_label.title()} {format_mmss(session.remaining_s)}"
    cycle_line = session.status_text
    draw_overlay_card(
        surface=surface,
        fonts=fonts,
        title="Pomodoro",
        subtitle_lines=[phase_line, cycle_line],
        progress=session.progress,
        accent=accent,
    )


def draw_footer(surface: pygame.Surface, fonts: FontPack) -> None:
    footer = pygame.Surface((surface.get_width() - 40, 44), pygame.SRCALPHA)
    pygame.draw.rect(footer, (8, 10, 16, 130), footer.get_rect(), border_radius=16)
    pygame.draw.rect(footer, (255, 255, 255, 22), footer.get_rect(), width=1, border_radius=16)

    y = 10
    for line in HELP_LINES:
        footer.blit(fonts.small.render(line, True, (205, 212, 224)), (16, y))
        y += 16

    surface.blit(footer, (20, surface.get_height() - 56))
