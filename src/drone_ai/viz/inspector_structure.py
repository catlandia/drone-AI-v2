"""Structure-and-thinking inspectors for modules that don't map neatly
onto a top-down world view: Storage, Personality, Adaptive.

Each one subclasses `StructureInspector` and provides:
  - `structure_diagram()` — boxes + arrows describing the module's
    pipeline. Drawn in the left pane as a static wiring diagram so the
    user can SEE what the module is made of without reading code.
  - `current_thinking()` — a short list of (label, value) pairs saying
    what the module is doing RIGHT NOW ("applying delta to sibling 3",
    "truncating log tail", …). Drawn in the right pane and replaces a
    terminal log.
  - `event_stream()` — the rolling list of concrete events produced so
    far. Shown at the bottom like a log tail but in the visual window.

The actual work still runs (real logic, real runs.csv row, real
graded .pt file) — the inspector just provides a window into it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pygame

from drone_ai.viz.inspector_common import (
    BG, BORDER, PANEL, PANEL_ALT, TEXT, TEXT_ACCENT, TEXT_DIM,
    TEXT_OK, TEXT_WARN, TEXT_BAD,
    InspectorBase,
)


@dataclass
class Box:
    label: str
    x: float        # 0-1 normalized in the diagram pane
    y: float
    highlight: bool = False
    status: str = ""   # "ok", "warn", "bad", or ""


@dataclass
class Arrow:
    frm: int        # Box index
    to: int


class StructureInspector(InspectorBase):
    """Pipeline-diagram-first inspector.

    The view pane is split:
      - left 55%  : structure diagram (boxes + arrows).
      - right 45% : current-thinking panel + event log.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._events: List[Tuple[str, str]] = []  # (text, style)
        self._max_events = 14

    # ---- subclass API --------------------------------------------------

    def structure_diagram(self) -> Tuple[List[Box], List[Arrow]]:
        return [], []

    def current_thinking(self) -> List[Tuple[str, str]]:
        return []

    def push_event(self, text: str, style: str = "text") -> None:
        self._events.append((text, style))
        if len(self._events) > self._max_events:
            self._events.pop(0)

    # ---- rendering ------------------------------------------------------

    def render(self, surface: pygame.Surface, view_rect: pygame.Rect) -> None:
        # Split the view rect.
        split = int(view_rect.width * 0.55)
        left = pygame.Rect(view_rect.x + 8, view_rect.y + 8,
                           split - 12, view_rect.height - 16)
        right = pygame.Rect(view_rect.x + split + 4, view_rect.y + 8,
                            view_rect.width - split - 12,
                            view_rect.height - 16)
        self._render_diagram(surface, left)
        self._render_right(surface, right)

    def _render_diagram(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PANEL_ALT, rect, border_radius=6)
        pygame.draw.rect(surface, BORDER, rect, 1, border_radius=6)
        title = self.font_lg.render("Pipeline", True, TEXT_TITLE_COLOR)
        surface.blit(title, (rect.x + 12, rect.y + 8))

        boxes, arrows = self.structure_diagram()
        if not boxes:
            msg = self.font_md.render("(no diagram)", True, TEXT_DIM)
            surface.blit(msg, (rect.x + 12, rect.y + 40))
            return
        box_w = min(200, (rect.width - 40) // 2)
        box_h = 44
        screen_boxes: List[pygame.Rect] = []
        for b in boxes:
            bx = int(rect.x + 20 + b.x * (rect.width - 40 - box_w))
            by = int(rect.y + 40 + b.y * (rect.height - 80 - box_h))
            r = pygame.Rect(bx, by, box_w, box_h)
            screen_boxes.append(r)

        # Arrows drawn behind boxes.
        for a in arrows:
            if 0 <= a.frm < len(screen_boxes) and 0 <= a.to < len(screen_boxes):
                fr = screen_boxes[a.frm]
                to = screen_boxes[a.to]
                p1 = (fr.right, fr.centery)
                p2 = (to.left, to.centery)
                pygame.draw.line(surface, (110, 130, 170), p1, p2, 2)
                # Arrow head.
                pygame.draw.polygon(
                    surface, (110, 130, 170),
                    [(p2[0], p2[1]),
                     (p2[0] - 7, p2[1] - 4),
                     (p2[0] - 7, p2[1] + 4)],
                )

        # Boxes.
        for b, r in zip(boxes, screen_boxes):
            color_fill = PANEL
            color_edge = (90, 120, 160) if b.highlight else BORDER
            pygame.draw.rect(surface, color_fill, r, border_radius=6)
            pygame.draw.rect(surface, color_edge, r, 2, border_radius=6)
            text_color = TEXT
            if b.status == "ok":
                text_color = TEXT_OK
            elif b.status == "warn":
                text_color = TEXT_WARN
            elif b.status == "bad":
                text_color = TEXT_BAD
            lbl = self.font_md.render(b.label, True, text_color)
            surface.blit(lbl, (r.x + 10, r.y + (r.height - lbl.get_height()) // 2))
            if b.highlight:
                chip = self.font_sm.render("▶ active", True, TEXT_ACCENT)
                surface.blit(chip, (r.x + 10, r.bottom - 16))

    def _render_right(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        # Top half: current thinking. Bottom half: event log.
        top = pygame.Rect(rect.x, rect.y, rect.width, rect.height // 2 - 4)
        bot = pygame.Rect(rect.x, rect.y + rect.height // 2 + 4,
                          rect.width, rect.height // 2 - 4)
        pygame.draw.rect(surface, PANEL_ALT, top, border_radius=6)
        pygame.draw.rect(surface, BORDER, top, 1, border_radius=6)
        pygame.draw.rect(surface, PANEL_ALT, bot, border_radius=6)
        pygame.draw.rect(surface, BORDER, bot, 1, border_radius=6)

        # Thinking.
        title = self.font_lg.render("Current thinking", True, TEXT_TITLE_COLOR)
        surface.blit(title, (top.x + 12, top.y + 8))
        y = top.y + 38
        for label, value in self.current_thinking():
            lbl = self.font_md.render(label, True, TEXT_DIM)
            val = self.font_md.render(value, True, TEXT)
            surface.blit(lbl, (top.x + 12, y))
            surface.blit(val, (top.right - val.get_width() - 12, y))
            y += 22
            if y > top.bottom - 14:
                break

        # Event log.
        title2 = self.font_lg.render("Events", True, TEXT_TITLE_COLOR)
        surface.blit(title2, (bot.x + 12, bot.y + 8))
        y = bot.y + 38
        style_color = {
            "text": TEXT, "dim": TEXT_DIM, "ok": TEXT_OK,
            "warn": TEXT_WARN, "bad": TEXT_BAD, "accent": TEXT_ACCENT,
        }
        for text, style in self._events:
            color = style_color.get(style, TEXT)
            surf = self.font_sm.render(text[:80], True, color)
            surface.blit(surf, (bot.x + 12, y))
            y += 15
            if y > bot.bottom - 12:
                break


# Alias so module references don't have to import from renderer3d.
TEXT_TITLE_COLOR = (180, 220, 250)
