"""Shared pygame scaffolding for the per-module visual inspectors.

Every module benchmark used to write to stdout. That was impossible for
a non-programmer to follow — the user asked for a proper UI on every
module, not a terminal. `InspectorBase` is a single pygame window with
a sidebar, a HUD, and auto/step playback controls. Each module's
inspector subclasses it and implements two methods:

    setup()     — prepare the first scenario.
    step()      — advance one scenario / tick and update internal state.
    render()    — draw the current state into self.view_rect.

The base handles window plumbing, controls, title bar, sidebar, and
the end-of-run results screen so the module-specific inspectors stay
short.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pygame


# Shared palette — matches the launcher's look so modules don't clash.
BG           = (14, 18, 26)
PANEL        = (22, 28, 38)
PANEL_ALT    = (28, 34, 48)
BORDER       = (60, 72, 95)
TEXT         = (225, 230, 240)
TEXT_DIM     = (150, 160, 175)
TEXT_ACCENT  = (120, 220, 150)
TEXT_TITLE   = (180, 220, 250)
TEXT_WARN    = (240, 180, 90)
TEXT_BAD     = (240, 110, 110)
TEXT_OK      = (120, 220, 150)

GRID_LINE    = (40, 50, 66)
GRID_AXIS    = (80, 95, 130)


@dataclass
class RunningStats:
    """Aggregates over the whole run so the sidebar can show averages."""
    label: str = ""
    values: List[float] = field(default_factory=list)

    def push(self, v: float) -> None:
        self.values.append(float(v))

    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    def last(self) -> float:
        return self.values[-1] if self.values else 0.0


class InspectorBase:
    """Base for every module's visual inspector.

    Subclass contract:
      - `title` / `subtitle` set before `run()`.
      - `total_trials` set before `run()` (how many scenarios to visit).
      - `setup()` sets up the first scenario.
      - `step()` advances to the next scenario / tick, returning True
        while still in progress.
      - `render(surface, view_rect)` draws the current scene.
      - `sidebar_lines()` returns [(label, value, style), …] rows.
      - `final_summary()` returns [(text, style), …] for the results
        modal shown when the run finishes.
    """

    WINDOW_W = 1280
    WINDOW_H = 860
    SIDEBAR_W = 360

    def __init__(self, title: str = "Inspector", subtitle: str = "",
                 total_trials: int = 50, autoplay_hz: float = 8.0):
        self.title = title
        self.subtitle = subtitle
        self.total_trials = total_trials
        self.trial_idx = 0
        self.autoplay = True
        self.autoplay_hz = autoplay_hz
        self._last_auto_t = 0.0
        self.speed_mult = 1.0
        self.finished = False
        self._start_t = time.monotonic()
        self._init_pygame()
        self.view_rect = pygame.Rect(
            16, 80, self.WINDOW_W - self.SIDEBAR_W - 32,
            self.WINDOW_H - 100,
        )
        self.sidebar_rect = pygame.Rect(
            self.WINDOW_W - self.SIDEBAR_W - 8, 80,
            self.SIDEBAR_W, self.WINDOW_H - 100,
        )

    def _init_pygame(self):
        # Pygame may or may not be initialized by the parent (the
        # launcher tears down its own display before handing off). We
        # init idempotently and own the window for the inspector's life.
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        self.screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H))
        pygame.display.set_caption(f"Drone AI — {self.title}")
        self.clock = pygame.time.Clock()
        self.font_xl = pygame.font.SysFont("Consolas", 26, bold=True)
        self.font_lg = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_md = pygame.font.SysFont("Consolas", 14)
        self.font_sm = pygame.font.SysFont("Consolas", 12)

    # ---- subclass hooks -------------------------------------------------

    def setup(self) -> None:
        raise NotImplementedError

    def step(self) -> bool:
        """Advance one unit of work. Return False when the run is done."""
        raise NotImplementedError

    def render(self, surface: pygame.Surface, view_rect: pygame.Rect) -> None:
        raise NotImplementedError

    def sidebar_lines(self) -> List[Tuple[str, str, str]]:
        """(label, value, style) rows — style in {'text','dim','ok','warn','bad'}."""
        return []

    def final_summary(self) -> List[Tuple[str, str]]:
        return [("Run complete.", "accent")]

    # ---- run loop -------------------------------------------------------

    def run(self) -> None:
        try:
            self.setup()
            running = True
            while running:
                running = self._handle_events()
                if running and self.autoplay and not self.finished:
                    now = time.monotonic()
                    interval = 1.0 / max(0.5, self.autoplay_hz * self.speed_mult)
                    if now - self._last_auto_t >= interval:
                        self._advance()
                        self._last_auto_t = now
                self._draw()
                pygame.display.flip()
                self.clock.tick(60)
                if self.finished:
                    self._show_final_modal()
                    break
        finally:
            try:
                pygame.display.quit()
            except Exception:
                pass

    def _advance(self) -> None:
        try:
            still = self.step()
        except Exception as e:
            # Surface the error in the sidebar rather than crashing the
            # inspector — we want the user to see what went wrong.
            self._error_msg = f"{type(e).__name__}: {e}"
            self.finished = True
            return
        if not still:
            self.finished = True

    def _handle_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif ev.key == pygame.K_SPACE:
                    self.autoplay = not self.autoplay
                elif ev.key in (pygame.K_RIGHT, pygame.K_n):
                    self.autoplay = False
                    if not self.finished:
                        self._advance()
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.speed_mult = min(8.0, self.speed_mult * 1.5)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.speed_mult = max(0.25, self.speed_mult / 1.5)
        return True

    # ---- drawing --------------------------------------------------------

    def _draw(self) -> None:
        self.screen.fill(BG)
        self._draw_titlebar()
        # Main view panel.
        pygame.draw.rect(self.screen, PANEL, self.view_rect, border_radius=8)
        pygame.draw.rect(self.screen, BORDER, self.view_rect, 1, border_radius=8)
        try:
            self.render(self.screen, self.view_rect)
        except Exception as e:
            # Surface render errors so a bad frame doesn't take the
            # whole window down. The sidebar keeps updating.
            msg = self.font_md.render(f"render error: {type(e).__name__}: {e}",
                                      True, TEXT_BAD)
            self.screen.blit(msg, (self.view_rect.x + 16, self.view_rect.y + 16))
        self._draw_sidebar()
        self._draw_footer()

    def _draw_titlebar(self) -> None:
        title = self.font_xl.render(self.title, True, TEXT_TITLE)
        self.screen.blit(title, (20, 16))
        if self.subtitle:
            sub = self.font_md.render(self.subtitle, True, TEXT_DIM)
            self.screen.blit(sub, (22, 50))
        # Progress chip in the top-right.
        chip_x = self.WINDOW_W - 260
        chip_text = (f"trial {self.trial_idx}/{self.total_trials}"
                     if not self.finished else "DONE")
        chip = self.font_md.render(chip_text, True, TEXT_ACCENT)
        self.screen.blit(chip, (chip_x, 20))
        mode = f"[{'auto' if self.autoplay else 'step'}]  x{self.speed_mult:.1f}"
        mchip = self.font_sm.render(mode, True, TEXT_DIM)
        self.screen.blit(mchip, (chip_x, 42))

    def _draw_sidebar(self) -> None:
        r = self.sidebar_rect
        pygame.draw.rect(self.screen, PANEL_ALT, r, border_radius=8)
        pygame.draw.rect(self.screen, BORDER, r, 1, border_radius=8)
        y = r.y + 14
        hdr = self.font_lg.render("Live metrics", True, TEXT_TITLE)
        self.screen.blit(hdr, (r.x + 14, y))
        y += 28
        for label, value, style in self.sidebar_lines():
            color = {
                "text": TEXT, "dim": TEXT_DIM,
                "ok": TEXT_OK, "warn": TEXT_WARN, "bad": TEXT_BAD,
                "accent": TEXT_ACCENT,
            }.get(style, TEXT)
            lbl_surf = self.font_md.render(label, True, TEXT_DIM)
            val_surf = self.font_md.render(value, True, color)
            self.screen.blit(lbl_surf, (r.x + 14, y))
            self.screen.blit(val_surf,
                             (r.right - val_surf.get_width() - 14, y))
            y += 22

    def _draw_footer(self) -> None:
        y = self.WINDOW_H - 20
        hint = (
            "[Space] play/pause   [→ or N] next   [+/-] speed   "
            "[Esc/Q] close"
        )
        surf = self.font_sm.render(hint, True, TEXT_DIM)
        self.screen.blit(surf, (20, y))

    # ---- results modal --------------------------------------------------

    def _show_final_modal(self) -> None:
        lines = [(self.title, "title")]
        err = getattr(self, "_error_msg", None)
        if err:
            lines.append((f"ERROR: {err}", "bad"))
        lines.extend(self.final_summary())
        lines.append(("", "dim"))
        lines.append(("Press any key / click to close.", "accent"))

        styles = {
            "title":  (self.font_xl, TEXT_TITLE),
            "accent": (self.font_md, TEXT_ACCENT),
            "text":   (self.font_md, TEXT),
            "dim":    (self.font_sm, TEXT_DIM),
            "ok":     (self.font_md, TEXT_OK),
            "warn":   (self.font_md, TEXT_WARN),
            "bad":    (self.font_md, TEXT_BAD),
        }
        rendered = []
        max_w = 0
        for text, style in lines:
            font, color = styles.get(style, styles["text"])
            surf = font.render(text, True, color)
            rendered.append(surf)
            max_w = max(max_w, surf.get_width())
        line_h = max((s.get_height() for s in rendered), default=16) + 4
        panel_w = max(520, max_w + 60)
        panel_h = len(rendered) * line_h + 40
        panel_x = (self.WINDOW_W - panel_w) // 2
        panel_y = (self.WINDOW_H - panel_h) // 2

        running = True
        while running:
            try:
                running_ok = pygame.display.get_surface() is not None
            except Exception:
                running_ok = False
            if not running_ok:
                break
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    running = False
            overlay = pygame.Surface((self.WINDOW_W, self.WINDOW_H), pygame.SRCALPHA)
            overlay.fill((10, 14, 22, 200))
            self.screen.blit(overlay, (0, 0))
            pygame.draw.rect(self.screen, (24, 30, 42),
                             (panel_x, panel_y, panel_w, panel_h), border_radius=10)
            pygame.draw.rect(self.screen, (90, 120, 160),
                             (panel_x, panel_y, panel_w, panel_h), 2, border_radius=10)
            y = panel_y + 20
            for s in rendered:
                self.screen.blit(s, (panel_x + (panel_w - s.get_width()) // 2, y))
                y += line_h
            pygame.display.flip()
            self.clock.tick(60)


# ---- 2D world projection helpers ------------------------------------------

class TopDownProjector:
    """Map world XY (meters) into a pygame rect (pixels) with padding.

    World +x maps to screen right, world +y maps to screen UP (standard
    math orientation, not pygame's default). The caller supplies
    `bounds` as ((xmin,xmax),(ymin,ymax)); autofit via `from_points()`.
    """

    def __init__(self, rect: pygame.Rect,
                 bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                 padding: int = 24):
        self.rect = rect
        (x0, x1), (y0, y1) = bounds
        # Pad to avoid glyphs touching the edge.
        self.world_w = max(1e-6, x1 - x0)
        self.world_h = max(1e-6, y1 - y0)
        self.x0, self.x1 = x0, x1
        self.y0, self.y1 = y0, y1
        self.padding = padding
        inner_w = rect.width - 2 * padding
        inner_h = rect.height - 2 * padding
        scale_x = inner_w / self.world_w
        scale_y = inner_h / self.world_h
        self.scale = min(scale_x, scale_y)
        self.cx = rect.x + rect.width // 2
        self.cy = rect.y + rect.height // 2
        self.wx = (x0 + x1) / 2
        self.wy = (y0 + y1) / 2

    @classmethod
    def from_points(cls, rect: pygame.Rect, points, padding_world: float = 5.0,
                    screen_padding: int = 24) -> "TopDownProjector":
        pts = [p for p in points if p is not None]
        if not pts:
            return cls(rect, ((-20, 20), (-20, 20)), screen_padding)
        arr = np.asarray(pts, dtype=np.float32)
        xmin = float(arr[:, 0].min()) - padding_world
        xmax = float(arr[:, 0].max()) + padding_world
        ymin = float(arr[:, 1].min()) - padding_world
        ymax = float(arr[:, 1].max()) + padding_world
        # Force a minimum extent so a single point doesn't zoom to infinity.
        if xmax - xmin < 10:
            xmin -= 5; xmax += 5
        if ymax - ymin < 10:
            ymin -= 5; ymax += 5
        return cls(rect, ((xmin, xmax), (ymin, ymax)), screen_padding)

    def to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        sx = int(self.cx + (wx - self.wx) * self.scale)
        sy = int(self.cy - (wy - self.wy) * self.scale)
        return sx, sy

    def size_px(self, world_size: float) -> int:
        return max(1, int(world_size * self.scale))


def draw_grid(surface: pygame.Surface, proj: TopDownProjector,
              step: float = 10.0) -> None:
    """Faint grid + origin axes in the projected view."""
    r = proj.rect
    # Vertical grid lines
    x = float(np.floor(proj.x0 / step) * step)
    while x <= proj.x1:
        sx, _ = proj.to_screen(x, 0.0)
        if r.left <= sx <= r.right:
            color = GRID_AXIS if abs(x) < 1e-6 else GRID_LINE
            pygame.draw.line(surface, color, (sx, r.top + 4), (sx, r.bottom - 4), 1)
        x += step
    # Horizontal grid lines
    y = float(np.floor(proj.y0 / step) * step)
    while y <= proj.y1:
        _, sy = proj.to_screen(0.0, y)
        if r.top <= sy <= r.bottom:
            color = GRID_AXIS if abs(y) < 1e-6 else GRID_LINE
            pygame.draw.line(surface, color, (r.left + 4, sy), (r.right - 4, sy), 1)
        y += step
