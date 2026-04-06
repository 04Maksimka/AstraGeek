"""Epicycles and Fourier series: from Ptolemy to spectral analysis.

Part 1 — basic epicycles with visible radius vectors and circles.
Part 2 — cat silhouette reconstructed via DFT.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from manim import (
    BLUE_B,
    BLUE_D,
    Circle,
    Create,
    Dot,
    DOWN,
    FadeIn,
    FadeOut,
    GREEN,
    GREEN_B,
    LEFT,
    Line,
    ORANGE,
    ORIGIN,
    PI,
    PURPLE,
    RED,
    Scene,
    TAU,
    TEAL,
    Text,
    TracedPath,
    UP,
    VGroup,
    VMobject,
    WHITE,
    YELLOW,
    always_redraw,
    rate_functions,
    ValueTracker,
    Write,
)

# ── contour data ─────────────────────────────────────────
try:
    from lectures.cel_mech_history.data.cat_contour import (
        get_contour_complex,
    )
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from data.cat_contour import get_contour_complex


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

CHAIN_COLORS = [BLUE_B, GREEN, RED, PURPLE, TEAL, ORANGE, YELLOW]


def _dft(z: np.ndarray) -> list[tuple[float, float, int]]:
    """DFT of a closed contour. Returns (amp, phase, freq) sorted.

    Uses numpy FFT with correct frequency mapping: indices above N/2
    are mapped to negative frequencies (e.g. index 799 for N=800
    becomes freq=-1, not freq=799).
    """
    N = len(z)
    Z = np.fft.fft(z) / N
    freqs = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    coeffs: list[tuple[float, float, int]] = []
    for i in range(N):
        amp = float(np.abs(Z[i]))
        phase = float(np.angle(Z[i]))
        coeffs.append((amp, phase, int(freqs[i])))
    return sorted(coeffs, key=lambda x: x[0], reverse=True)


def _epicycle_tip(
    coeffs: Sequence[tuple[float, float, int]],
    t: float,
    n_terms: int,
) -> np.ndarray:
    """Tip of the epicycle chain at parameter *t*."""
    x, y = 0.0, 0.0
    for amp, phase, freq in coeffs[:n_terms]:
        angle = freq * t + phase
        x += amp * np.cos(angle)
        y += amp * np.sin(angle)
    return np.array([x, y, 0.0])


def _build_chain(
    coeffs: Sequence[tuple[float, float, int]],
    t: float,
    n_terms: int,
) -> VGroup:
    """VGroup of circles + radius-vector lines (DFT chain)."""
    group = VGroup()
    cx, cy = 0.0, 0.0
    for i, (amp, phase, freq) in enumerate(coeffs[:n_terms]):
        if amp < 1e-6:
            continue
        angle = freq * t + phase
        col = CHAIN_COLORS[i % len(CHAIN_COLORS)]
        circ = Circle(radius=amp, color=col)
        circ.set_stroke(opacity=0.4, width=1.5)
        circ.move_to([cx, cy, 0])
        nx = cx + amp * np.cos(angle)
        ny = cy + amp * np.sin(angle)
        line = Line(
            [cx, cy, 0], [nx, ny, 0],
            color=WHITE,
            stroke_width=2,
        )
        line.set_stroke(opacity=0.7)
        group.add(circ, line)
        cx, cy = nx, ny
    return group


def _build_basic_chain(
    radii: list[float],
    speeds: list[int],
    t: float,
) -> VGroup:
    """Chain of circles + radius vectors for basic epicycles."""
    group = VGroup()
    cx, cy = 0.0, 0.0
    for i, (r, s) in enumerate(zip(radii, speeds)):
        angle = s * t
        col = CHAIN_COLORS[i % len(CHAIN_COLORS)]
        circ = Circle(radius=r, color=col)
        circ.set_stroke(width=2, opacity=0.45)
        circ.move_to([cx, cy, 0])
        nx = cx + r * np.cos(angle)
        ny = cy + r * np.sin(angle)
        line = Line(
            [cx, cy, 0], [nx, ny, 0],
            color=col,
            stroke_width=3,
        )
        group.add(circ, line)
        cx, cy = nx, ny
    return group


# ═══════════════════════════════════════════════════════════
#  Part 1: basic epicycles with visible radius vectors
# ═══════════════════════════════════════════════════════════


def _part1_basic_epicycles(scene: Scene) -> None:
    title = Text("Ptolemy's Epicycles", font_size=42).to_edge(UP)
    scene.play(Write(title))
    scene.wait(0.5)

    R_def = 2.5
    tracker = ValueTracker(0)

    for n_epi, trace_color, run_t, label_text in [
        (1, ORANGE, 10, "1 epicycle"),
        (3, BLUE_B, 12, "3 epicycles"),
        (5, RED, 14, "5 epicycles"),
    ]:
        radii = [R_def * (0.35 ** i) for i in range(n_epi)]
        speeds = [(2 * k + 1) for k in range(n_epi)]

        def _pos(t, _r=radii, _s=speeds):
            x = sum(r * np.cos(s * t) for r, s in zip(_r, _s))
            y = sum(r * np.sin(s * t) for r, s in zip(_r, _s))
            return np.array([x, y, 0.0])

        planet = Dot(color=YELLOW, radius=0.08)
        planet.move_to(_pos(0))

        def _upd(mob, _fn=_pos):
            mob.move_to(_fn(tracker.get_value()))

        planet.add_updater(_upd)

        chain = always_redraw(
            lambda _r=radii, _s=speeds: _build_basic_chain(
                _r, _s, tracker.get_value(),
            ),
        )

        trace = TracedPath(
            planet.get_center,
            stroke_color=trace_color,
            stroke_width=2,
        )

        lbl = Text(
            label_text, font_size=28, color=trace_color,
        ).to_corner(DOWN + LEFT)

        scene.add(chain, trace, planet)
        scene.play(FadeIn(lbl))
        scene.play(
            tracker.animate.set_value(TAU),
            run_time=run_t,
            rate_func=rate_functions.linear,
        )
        scene.wait(1)

        planet.clear_updaters()
        scene.play(
            FadeOut(chain), FadeOut(trace),
            FadeOut(planet), FadeOut(lbl),
        )
        tracker.set_value(0)

    scene.play(FadeOut(title))
    scene.wait(0.3)


# ═══════════════════════════════════════════════════════════
#  Part 2: cat silhouette via DFT
# ═══════════════════════════════════════════════════════════


def _part2_fourier_cat(scene: Scene) -> None:
    title = Text(
        "Fourier Series: Cat Silhouette", font_size=40,
    ).to_edge(UP)
    scene.play(Write(title))
    scene.wait(0.5)

    z_pts = get_contour_complex(n_points=800)
    coeffs = _dft(z_pts)

    scale = 0.7

    def _tip(t: float, n: int) -> np.ndarray:
        pt = _epicycle_tip(coeffs, t, n)
        return pt * scale

    for n_terms, run_t, label_text in [
        (10, 16, "~10 terms (rough)"),
        (30, 20, "~30 terms (better)"),
        (100, 24, "~100 terms (detailed)"),
    ]:
        tracker = ValueTracker(0)

        dot = Dot(color=YELLOW, radius=0.05)
        dot.move_to(_tip(0, n_terms))

        trace = TracedPath(
            dot.get_center,
            stroke_color=ORANGE,
            stroke_width=2.5,
        )

        chain = always_redraw(
            lambda _n=n_terms: _build_chain(
                coeffs, tracker.get_value(), _n,
            ).scale(scale, about_point=ORIGIN),
        )

        def _make_dot_updater(nt: int):
            def _upd(mob: Dot) -> None:
                mob.move_to(_tip(tracker.get_value(), nt))
            return _upd

        dot.add_updater(_make_dot_updater(n_terms))

        lbl = Text(
            label_text, font_size=26, color=GREEN_B,
        ).to_corner(DOWN + LEFT)

        scene.add(chain, trace, dot)
        scene.play(FadeIn(lbl))
        scene.play(
            tracker.animate.set_value(TAU),
            run_time=run_t,
            rate_func=rate_functions.linear,
        )
        scene.wait(1)

        dot.clear_updaters()
        scene.play(FadeOut(chain), FadeOut(dot), FadeOut(lbl))
        scene.play(trace.animate.set_stroke(opacity=0.2))
        scene.wait(0.5)

    scene.play(FadeOut(title))
    scene.wait(0.5)


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════


def construct_animation(scene: Scene) -> None:
    """Main epicycle animation."""
    _part1_basic_epicycles(scene)
    _part2_fourier_cat(scene)


class EpicycleScene(Scene):
    """Manim scene: epicycles and Fourier series."""

    def construct(self) -> None:
        construct_animation(self)
