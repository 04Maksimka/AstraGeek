"""Lagrange points and whole-system rotation (inertial frame).

Part 1: static diagram of all five Lagrange points (Sun-Jupiter).
Part 2: inertial-frame animation showing Sun, Jupiter, and Trojan
asteroids orbiting the barycenter while preserving the equilateral
triangle configuration.
"""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    ORIGIN,
    TAU,
    BLUE,
    GREEN,
    ORANGE,
    RED,
    WHITE,
    YELLOW,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Scene,
    Text,
    TracedPath,
    VGroup,
    always_redraw,
    rate_functions,
    ValueTracker,
)
from scipy.optimize import brentq

# ── CR3BP parameters ────────────────────────────────────

MU = 1e-3  # mass parameter (~ Sun-Jupiter)

SUN_POS = np.array([-MU, 0.0])
PLANET_POS = np.array([1.0 - MU, 0.0])

SC = 3.2  # dimensionless -> Manim scale
CENTER_SHIFT = np.array([-0.3, 0.0, 0.0])


# ── collinear Lagrange points ───────────────────────────

def _collinear_eq(x, mu, case):
    r1 = abs(x + mu)
    r2 = abs(x - 1.0 + mu)
    if r1 < 1e-14 or r2 < 1e-14:
        return 1e30
    s1 = np.sign(x + mu)
    s2 = np.sign(x - 1.0 + mu)
    return x - (1.0 - mu) * s1 / r1**2 - mu * s2 / r2**2


def _find_lagrange_collinear(mu):
    x_l1 = brentq(lambda x: _collinear_eq(x, mu, 1), -mu + 0.01, 1.0 - mu - 0.01)
    x_l2 = brentq(lambda x: _collinear_eq(x, mu, 2), 1.0 - mu + 0.01, 2.5)
    x_l3 = brentq(lambda x: _collinear_eq(x, mu, 3), -2.5, -mu - 0.01)
    return x_l1, x_l2, x_l3


# ── triangular points ──────────────────────────────────

def _triangular_points(mu):
    l4 = np.array([0.5 - mu, np.sqrt(3.0) / 2.0])
    l5 = np.array([0.5 - mu, -np.sqrt(3.0) / 2.0])
    return l4, l5


# ── Manim helpers ───────────────────────────────────────

def _to_manim(xy):
    return np.array([xy[0] * SC, xy[1] * SC, 0.0]) + CENTER_SHIFT


def _make_label(text, point, color, direction=UP, font_size=20):
    lbl = Text(text, font_size=font_size, color=color)
    lbl.next_to(_to_manim(point), direction, buff=0.15)
    return lbl


# ── inertial-frame rotation helpers ─────────────────────

def _rotate_point(xy_rot, theta):
    """Rotate a rotating-frame point to inertial frame."""
    x, y = xy_rot
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([x * ct - y * st, x * st + y * ct])


# ── animation ──────────────────────────────────────────

def construct_animation(scene: Scene) -> None:
    """Build Lagrange points animation."""

    # ====================================================
    # Part 1: static diagram (rotating frame, theta=0)
    # ====================================================

    title = Text(
        "Lagrange Points", font_size=36,
    ).to_edge(UP, buff=0.3)
    scene.play(FadeIn(title))

    # --- main bodies ---
    sun_dot = Dot(_to_manim(SUN_POS), radius=0.22, color=YELLOW)
    sun_lbl = Text(
        "Sun", font_size=16, color=YELLOW,
    ).next_to(sun_dot, DOWN, buff=0.15)

    planet_dot = Dot(_to_manim(PLANET_POS), radius=0.12, color=ORANGE)
    planet_lbl = Text(
        "Jupiter", font_size=16, color=ORANGE,
    ).next_to(planet_dot, DOWN, buff=0.15)

    axis_line = DashedLine(
        _to_manim(np.array([-1.5, 0.0])),
        _to_manim(np.array([1.7, 0.0])),
        stroke_width=1, stroke_opacity=0.3, color=WHITE,
    )

    scene.play(
        Create(axis_line),
        FadeIn(sun_dot), FadeIn(sun_lbl),
        FadeIn(planet_dot), FadeIn(planet_lbl),
    )

    # --- collinear points ---
    x_l1, x_l2, x_l3 = _find_lagrange_collinear(MU)
    collinear = {
        "L1": np.array([x_l1, 0.0]),
        "L2": np.array([x_l2, 0.0]),
        "L3": np.array([x_l3, 0.0]),
    }

    col_dots = VGroup()
    col_labels = VGroup()
    for name, pos in collinear.items():
        d = Dot(_to_manim(pos), radius=0.06, color=RED)
        lbl = _make_label(name, pos, RED, UP)
        col_dots.add(d)
        col_labels.add(lbl)

    # --- triangular points ---
    l4, l5 = _triangular_points(MU)
    tri_data = {"L4": l4, "L5": l5}

    tri_dots = VGroup()
    tri_labels = VGroup()
    for name, pos in tri_data.items():
        d = Dot(_to_manim(pos), radius=0.07, color=GREEN)
        direction = UP if pos[1] > 0 else DOWN
        lbl = _make_label(name, pos, GREEN, direction)
        tri_dots.add(d)
        tri_labels.add(lbl)

    # dashed triangles
    tri_lines = VGroup()
    for lp in [l4, l5]:
        for a, b in [(SUN_POS, lp), (PLANET_POS, lp)]:
            tri_lines.add(DashedLine(
                _to_manim(a), _to_manim(b),
                stroke_width=1, stroke_opacity=0.25, color=GREEN,
            ))

    scene.play(FadeIn(col_dots), FadeIn(col_labels))
    scene.play(
        FadeIn(tri_dots), FadeIn(tri_labels),
        *[Create(l) for l in tri_lines],
    )

    trojan_lbl = Text(
        "Trojans", font_size=18, color=GREEN,
    ).next_to(_to_manim(l4), RIGHT, buff=0.2)
    greek_lbl = Text(
        "Greeks", font_size=18, color=GREEN,
    ).next_to(_to_manim(l5), RIGHT, buff=0.2)
    scene.play(FadeIn(trojan_lbl), FadeIn(greek_lbl))
    scene.wait(5)

    # ====================================================
    # Part 2: inertial-frame rotation
    # ====================================================

    # Fade out static diagram elements
    scene.play(
        FadeOut(col_dots), FadeOut(col_labels),
        FadeOut(trojan_lbl), FadeOut(greek_lbl),
        FadeOut(tri_dots), FadeOut(tri_labels),
        FadeOut(tri_lines), FadeOut(axis_line),
        FadeOut(sun_lbl), FadeOut(planet_lbl),
    )

    sub = Text(
        "Trojans preserve equilateral triangle",
        font_size=24,
    ).next_to(title, DOWN, buff=0.12)
    scene.play(FadeIn(sub))

    # Barycenter marker
    bary = Dot(CENTER_SHIFT, radius=0.03, color=WHITE)
    bary_lbl = Text(
        "CM", font_size=12, color=WHITE,
    ).next_to(bary, DOWN, buff=0.08)
    scene.play(FadeIn(bary), FadeIn(bary_lbl))

    # --- rotating bodies (updaters) ---
    tracker = ValueTracker(0)  # rotation angle theta

    def _sun_pos():
        th = tracker.get_value()
        xy = _rotate_point(SUN_POS, th)
        return _to_manim(xy)

    def _jup_pos():
        th = tracker.get_value()
        xy = _rotate_point(PLANET_POS, th)
        return _to_manim(xy)

    def _l4_pos():
        th = tracker.get_value()
        xy = _rotate_point(l4, th)
        return _to_manim(xy)

    def _l5_pos():
        th = tracker.get_value()
        xy = _rotate_point(l5, th)
        return _to_manim(xy)

    # Update existing dots to initial rotated positions (theta=0)
    sun_dot.move_to(_sun_pos())
    sun_dot.add_updater(lambda m: m.move_to(_sun_pos()))

    planet_dot.move_to(_jup_pos())
    planet_dot.add_updater(lambda m: m.move_to(_jup_pos()))

    # Trojan dots
    trojan_dot = Dot(_l4_pos(), radius=0.06, color=BLUE)
    trojan_dot.set_z_index(3)
    trojan_dot.add_updater(lambda m: m.move_to(_l4_pos()))

    greek_dot = Dot(_l5_pos(), radius=0.06, color=GREEN)
    greek_dot.set_z_index(3)
    greek_dot.add_updater(lambda m: m.move_to(_l5_pos()))

    # Traced paths
    jup_trace = TracedPath(
        planet_dot.get_center,
        stroke_color=ORANGE, stroke_width=1.5,
        stroke_opacity=[0.6, 0.0], dissipating_time=4.0,
    )
    trojan_trace = TracedPath(
        trojan_dot.get_center,
        stroke_color=BLUE, stroke_width=1.5,
        stroke_opacity=[0.6, 0.0], dissipating_time=4.0,
    )
    greek_trace = TracedPath(
        greek_dot.get_center,
        stroke_color=GREEN, stroke_width=1.5,
        stroke_opacity=[0.6, 0.0], dissipating_time=4.0,
    )

    # Rotating triangle lines (always_redraw)
    triangle = always_redraw(lambda: VGroup(
        DashedLine(
            _sun_pos(), _jup_pos(),
            stroke_width=1, stroke_opacity=0.3, color=WHITE,
        ),
        DashedLine(
            _sun_pos(), _l4_pos(),
            stroke_width=1, stroke_opacity=0.3, color=GREEN,
        ),
        DashedLine(
            _jup_pos(), _l4_pos(),
            stroke_width=1, stroke_opacity=0.3, color=GREEN,
        ),
        DashedLine(
            _sun_pos(), _l5_pos(),
            stroke_width=1, stroke_opacity=0.3, color=GREEN,
        ),
        DashedLine(
            _jup_pos(), _l5_pos(),
            stroke_width=1, stroke_opacity=0.3, color=GREEN,
        ),
    ))

    scene.add(
        jup_trace, trojan_trace, greek_trace,
        triangle, trojan_dot, greek_dot,
    )
    scene.play(FadeIn(trojan_dot), FadeIn(greek_dot))

    # Animate 3 full orbits
    n_orbits = 3
    scene.play(
        tracker.animate.set_value(n_orbits * TAU),
        run_time=24,
        rate_func=rate_functions.linear,
    )
    scene.wait(2)

    # --- cleanup ---
    sun_dot.clear_updaters()
    planet_dot.clear_updaters()
    trojan_dot.clear_updaters()
    greek_dot.clear_updaters()

    scene.play(
        FadeOut(VGroup(
            sun_dot, planet_dot, trojan_dot, greek_dot,
            jup_trace, trojan_trace, greek_trace,
            triangle, bary, bary_lbl, title, sub,
        )),
    )


# ── Scene wrapper ───────────────────────────────────────

class LagrangePointsScene(Scene):
    """Manim scene: Lagrange points and system rotation."""

    def construct(self) -> None:
        construct_animation(self)
