"""Newton's cannonball thought experiment.

Projectile launched from a mountaintop with increasing speed:
parabola -> sub-orbit -> circular -> elliptical orbit.
Camera zooms into the short-range trajectory for low speeds.
Speed displayed in km/h.
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    BLUE_A,
    BLUE_E,
    DOWN,
    GREEN,
    ORANGE,
    ORIGIN,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    MovingCameraScene,
    Scene,
    Text,
    VGroup,
    VMobject,
    Write,
)
from scipy.integrate import solve_ivp

# ── Parameters (dimensionless units) ────────────────────

R_EARTH = 1.5
GM = R_EARTH * 10
R_MAX = 6.0

V_CIRC = np.sqrt(GM / R_EARTH)
V_CIRC_KMH = 28_440  # km/h at Earth's surface

SPEED_FRACS = [0.40, 0.60, 0.80, 1.00, 1.15]
SPEEDS = [f * V_CIRC for f in SPEED_FRACS]
COLORS = [RED, ORANGE, YELLOW, GREEN, BLUE_A]


# ── ODE ─────────────────────────────────────────────────


def _gravity_ode(t, state):
    x, y, vx, vy = state
    r = np.hypot(x, y)
    if r < 0.01:
        return [0, 0, 0, 0]
    a = -GM / r**3
    return [vx, vy, a * x, a * y]


def _hit_earth(t, state):
    return np.hypot(state[0], state[1]) - R_EARTH * 0.98


_hit_earth.terminal = True
_hit_earth.direction = -1


def _escape(t, state):
    return np.hypot(state[0], state[1]) - R_MAX


_escape.terminal = True
_escape.direction = 1


def _compute_trajectory(v0):
    y0 = [0.0, R_EARTH, v0, 0.0]
    sol = solve_ivp(
        _gravity_ode, [0, 80], y0,
        method="RK45",
        events=[_hit_earth, _escape],
        max_step=0.02, rtol=1e-8, atol=1e-10,
    )
    return np.column_stack([sol.y[0], sol.y[1]])


# ── Camera helpers ──────────────────────────────────────


def _try_zoom(scene, scale, target, run_time=1.0):
    """Zoom camera if MovingCameraScene is available."""
    try:
        scene.play(
            scene.camera.frame.animate.scale(scale).move_to(target),
            run_time=run_time,
        )
    except (AttributeError, Exception):
        pass


def _try_restore_camera(scene, run_time=1.0):
    """Restore camera to default framing."""
    try:
        from manim import config
        scene.play(
            scene.camera.frame.animate.set(
                width=config.frame_width,
            ).move_to(ORIGIN),
            run_time=run_time,
        )
    except (AttributeError, Exception):
        pass


# ── Animation ──────────────────────────────────────────


def construct_animation(scene: Scene) -> None:
    """Build Newton's cannon animation."""
    title = Text(
        "Newton's Cannon", font_size=36,
    ).to_edge(UP, buff=0.3)
    scene.play(FadeIn(title))

    earth = Circle(
        radius=R_EARTH, color=BLUE_E,
        fill_opacity=0.25, stroke_width=2,
    )
    earth_label = Text(
        "Earth", font_size=20, color=BLUE,
    ).move_to(earth)
    scene.play(Create(earth), FadeIn(earth_label))

    cannon = Dot([0, R_EARTH, 0], radius=0.08, color=WHITE)
    scene.play(FadeIn(cannon))

    # Pre-compute all trajectories
    all_pts = [_compute_trajectory(v0) for v0 in SPEEDS]
    all_paths = VGroup()
    all_labels = VGroup()

    for i, (pts, color) in enumerate(zip(all_pts, COLORS)):
        path = VMobject(color=color, stroke_width=2.5)
        path.set_points_smoothly(
            [np.array([x, y, 0]) for x, y in pts[::2]],
        )
        all_paths.add(path)

        speed_kmh = int(SPEED_FRACS[i] * V_CIRC_KMH)
        speed_str = f"{speed_kmh:,}".replace(",", ",")
        lbl = Text(
            f"v = {speed_str} km/h",
            font_size=16, color=color,
        ).to_corner(UP + RIGHT, buff=0.4).shift(DOWN * i * 0.35)
        all_labels.add(lbl)

    # --- First trajectory: zoom in ---
    pts0 = all_pts[0]
    mid_idx = len(pts0) // 2
    zoom_center = np.array([pts0[mid_idx, 0], pts0[mid_idx, 1], 0])
    _try_zoom(scene, 0.18, zoom_center)

    scene.play(Create(all_paths[0], run_time=4), FadeIn(all_labels[0]))
    scene.wait(1)

    # --- Second trajectory: still zoomed ---
    pts1 = all_pts[1]
    mid1 = len(pts1) // 2
    zoom2 = np.array([pts1[mid1, 0], pts1[mid1, 1], 0])
    _try_zoom(scene, 1.5, (zoom_center + zoom2) / 2, run_time=1)

    scene.play(Create(all_paths[1], run_time=4), FadeIn(all_labels[1]))
    scene.wait(1)

    # --- Zoom back out ---
    _try_restore_camera(scene)

    # --- Remaining trajectories ---
    for i in range(2, len(SPEEDS)):
        scene.play(
            Create(all_paths[i], run_time=4),
            FadeIn(all_labels[i]),
        )

    conclusion = Text(
        "Parabola and ellipse \u2014 same gravity",
        font_size=24, color=YELLOW,
    ).to_edge(DOWN, buff=0.2)
    scene.play(Write(conclusion))
    scene.wait(4)

    all_objs = VGroup(
        earth, earth_label, cannon,
        all_paths, all_labels, conclusion, title,
    )
    scene.play(FadeOut(all_objs))


# ── Scene wrapper ───────────────────────────────────────


class NewtonCannonballScene(MovingCameraScene):
    """Manim scene: Newton's cannonball thought experiment."""

    def construct(self) -> None:
        construct_animation(self)
