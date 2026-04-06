"""Хаотическая задача трёх тел (пифагорова конфигурация).

Три тела с массами 3, 4, 5 стартуют из покоя в вершинах
прямоугольного треугольника 3-4-5.  Гравитационная постоянная G=1.
Интегрирование методом DOP853 демонстрирует хаотическое
поведение системы.
"""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    BLUE,
    GREEN,
    RED,
    WHITE,
    Dot,
    FadeIn,
    FadeOut,
    Scene,
    Text,
    TracedPath,
    VGroup,
    rate_functions,
)
from scipy.integrate import solve_ivp

# ── физические параметры ──────────────────────────────────

G = 1.0
MASSES = np.array([3.0, 4.0, 5.0])

# Начальные позиции: вершины прямоугольного треугольника
# со сторонами 3-4-5.
INIT_POS = np.array([
    [1.0, 3.0],    # тело 1 (m=3)
    [-2.0, -1.0],  # тело 2 (m=4)
    [1.0, -1.0],   # тело 3 (m=5)
])
INIT_VEL = np.zeros((3, 2))

T_MAX = 50.0
DT_FRAME = 0.02  # шаг вывода для анимации

# ── цвета тел ─────────────────────────────────────────────

BODY_COLORS = [RED, BLUE, GREEN]
BODY_RADII = [0.10, 0.09, 0.11]

# ── масштаб сцены (Manim-координаты) ─────────────────────

SCENE_SCALE = 0.55


# ── правые части ОДУ ─────────────────────────────────────

def _derivatives(
    _t: float,
    state: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """Правая часть системы 12 ОДУ задачи трёх тел.

    state = [x1,y1, x2,y2, x3,y3, vx1,vy1, vx2,vy2, vx3,vy3]
    """
    pos = state[:6].reshape(3, 2)
    vel = state[6:].reshape(3, 2)
    acc = np.zeros_like(pos)

    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            r_vec = pos[j] - pos[i]
            dist = np.linalg.norm(r_vec)
            acc[i] += G * masses[j] * r_vec / dist ** 3

    return np.concatenate([vel.ravel(), acc.ravel()])


# ── интегрирование ────────────────────────────────────────

def _compute_trajectories() -> np.ndarray:
    """Вычислить траектории трёх тел.

    :return: массив (N, 3, 2) — позиции на каждом шаге
    """
    y0 = np.concatenate([INIT_POS.ravel(), INIT_VEL.ravel()])
    t_eval = np.arange(0.0, T_MAX, DT_FRAME)

    sol = solve_ivp(
        fun=lambda t, y: _derivatives(t, y, MASSES),
        t_span=(0.0, T_MAX),
        y0=y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    # sol.y shape: (12, N)
    n_steps = sol.y.shape[1]
    positions = sol.y[:6, :].T.reshape(n_steps, 3, 2)
    return positions


# ── построение анимации ───────────────────────────────────

def construct_animation(scene: Scene) -> None:
    """Построить и проиграть анимацию задачи трёх тел."""
    # --- заголовок ---
    title = Text(
        "Three-Body Problem",
        font_size=36,
    ).to_edge(UP, buff=0.3)
    subtitle = Text(
        "Pythagorean configuration  (m = 3, 4, 5)",
        font_size=22,
        color=WHITE,
    ).next_to(title, DOWN, buff=0.15)

    scene.play(FadeIn(title), FadeIn(subtitle))
    scene.wait(3)
    scene.play(FadeOut(subtitle))

    # --- вычисление траекторий ---
    positions = _compute_trajectories()  # (N, 3, 2)
    n_frames = positions.shape[0]

    # масштабирование в Manim-координаты
    positions_scaled = positions * SCENE_SCALE

    # --- создание Dot'ов и TracedPath ---
    dots: list[Dot] = []
    traces: list[TracedPath] = []

    for i in range(3):
        x0, y0 = positions_scaled[0, i]
        dot = Dot(
            point=np.array([x0, y0, 0.0]),
            radius=BODY_RADII[i],
            color=BODY_COLORS[i],
        )
        dot.set_z_index(2)
        dots.append(dot)

        trace = TracedPath(
            dot.get_center,
            stroke_color=BODY_COLORS[i],
            stroke_width=1.8,
            stroke_opacity=[1.0, 0.0],
            dissipating_time=1.2,
        )
        traces.append(trace)

    group = VGroup(*dots, *traces)
    scene.add(group)

    # --- анимация покадрово через updater ---
    frame_idx = [0]
    speed = max(1, n_frames // 1500)

    def _updater(_mob, dt):
        idx = frame_idx[0]
        if idx >= n_frames:
            return
        for i, dot in enumerate(dots):
            x, y = positions_scaled[idx, i]
            dot.move_to(np.array([x, y, 0.0]))
        frame_idx[0] = min(idx + speed, n_frames - 1)

    sentinel = dots[0].copy().set_opacity(0)
    sentinel.add_updater(_updater)
    scene.add(sentinel)

    anim_duration = n_frames / (speed * 60.0)
    anim_duration = max(anim_duration, 16.0)
    anim_duration = min(anim_duration, 50.0)
    scene.wait(anim_duration)

    sentinel.clear_updaters()
    scene.remove(sentinel)

    # --- финал ---
    outro = Text(
        "Chaos: small changes lead to unpredictability",
        font_size=24,
    ).to_edge(DOWN, buff=0.4)
    scene.play(FadeIn(outro))
    scene.wait(4)
    scene.play(FadeOut(group), FadeOut(title), FadeOut(outro))


# ── обёртка Scene ─────────────────────────────────────────

class ThreeBodyScene(Scene):
    """Manim-сцена хаотической задачи трёх тел."""

    def construct(self) -> None:
        construct_animation(self)
