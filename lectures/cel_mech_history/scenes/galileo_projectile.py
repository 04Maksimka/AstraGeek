"""Визуализация параболического движения по Галилею.

Бросание камней к горизонту под разными углами.
Демонстрация: x = v₀ cos θ · t, y = v₀ sin θ · t − ½ g t².
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    ORANGE,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Axes,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    MathTex,
    MoveAlongPath,
    ParametricFunction,
    Scene,
    Text,
    VGroup,
    Write,
)

# ── Параметры ────────────────────────────────────────────

V0 = 7.0  # начальная скорость (условные единицы)
G = 10.0  # ускорение свободного падения
ANGLES_DEG = [15, 30, 45, 60, 75]
COLORS = [RED, ORANGE, YELLOW, GREEN, BLUE]


def _flight_time(theta: float) -> float:
    """Полное время полёта."""
    return 2.0 * V0 * np.sin(theta) / G


def _trajectory(theta: float, t: float) -> np.ndarray:
    """Координаты (x, y) в момент *t*."""
    x = V0 * np.cos(theta) * t
    y = V0 * np.sin(theta) * t - 0.5 * G * t * t
    return np.array([x, y, 0.0])


# ── Анимация ─────────────────────────────────────────────


def construct_animation(scene: Scene) -> None:
    """Построить анимацию бросания камней.

    :param scene: экземпляр Manim-сцены
    """
    # заголовок
    title = Text(
        "Galileo: Projectile Motion",
        font_size=36,
    ).to_edge(UP, buff=0.3)
    scene.play(FadeIn(title))

    # вычислить масштаб
    max_range = V0**2 / G  # при 45°
    max_h = V0**2 / (4 * G)

    axes = Axes(
        x_range=[0, max_range * 1.1, max_range / 5],
        y_range=[0, max_h * 1.3, max_h / 4],
        x_length=10,
        y_length=5,
        tips=False,
        axis_config={"include_numbers": False},
    ).shift(DOWN * 0.5)

    x_lbl = axes.get_x_axis_label("x", direction=DOWN)
    y_lbl = axes.get_y_axis_label("y")
    scene.play(Create(axes), FadeIn(x_lbl), FadeIn(y_lbl))

    # формула параболы (overlay) — показываем сразу
    parabola_eq = MathTex(
        r"y = x\tan\theta - \frac{g\,x^2}{2\,v_0^2\cos^2\!\theta}",
        font_size=24,
        color="#5A8CFA",
    ).to_corner(UP + RIGHT, buff=0.4)
    parabola_eq.set_z_index(5)
    scene.play(Write(parabola_eq), run_time=1.5)

    # траектории
    curves = VGroup()
    labels = VGroup()
    dots = []

    for angle_deg, color in zip(ANGLES_DEG, COLORS):
        theta = np.radians(angle_deg)
        t_f = _flight_time(theta)

        curve = ParametricFunction(
            lambda t, th=theta: axes.c2p(
                *_trajectory(th, t)[:2]
            ),
            t_range=[0, t_f, t_f / 200],
            color=color,
            stroke_width=3,
        )
        curves.add(curve)

        # подпись угла
        lbl = MathTex(
            f"{angle_deg}^\\circ",
            font_size=22,
            color=color,
        )
        # позиция подписи — у вершины параболы
        t_top = t_f / 2
        top_pt = _trajectory(theta, t_top)
        lbl.move_to(
            axes.c2p(top_pt[0], top_pt[1]) + UP * 0.25
        )
        labels.add(lbl)

        dot = Dot(
            axes.c2p(0, 0), radius=0.06, color=color
        )
        dots.append((dot, curve))

    # показать все траектории
    scene.play(
        *[Create(c, run_time=3) for c in curves],
    )
    scene.play(FadeIn(labels))

    # анимировать камни по траекториям
    for dot, curve in dots:
        scene.add(dot)
    scene.play(
        *[
            MoveAlongPath(d, c, run_time=4, rate_func=lambda t: t)
            for d, c in dots
        ],
    )

    # подсветить 45°
    highlight = Text(
        "Maximum range at 45\u00b0",
        font_size=24,
        color=YELLOW,
    ).to_edge(DOWN, buff=0.4)
    scene.play(FadeIn(highlight))

    # формула
    formula = MathTex(
        r"R = \frac{v_0^2 \sin 2\theta}{g}",
        font_size=32,
    ).next_to(highlight, UP, buff=0.2)
    scene.play(Write(formula))

    scene.wait(4)

    # очистка
    all_objs = VGroup(
        axes, x_lbl, y_lbl, curves, labels,
        highlight, formula, parabola_eq, title,
        *[d for d, _ in dots],
    )
    scene.play(FadeOut(all_objs))


# ── Scene-обёртка ────────────────────────────────────────


class GalileoProjectileScene(Scene):
    """Manim-сцена: баллистика Галилея."""

    def construct(self) -> None:  # noqa: D102
        construct_animation(self)
