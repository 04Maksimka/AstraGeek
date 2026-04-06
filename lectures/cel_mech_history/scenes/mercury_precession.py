"""Визуализация прецессии перигелия Меркурия (ОТО).

Сравнение ньютоновской замкнутой орбиты с орбитой,
скорректированной поправкой общей теории относительности.
Эффект прецессии усилен для наглядности.
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    DARK_GREY,
    DOWN,
    ORANGE,
    RIGHT,
    UP,
    YELLOW,
    Arc,
    Create,
    Dot,
    Ellipse,
    FadeIn,
    FadeOut,
    MathTex,
    MoveAlongPath,
    Scene,
    Text,
    TracedPath,
    VGroup,
    VMobject,
    Write,
)
from scipy.integrate import solve_ivp

# ── Параметры орбиты ─────────────────────────────────────

# Безразмерные единицы: GM = 1, начальное r = 1 (перигелий)
GM = 1.0
E_NEWTON = 0.6  # эксцентриситет
A = 1.0 / (1 - E_NEWTON)  # полуось ~ 2.5
R_PERI = A * (1 - E_NEWTON)  # = 1.0
V_PERI = np.sqrt(GM * (2 / R_PERI - 1 / A))

# ОТО-поправка: 3 GM / c².
# Реальное значение ≈ 1.1e-8 для Меркурия, мы усиливаем.
ALPHA_GR = 0.015  # усиленная поправка для наглядности

N_ORBITS = 10  # количество витков
T_ORBIT = 2 * np.pi * A**1.5 / np.sqrt(GM)
T_MAX = N_ORBITS * T_ORBIT

SCALE = 1.8  # масштаб на экране


# ── ODE ──────────────────────────────────────────────────


def _ode_gr(t, state):  # noqa: ARG001
    """Уравнения движения с ОТО-поправкой.

    Эффективный потенциал:
      V = -GM/r + L²/(2r²) - alpha·GM·L²/r³
    """
    x, y, vx, vy = state
    r2 = x * x + y * y
    r = np.sqrt(r2)
    if r < 1e-6:
        return [0, 0, 0, 0]
    # удельный момент импульса (сохраняется в Ньютоне)
    L2 = (x * vy - y * vx) ** 2
    # ньютоновское ускорение + ОТО-поправка
    a_newton = -GM / r**3
    # поправка: доп. сила = -d/dr(-alpha*GM*L²/r³)
    #         = -3 alpha GM L² / r^5 (радиальная)
    a_gr = -3 * ALPHA_GR * GM * L2 / r**5
    a_total = a_newton + a_gr
    return [vx, vy, a_total * x, a_total * y]


def _ode_newton(t, state):  # noqa: ARG001
    """Чисто ньютоновские уравнения."""
    x, y, vx, vy = state
    r = np.hypot(x, y)
    if r < 1e-6:
        return [0, 0, 0, 0]
    a = -GM / r**3
    return [vx, vy, a * x, a * y]


def _integrate(ode_func, t_max):
    """Проинтегрировать орбиту."""
    y0 = [R_PERI, 0.0, 0.0, V_PERI]
    sol = solve_ivp(
        ode_func,
        [0, t_max],
        y0,
        method="DOP853",
        max_step=T_ORBIT / 500,
        rtol=1e-11,
        atol=1e-13,
    )
    return sol.y[0], sol.y[1]


# ── Анимация ─────────────────────────────────────────────


def construct_animation(scene: Scene) -> None:
    """Построить анимацию прецессии Меркурия.

    :param scene: экземпляр Manim-сцены
    """
    title = Text(
        "Mercury Perihelion Precession (GR)",
        font_size=34,
    ).to_edge(UP, buff=0.3)
    scene.play(FadeIn(title))

    # Солнце
    sun = Dot([0, 0, 0], radius=0.15, color=YELLOW)
    sun_lbl = Text(
        "Sun", font_size=16, color=YELLOW,
    ).next_to(sun, DOWN, buff=0.15)
    scene.play(FadeIn(sun), FadeIn(sun_lbl))

    # ── Ньютоновская орбита (аналитический эллипс) ───────
    a_orbit = A
    b_orbit = A * np.sqrt(1 - E_NEWTON**2)
    center_x = -(a_orbit - R_PERI)  # центр эллипса относительно Солнца

    newton_ellipse = Ellipse(
        width=2 * a_orbit * SCALE,
        height=2 * b_orbit * SCALE,
        color=DARK_GREY,
        stroke_width=1.5,
        stroke_opacity=0.5,
    )
    newton_ellipse.move_to([center_x * SCALE, 0, 0])
    scene.play(Create(newton_ellipse, run_time=3))

    newton_lbl = Text(
        "Newton (closed orbit)",
        font_size=18,
        color=DARK_GREY,
    ).to_edge(DOWN, buff=0.8)
    scene.play(FadeIn(newton_lbl))

    # formula overlay: Binet equation with GR correction
    binet_eq = MathTex(
        r"u'' + u = \frac{GM}{L^2} + ",
        r"\frac{3GM}{c^2} u^2",
        font_size=22,
    ).to_corner(UP + RIGHT, buff=0.3)
    binet_eq[0].set_color("#AAAACC")
    binet_eq[1].set_color("#FF6666")    # GR term in red
    binet_eq.set_z_index(5)
    scene.play(Write(binet_eq), run_time=1.5)

    # ── ОТО-орбита (цветная, с прецессией) ───────────────
    xg, yg = _integrate(_ode_gr, T_MAX)

    # анимируем точку
    planet = Dot(
        [xg[0] * SCALE, yg[0] * SCALE, 0],
        radius=0.07,
        color=ORANGE,
    )
    trail = TracedPath(
        planet.get_center,
        stroke_color=BLUE,
        stroke_width=1.8,
        stroke_opacity=0.7,
    )
    scene.add(trail)

    gr_lbl = Text(
        "GR (precessing orbit)",
        font_size=18,
        color=BLUE,
    ).to_edge(DOWN, buff=0.4)
    scene.play(FadeIn(planet), FadeIn(gr_lbl))

    # анимировать движение планеты (много точек → гладкая кривая)
    step = max(1, len(xg) // 3000)
    path_pts = [
        np.array([xg[i] * SCALE, yg[i] * SCALE, 0])
        for i in range(0, len(xg), step)
    ]
    gr_path = VMobject()
    gr_path.set_points_as_corners(path_pts)

    scene.play(MoveAlongPath(planet, gr_path, run_time=20, rate_func=lambda t: t))

    # отметить прецессию
    # вычислить угол прецессии из последнего перигелия
    # найти перигелии: минимумы r
    r_gr = np.sqrt(xg**2 + yg**2)
    perihelion_idx = []
    for i in range(1, len(r_gr) - 1):
        if r_gr[i] < r_gr[i - 1] and r_gr[i] < r_gr[i + 1]:
            perihelion_idx.append(i)

    if len(perihelion_idx) >= 2:
        i0 = perihelion_idx[0]
        i_last = perihelion_idx[-1]
        angle0 = np.arctan2(yg[i0], xg[i0])
        angle_last = np.arctan2(yg[i_last], xg[i_last])
        delta = angle_last - angle0

        arc = Arc(
            radius=0.6,
            start_angle=angle0,
            angle=delta,
            color=ORANGE,
            stroke_width=3,
        )
        arc_lbl = MathTex(
            rf"\Delta\varphi \approx {np.degrees(delta):.1f}^\circ",
            font_size=24,
            color=ORANGE,
        ).next_to(arc, direction=UP, buff=0.1)
        scene.play(Create(arc), Write(arc_lbl))

    # итоговая подпись
    real_val = Text(
        'Real Mercury precession: 43" per century',
        font_size=22,
        color=YELLOW,
    ).to_edge(UP, buff=0.8)
    scene.play(FadeIn(real_val))
    scene.wait(4)

    all_objs = VGroup(
        sun, sun_lbl, newton_ellipse, newton_lbl,
        planet, gr_lbl, title, real_val, binet_eq,
    )
    scene.play(FadeOut(all_objs))


# ── Scene-обёртка ────────────────────────────────────────


class MercuryPrecessionScene(Scene):
    """Manim-сцена: прецессия Меркурия (ОТО vs Ньютон)."""

    def construct(self) -> None:  # noqa: D102
        construct_animation(self)
