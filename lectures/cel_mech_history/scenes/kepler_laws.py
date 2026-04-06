"""Visualization of Kepler's three laws.

1. Ellipse construction from geometric definition + orbital motion.
2. Equal areas in equal times.
3. T^2 proportional to a^3 with real planetary data.
"""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    ORIGIN,
    PI,
    TAU,
    YELLOW,
    Axes,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Line,
    MathTex,
    MoveAlongPath,
    Polygon,
    Scene,
    Text,
    TracedPath,
    VMobject,
    VGroup,
    Write,
    always_redraw,
    rate_functions,
    ValueTracker,
)

# ── Planetary data ───────────────────────────────────────
# (a [AU], T [years], e)
PLANETS: dict[str, tuple[float, float, float]] = {
    "Mercury": (0.387, 0.241, 0.206),
    "Venus":   (0.723, 0.615, 0.007),
    "Earth":   (1.000, 1.000, 0.017),
    "Mars":    (1.524, 1.881, 0.093),
    "Jupiter": (5.203, 11.86, 0.049),
    "Saturn":  (9.537, 29.46, 0.054),
}

PLANET_COLORS: dict[str, str] = {
    "Mercury": "#A0A0A0",
    "Venus":   "#E8C56D",
    "Earth":   "#4A90D9",
    "Mars":    "#C1440E",
    "Jupiter": "#C88B3A",
    "Saturn":  "#D4A55A",
}

# ── Helpers ──────────────────────────────────────────────


def _solve_kepler(M, e, tol=1e-10, max_iter=50):
    """Solve Kepler's equation M = E - e sin(E) via Newton-Raphson."""
    E = np.copy(np.asarray(M, dtype=float))
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if np.all(np.abs(dE) < tol):
            break
    return float(E) if np.ndim(M) == 0 else E


def _true_anomaly(E, e):
    """Eccentric anomaly -> true anomaly."""
    return 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(np.asarray(E) / 2),
        np.sqrt(1 - e) * np.cos(np.asarray(E) / 2),
    )


def _ellipse_point(a, e, theta):
    """(x, y, 0) on the ellipse in the focus frame."""
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    return np.array([r * np.cos(theta), r * np.sin(theta), 0])


def _ellipse_curve(a, e, n_pts=200, center=ORIGIN):
    """VMobject ellipse centered on the focus."""
    thetas = np.linspace(0, TAU, n_pts)
    pts = [_ellipse_point(a, e, th) + center for th in thetas]
    curve = VMobject()
    curve.set_points_smoothly(pts)
    return curve


# ── First Law ────────────────────────────────────────────


def _law1(scene: Scene) -> None:
    """First Law: ellipse construction + orbital motion."""
    a, e = 3.0, 0.6
    b = a * np.sqrt(1 - e**2)
    c = a * e  # focus-to-center distance

    title = Text(
        "First Law: orbits are ellipses", font_size=28,
    ).to_edge(UP)
    scene.play(Write(title))

    # ── Phase 1: ellipse construction (string method) ────
    F1_pos = ORIGIN
    F2_pos = np.array([-2 * c, 0, 0])

    f1_dot = Dot(F1_pos, color="#FFD700", radius=0.10)
    f2_dot = Dot(F2_pos, color="#888888", radius=0.06)
    f1_lbl = Text(
        "F\u2081", font_size=20,
    ).next_to(f1_dot, DOWN, buff=0.15)
    f2_lbl = Text(
        "F\u2082", font_size=18,
    ).next_to(f2_dot, DOWN, buff=0.1)

    scene.play(
        FadeIn(f1_dot), FadeIn(f2_dot),
        Write(f1_lbl), Write(f2_lbl),
    )

    eq = MathTex(
        r"|PF_1| + |PF_2| = 2a", font_size=28,
    ).to_edge(DOWN, buff=0.5)
    scene.play(Write(eq))

    tracker = ValueTracker(0)

    def _P_pos():
        theta = tracker.get_value()
        r = a * (1 - e**2) / (1 + e * np.cos(theta))
        return np.array([r * np.cos(theta), r * np.sin(theta), 0])

    P = Dot(color="#4A90D9", radius=0.06)
    P.move_to(_P_pos())
    P.add_updater(lambda m: m.move_to(_P_pos()))

    s1 = always_redraw(
        lambda: Line(
            F1_pos, _P_pos(),
            color=YELLOW, stroke_width=2, stroke_opacity=0.7,
        ),
    )
    s2 = always_redraw(
        lambda: Line(
            _P_pos(), F2_pos,
            color=YELLOW, stroke_width=2, stroke_opacity=0.7,
        ),
    )
    trace = TracedPath(
        P.get_center,
        stroke_color="#6688CC",
        stroke_width=2,
    )

    scene.add(s1, s2, trace, P)
    scene.play(
        tracker.animate.set_value(TAU),
        run_time=16,
        rate_func=rate_functions.linear,
    )
    scene.wait(2)

    # ── Phase 2: transition to planetary orbit ───────────
    P.clear_updaters()
    scene.play(FadeOut(s1), FadeOut(s2), FadeOut(P), FadeOut(eq))

    sun_lbl = Text(
        "Sun", font_size=20, color="#FFD700",
    ).next_to(f1_dot, DOWN, buff=0.15)
    scene.play(FadeOut(f1_lbl), Write(sun_lbl))

    planet = Dot(color="#4A90D9", radius=0.08)
    n_frames = 200
    Ms = np.linspace(0, TAU, n_frames, endpoint=False)
    Es = _solve_kepler(Ms, e)
    thetas = _true_anomaly(Es, e)
    positions = [_ellipse_point(a, e, th) for th in thetas]
    planet.move_to(positions[0])
    scene.add(planet)

    path = VMobject()
    path.set_points_smoothly(positions)
    scene.play(
        MoveAlongPath(planet, path, rate_func=lambda t: t),
        run_time=12,
    )

    scene.wait(2)
    scene.play(
        *[FadeOut(m) for m in [
            title, trace, f1_dot, f2_dot,
            f2_lbl, sun_lbl, planet,
        ]],
    )


# ── Second Law ───────────────────────────────────────────


def _build_sector(a, e, M_start, M_end, center=ORIGIN, n_arc=40):
    """Sector polygon (Sun -> arc -> Sun)."""
    Ms = np.linspace(M_start, M_end, n_arc)
    Es = _solve_kepler(Ms, e)
    nus = _true_anomaly(Es, e)
    arc_pts = [_ellipse_point(a, e, nu) + center for nu in nus]
    vertices = [center] + arc_pts + [center]
    return Polygon(*vertices, stroke_width=1, fill_opacity=0.4)


def _law2(scene: Scene) -> None:
    """Second Law: equal areas in equal times."""
    a, e = 3.0, 0.6

    orbit = _ellipse_curve(a, e, n_pts=300)
    orbit.set_stroke(color="#6688CC", width=2)
    sun = Dot(ORIGIN, color="#FFD700", radius=0.12)

    title = Text(
        "Second Law: equal areas in equal times",
        font_size=26,
    ).to_edge(UP)

    scene.play(Write(title))
    scene.play(Create(orbit), FadeIn(sun))

    # formula overlay
    area_eq = MathTex(
        r"\frac{dA}{dt} = \frac{L}{2m} = \mathrm{const}",
        font_size=24,
        color="#5A8CFA",
    ).to_corner(UP + RIGHT, buff=0.35)
    area_eq.set_z_index(5)
    scene.play(Write(area_eq), run_time=1)

    dM = 0.35

    sector_peri = _build_sector(a, e, -dM / 2, dM / 2)
    sector_peri.set_fill(color="#FF6666")
    sector_peri.set_stroke(color="#FF6666")

    sector_aph = _build_sector(a, e, PI - dM / 2, PI + dM / 2)
    sector_aph.set_fill(color="#6666FF")
    sector_aph.set_stroke(color="#6666FF")

    scene.play(
        GrowFromCenter(sector_peri),
        GrowFromCenter(sector_aph),
        run_time=4,
    )

    lbl_peri = Text(
        "S\u2081", font_size=22, color="#FF6666",
    ).move_to(_ellipse_point(a, e, 0) * 0.5 + UP * 0.3)
    lbl_aph = Text(
        "S\u2082", font_size=22, color="#6666FF",
    ).move_to(_ellipse_point(a, e, PI) * 0.5 + UP * 0.3)
    eq_label = Text("S\u2081 = S\u2082", font_size=24).to_edge(DOWN)

    scene.play(Write(lbl_peri), Write(lbl_aph))
    scene.play(Write(eq_label))

    planet = Dot(color="#4A90D9", radius=0.08)
    n_frames = 200
    Ms = np.linspace(0, TAU, n_frames, endpoint=False)
    Es = _solve_kepler(Ms, e)
    nus = _true_anomaly(Es, e)
    positions = [_ellipse_point(a, e, nu) for nu in nus]
    planet.move_to(positions[0])
    scene.add(planet)

    path = VMobject()
    path.set_points_smoothly(positions)
    scene.play(
        MoveAlongPath(planet, path, rate_func=lambda t: t),
        run_time=12,
    )

    scene.wait(2)
    scene.play(
        *[FadeOut(m) for m in [
            title, orbit, sun, sector_peri, sector_aph,
            lbl_peri, lbl_aph, eq_label, planet, area_eq,
        ]],
    )


# ── Third Law ────────────────────────────────────────────


def _law3(scene: Scene) -> None:
    """Third Law: T^2 ~ a^3 for planets."""
    title = Text(
        "Third Law: T\u00b2 \u221d a\u00b3", font_size=28,
    ).to_edge(UP)
    scene.play(Write(title))

    scale = 0.38
    orbit_center = np.array([-3.0, -0.5, 0])

    orbits_group = VGroup()
    planets_group = VGroup()

    for name, (a_au, T_yr, ecc) in PLANETS.items():
        a_sc = a_au * scale
        orb = _ellipse_curve(a_sc, ecc, n_pts=120, center=orbit_center)
        col = PLANET_COLORS[name]
        orb.set_stroke(color=col, width=1.5, opacity=0.6)
        orbits_group.add(orb)
        peri_pos = _ellipse_point(a_sc, ecc, 0) + orbit_center
        dot = Dot(peri_pos, color=col, radius=0.04)
        planets_group.add(dot)

    sun_small = Dot(orbit_center, color="#FFD700", radius=0.06)

    scene.play(
        FadeIn(sun_small),
        *[Create(o) for o in orbits_group],
        *[FadeIn(d) for d in planets_group],
        run_time=4,
    )

    anim_time = 10.0
    for dot_mob, (name, (a_au, T_yr, ecc)) in zip(
        planets_group, PLANETS.items(),
    ):
        a_sc = a_au * scale
        n_fr = 120
        Ms = np.linspace(0, TAU, n_fr, endpoint=False)
        Es = _solve_kepler(Ms, ecc)
        nus = _true_anomaly(Es, ecc)
        pts = [
            _ellipse_point(a_sc, ecc, nu) + orbit_center
            for nu in nus
        ]
        path = VMobject()
        path.set_points_smoothly(pts)
        scene.play(
            MoveAlongPath(
                dot_mob, path, rate_func=lambda t: t,
            ),
            run_time=min(T_yr * 0.8, anim_time),
        )

    ax = Axes(
        x_range=[0, 900, 200],
        y_range=[0, 900, 200],
        x_length=4,
        y_length=4,
        axis_config={"font_size": 18},
        tips=False,
    ).shift(RIGHT * 3 + DOWN * 0.3)

    ax_labels = ax.get_axis_labels(
        x_label=Text("a\u00b3 (AU\u00b3)", font_size=18),
        y_label=Text("T\u00b2 (yr\u00b2)", font_size=18),
    )

    scene.play(Create(ax), Write(ax_labels), run_time=3)

    data_dots = VGroup()
    data_labels = VGroup()
    for name, (a_au, T_yr, _ecc) in PLANETS.items():
        a3 = a_au**3
        T2 = T_yr**2
        pt = ax.coords_to_point(a3, T2)
        col = PLANET_COLORS[name]
        d = Dot(pt, color=col, radius=0.05)
        lbl = Text(name, font_size=14, color=col).next_to(d, UP, buff=0.08)
        data_dots.add(d)
        data_labels.add(lbl)

    scene.play(
        *[FadeIn(d) for d in data_dots],
        *[Write(l) for l in data_labels],
        run_time=4,
    )

    line = ax.plot(
        lambda x: x, x_range=[0, 870],
        color="#FFFFFF", stroke_width=1.5,
    )
    line_label = Text(
        "T\u00b2 = a\u00b3", font_size=18,
    ).next_to(line, RIGHT, buff=0.1)

    scene.play(Create(line), Write(line_label))
    scene.wait(4)

    all_mobs = [
        title, sun_small, orbits_group, planets_group,
        ax, ax_labels, data_dots, data_labels,
        line, line_label,
    ]
    scene.play(*[FadeOut(m) for m in all_mobs])


# ── Main ─────────────────────────────────────────────────


def construct_animation(scene: Scene) -> None:
    """Sequentially demonstrate three Kepler laws."""
    _law1(scene)
    _law2(scene)
    _law3(scene)


class KeplerLawsScene(Scene):
    """Manim scene: Kepler's three laws."""

    def construct(self) -> None:
        construct_animation(self)
