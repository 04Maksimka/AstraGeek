"""Retrograde motion: heliocentric vs geocentric explanation.

Part 1 — Heliocentric model: Earth overtakes Mars and the apparent
         position on the sky reverses (retrograde loop).
Part 2 — Geocentric model:  deferent + epicycle produce the same
         retrograde pattern.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    PI,
    RIGHT,
    TAU,
    UP,
    WHITE,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Rectangle,
    Scene,
    Text,
    TracedPath,
    VGroup,
    ValueTracker,
    Write,
    rate_functions,
)

try:
    from lectures.cel_mech_history.scenes.scene_utils import (
        ACCENT_BLUE,
        DIM_TEXT,
        EARTH_COLOR,
        HIGHLIGHT_GOLD,
        MARS_COLOR,
        SUN_COLOR,
        add_starfield,
    )
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from scene_utils import (
        ACCENT_BLUE,
        DIM_TEXT,
        EARTH_COLOR,
        HIGHLIGHT_GOLD,
        MARS_COLOR,
        SUN_COLOR,
        add_starfield,
    )

# ═══════════════════════════════════════════════════════════
#  Physical / visual constants
# ═══════════════════════════════════════════════════════════

T_EARTH = 1.0                   # Earth period  (years, normalised)
T_MARS = 1.881                  # Mars sidereal period
OMEGA_E = TAU / T_EARTH
OMEGA_M = TAU / T_MARS

A_E = 1.5                       # visual Earth‑orbit radius (Manim units)
A_M = 2.4                       # visual Mars‑orbit radius
MARS_INCL = 0.16                # slight inclination → 2‑D retrograde loop

T_ANIM = 2.5                    # total simulated time (years)
ANIM_DUR = 28                   # animation run‑time (seconds) per part

# Initial Mars angle chosen so opposition is near t ≈ T_ANIM/2
THETA0_MARS = (OMEGA_E - OMEGA_M) * T_ANIM / 2   # ≈ 3.68 rad

# ── Layout ────────────────────────────────────────────────
ORBIT_CY = -0.6                 # vertical centre of orbit diagrams
STRIP_Y = 3.3                   # vertical centre of sky strip
STRIP_HH = 0.35                 # half‑height
STRIP_LEFT = -6.0
STRIP_RIGHT = 6.0

# ═══════════════════════════════════════════════════════════
#  Orbital helpers
# ═══════════════════════════════════════════════════════════


def _earth_xy(t: float) -> tuple[float, float]:
    return A_E * np.cos(OMEGA_E * t), A_E * np.sin(OMEGA_E * t)


def _mars_xy(t: float) -> tuple[float, float]:
    a = THETA0_MARS + OMEGA_M * t
    return A_M * np.cos(a), A_M * np.sin(a)


def _mars_z(t: float) -> float:
    """Out‑of‑plane component (gives 2‑D loop on the strip)."""
    a = THETA0_MARS + OMEGA_M * t
    return A_M * np.sin(a) * np.sin(MARS_INCL)


# ── Pre‑compute unwrapped apparent longitude for strip mapping ──

_N_SAMP = 3000
_t_arr = np.linspace(0, T_ANIM, _N_SAMP)
_lons_raw: list[float] = []
_lats: list[float] = []

for _t in _t_arr:
    _ex, _ey = _earth_xy(_t)
    _mx, _my = _mars_xy(_t)
    _mz = _mars_z(_t)
    _dx, _dy, _dz = _mx - _ex, _my - _ey, _mz
    _lons_raw.append(np.arctan2(_dy, _dx))
    _dist = max(np.sqrt(_dx**2 + _dy**2 + _dz**2), 1e-9)
    _lats.append(_dz / _dist)

_lons_uw = np.unwrap(_lons_raw)
_lon_min, _lon_max = float(_lons_uw[0]), float(_lons_uw[-1])
_lon_span = _lon_max - _lon_min if abs(_lon_max - _lon_min) > 0.01 else 1.0

_lon_interp = interp1d(_t_arr, _lons_uw, kind="linear", fill_value="extrapolate")
_lat_interp = interp1d(_t_arr, _lats, kind="linear", fill_value="extrapolate")


def _strip_pos(t: float) -> np.ndarray:
    """Position of Mars on the sky strip at simulation‑time *t*."""
    tc = float(np.clip(t, 0, T_ANIM - 1e-6))
    lon = float(_lon_interp(tc))
    lat = float(_lat_interp(tc))
    x = STRIP_LEFT + (lon - _lon_min) / _lon_span * (STRIP_RIGHT - STRIP_LEFT)
    y = STRIP_Y + lat * 4.0        # exaggerate latitude for visible loop
    return np.array([x, y, 0.0])


# ═══════════════════════════════════════════════════════════
#  Sky‑strip builder (shared by both parts)
# ═══════════════════════════════════════════════════════════


def _build_strip() -> VGroup:
    """Background rectangle + faint 'star' ticks for the sky strip."""
    bg = Rectangle(
        width=STRIP_RIGHT - STRIP_LEFT + 0.6,
        height=2 * STRIP_HH,
        fill_color="#080818",
        fill_opacity=0.65,
        stroke_color="#333355",
        stroke_width=1,
    ).move_to([0, STRIP_Y, 0])

    lbl = Text(
        "Apparent sky position",
        font_size=13,
        color=DIM_TEXT,
    ).next_to(bg, UP, buff=0.04)

    # Zodiac ticks
    zodiac = ["Gem", "Cnc", "Leo", "Vir", "Lib", "Sco"]
    ticks = VGroup()
    for i, name in enumerate(zodiac):
        x = STRIP_LEFT + (i + 0.5) / len(zodiac) * (STRIP_RIGHT - STRIP_LEFT)
        tick = Text(name, font_size=10, color="#3A3A50")
        tick.move_to([x, STRIP_Y - STRIP_HH + 0.08, 0])
        ticks.add(tick)

    return VGroup(bg, lbl, ticks)


# ═══════════════════════════════════════════════════════════
#  Part 1 — Heliocentric model
# ═══════════════════════════════════════════════════════════


def _part1_heliocentric(scene: Scene) -> None:
    title = Text(
        "Heliocentric Model: Earth Overtakes Mars",
        font_size=30,
    ).to_edge(UP, buff=0.08).set_z_index(5)
    scene.play(Write(title), run_time=1.5)

    # ── sky strip ──
    strip = _build_strip()
    scene.play(FadeIn(strip), run_time=1)

    # ── orbits ──
    center = np.array([0, ORBIT_CY, 0])
    sun = Dot(center, radius=0.18, color=SUN_COLOR).set_z_index(2)
    sun_lbl = Text("Sun", font_size=12, color=SUN_COLOR).next_to(
        sun, direction=DOWN, buff=0.12
    )

    e_orbit = Circle(radius=A_E, color=EARTH_COLOR, stroke_width=1.2, stroke_opacity=0.35)
    e_orbit.move_to(center)
    m_orbit = Circle(radius=A_M, color=MARS_COLOR, stroke_width=1.2, stroke_opacity=0.35)
    m_orbit.move_to(center)

    scene.play(FadeIn(sun), FadeIn(sun_lbl), Create(e_orbit), Create(m_orbit), run_time=1.5)

    # ── planets ──
    tracker = ValueTracker(0)

    earth = Dot(color=EARTH_COLOR, radius=0.10).set_z_index(3)
    mars = Dot(color=MARS_COLOR, radius=0.10).set_z_index(3)

    def _upd_earth(mob):
        ex, ey = _earth_xy(tracker.get_value())
        mob.move_to([ex + center[0], ey + center[1], 0])

    def _upd_mars(mob):
        mx, my = _mars_xy(tracker.get_value())
        mob.move_to([mx + center[0], my + center[1], 0])

    earth.add_updater(_upd_earth)
    mars.add_updater(_upd_mars)

    # sight line (plain Line — DashedLine breaks always_redraw/become)
    sight = Line(
        earth.get_center(), mars.get_center(),
        color=WHITE, stroke_opacity=0.25, stroke_width=1,
    )

    def _upd_sight(mob):
        mob.put_start_and_end_on(earth.get_center(), mars.get_center())

    sight.add_updater(_upd_sight)

    # projected dot on strip
    proj_dot = Dot(color=MARS_COLOR, radius=0.055).set_z_index(4)

    def _upd_proj(mob):
        mob.move_to(_strip_pos(tracker.get_value()))

    proj_dot.add_updater(_upd_proj)

    trace_strip = TracedPath(
        proj_dot.get_center,
        stroke_color=MARS_COLOR,
        stroke_width=2.8,
        stroke_opacity=0.85,
    )

    # legend
    legend = VGroup(
        VGroup(Dot(color=EARTH_COLOR, radius=0.06),
               Text("Earth", font_size=14, color=EARTH_COLOR)).arrange(buff=0.1),
        VGroup(Dot(color=MARS_COLOR, radius=0.06),
               Text("Mars", font_size=14, color=MARS_COLOR)).arrange(buff=0.1),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_corner(
        DOWN + LEFT, buff=0.3
    )

    scene.add(sight, trace_strip, earth, mars, proj_dot)
    scene.play(FadeIn(legend), run_time=0.8)

    # ── animate ──
    scene.play(
        tracker.animate.set_value(T_ANIM),
        run_time=ANIM_DUR,
        rate_func=rate_functions.linear,
    )

    # highlight the retrograde portion
    retro_lbl = Text("← Retrograde →", font_size=18, color=HIGHLIGHT_GOLD)
    retro_lbl.move_to([0, STRIP_Y + STRIP_HH + 0.25, 0])
    scene.play(FadeIn(retro_lbl))
    scene.wait(3)

    # cleanup
    earth.clear_updaters()
    mars.clear_updaters()
    sight.clear_updaters()
    proj_dot.clear_updaters()
    scene.play(
        FadeOut(VGroup(
            title, strip, sun, sun_lbl, e_orbit, m_orbit,
            earth, mars, sight, proj_dot, trace_strip,
            legend, retro_lbl,
        )),
        run_time=1.5,
    )


# ═══════════════════════════════════════════════════════════
#  Part 2 — Geocentric (Ptolemaic) model
# ═══════════════════════════════════════════════════════════


def _part2_geocentric(scene: Scene) -> None:
    title = Text(
        "Geocentric Model: Epicycle Produces the Same Loop",
        font_size=28,
    ).to_edge(UP, buff=0.08).set_z_index(5)
    scene.play(Write(title), run_time=1.5)

    # ── sky strip ──
    strip = _build_strip()
    scene.play(FadeIn(strip), run_time=1)

    # ── geocentric diagram ──
    center = np.array([0, ORBIT_CY, 0])
    earth_dot = Dot(center, radius=0.14, color=EARTH_COLOR).set_z_index(2)
    earth_lbl = Text("Earth", font_size=12, color=EARTH_COLOR).next_to(
        earth_dot, direction=DOWN, buff=0.12
    )

    deferent = Circle(radius=A_M, color="#555577", stroke_width=1.2, stroke_opacity=0.4)
    deferent.move_to(center)
    def_lbl = Text("Deferent", font_size=12, color="#555577").move_to(
        center + np.array([A_M + 0.3, 0.3, 0])
    )

    scene.play(
        FadeIn(earth_dot), FadeIn(earth_lbl),
        Create(deferent), FadeIn(def_lbl),
        run_time=1.5,
    )

    tracker = ValueTracker(0)

    # epicycle centre on deferent
    epi_center_dot = Dot(color="#777799", radius=0.04).set_z_index(1)

    def _epi_center(t: float) -> np.ndarray:
        a = THETA0_MARS + OMEGA_M * t
        return center + np.array([A_M * np.cos(a), A_M * np.sin(a), 0])

    def _upd_epi_c(mob):
        mob.move_to(_epi_center(tracker.get_value()))

    epi_center_dot.add_updater(_upd_epi_c)

    # epicycle circle follows its centre
    epicycle_circle = Circle(
        radius=A_E, color="#6666AA", stroke_width=1, stroke_opacity=0.35,
    ).move_to(_epi_center(0))

    def _upd_epi_circ(mob):
        mob.move_to(_epi_center(tracker.get_value()))

    epicycle_circle.add_updater(_upd_epi_circ)

    epi_lbl = Text("Epicycle", font_size=11, color="#6666AA").move_to(
        _epi_center(0) + np.array([0, A_E + 0.2, 0])
    )

    def _upd_epi_lbl(mob):
        mob.move_to(_epi_center(tracker.get_value()) + np.array([0, A_E + 0.2, 0]))

    epi_lbl.add_updater(_upd_epi_lbl)

    # Mars on epicycle  (phase offset π = opposite direction)
    mars = Dot(color=MARS_COLOR, radius=0.10).set_z_index(3)

    def _mars_geo(t: float) -> np.ndarray:
        ec = _epi_center(t)
        a = OMEGA_E * t + PI           # epicycle rotation + π offset
        return ec + np.array([A_E * np.cos(a), A_E * np.sin(a), 0])

    def _upd_mars(mob):
        mob.move_to(_mars_geo(tracker.get_value()))

    mars.add_updater(_upd_mars)

    # radius vector inside epicycle
    radius_vec = Line(
        _epi_center(0), _mars_geo(0),
        color=MARS_COLOR, stroke_width=1.5, stroke_opacity=0.5,
    )

    def _upd_rv(mob):
        mob.put_start_and_end_on(
            _epi_center(tracker.get_value()),
            _mars_geo(tracker.get_value()),
        )

    radius_vec.add_updater(_upd_rv)

    # sight line from Earth to Mars
    sight = Line(
        center, _mars_geo(0),
        color=WHITE, stroke_opacity=0.25, stroke_width=1,
    )

    def _upd_sight2(mob):
        mob.put_start_and_end_on(center, _mars_geo(tracker.get_value()))

    sight.add_updater(_upd_sight2)

    # projected dot on strip (same _strip_pos — same apparent motion!)
    proj_dot = Dot(color=MARS_COLOR, radius=0.055).set_z_index(4)

    def _upd_proj(mob):
        mob.move_to(_strip_pos(tracker.get_value()))

    proj_dot.add_updater(_upd_proj)

    trace_strip = TracedPath(
        proj_dot.get_center,
        stroke_color=ACCENT_BLUE,
        stroke_width=2.8,
        stroke_opacity=0.85,
    )

    scene.add(
        epicycle_circle, epi_center_dot, epi_lbl,
        radius_vec, sight, mars, trace_strip, proj_dot,
    )
    scene.wait(0.5)

    # ── animate ──
    scene.play(
        tracker.animate.set_value(T_ANIM),
        run_time=ANIM_DUR,
        rate_func=rate_functions.linear,
    )

    retro_lbl = Text("Same retrograde loop!", font_size=18, color=HIGHLIGHT_GOLD)
    retro_lbl.move_to([0, STRIP_Y + STRIP_HH + 0.25, 0])
    scene.play(FadeIn(retro_lbl))
    scene.wait(3)

    # cleanup
    epi_center_dot.clear_updaters()
    epicycle_circle.clear_updaters()
    epi_lbl.clear_updaters()
    mars.clear_updaters()
    radius_vec.clear_updaters()
    sight.clear_updaters()
    proj_dot.clear_updaters()
    scene.play(
        FadeOut(VGroup(
            title, strip, earth_dot, earth_lbl, deferent, def_lbl,
            epicycle_circle, epi_center_dot, epi_lbl,
            radius_vec, sight, mars, trace_strip, proj_dot, retro_lbl,
        )),
        run_time=1.5,
    )


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════


def construct_animation(scene: Scene) -> None:
    """Full retrograde‑motion animation."""
    add_starfield(scene, n_stars=150, seed=77)
    _part1_heliocentric(scene)
    _part2_geocentric(scene)


class RetrogradeMotionScene(Scene):
    """Standalone Manim scene for retrograde motion."""

    def construct(self) -> None:
        construct_animation(self)
