"""Shared utilities for celestial mechanics lecture scenes.

Provides a consistent color palette and reusable background elements
(starfield) that match the Beamer presentation theme.
"""

from __future__ import annotations

import numpy as np
from manim import Dot, VGroup

# ═══════════════════════════════════════════════════════════
#  Consistent color palette  (Beamer dark‑astronomy theme)
# ═══════════════════════════════════════════════════════════

SUN_COLOR = "#FFD700"
EARTH_COLOR = "#4A90D9"
MARS_COLOR = "#C1440E"
JUPITER_COLOR = "#C88B3A"
SATURN_COLOR = "#D4A76A"
VENUS_COLOR = "#E8D5B7"
MERCURY_COLOR = "#B0B0B0"

ACCENT_BLUE = "#5A8CFA"       # Beamer accent  RGB(90, 140, 250)
HIGHLIGHT_GOLD = "#FFC832"    # Beamer highlight RGB(255, 200, 50)
DIM_TEXT = "#646478"          # Beamer dimtext
ORBIT_GREY = "#6688AA"       # Generic orbit stroke
DARK_BG = "#0C0C18"          # Beamer background RGB(12, 12, 24)

TRAJECTORY_COLORS = [
    "#FF6B6B",   # red
    "#FFA726",   # orange
    "#FFEE58",   # yellow
    "#66BB6A",   # green
    "#42A5F5",   # blue
]


# ═══════════════════════════════════════════════════════════
#  Starfield background
# ═══════════════════════════════════════════════════════════

def add_starfield(
    scene,
    n_stars: int = 200,
    seed: int = 42,
    x_range: tuple[float, float] = (-8.0, 8.0),
    y_range: tuple[float, float] = (-4.5, 4.5),
) -> VGroup:
    """Add a subtle starfield behind all scene content.

    Stars are placed at ``z_index = -10`` so they stay
    behind every other Mobject.  Returns the VGroup for
    optional later removal.
    """
    rng = np.random.default_rng(seed)
    stars = VGroup()
    for _ in range(n_stars):
        x = rng.uniform(*x_range)
        y = rng.uniform(*y_range)
        r = rng.uniform(0.008, 0.025)
        opacity = rng.uniform(0.15, 0.55)
        star = Dot(
            point=[x, y, 0],
            radius=r,
            color="#FFFFFF",
            fill_opacity=opacity,
        )
        star.set_z_index(-10)
        stars.add(star)
    scene.add(stars)
    return stars
