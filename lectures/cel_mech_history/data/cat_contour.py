"""Cat silhouette contour for Fourier series demonstration.

Sitting cat viewed from the side (facing right).
Coordinates normalized roughly in [-4, 4].
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

# fmt: off
# Sitting cat silhouette — closed contour (side view, facing right).
# Points go counter-clockwise starting from the tail tip.
CAT_POINTS: list[tuple[float, float]] = [
    # ── tail tip (curling upward, left side) ──────────
    (-3.5, 2.8), (-3.45, 2.55), (-3.35, 2.3),
    (-3.2, 2.05), (-3.0, 1.8), (-2.8, 1.6),
    (-2.6, 1.5), (-2.45, 1.45),
    # ── tail meets back ───────────────────────────────
    (-2.3, 1.5), (-2.15, 1.6), (-2.0, 1.75),
    (-1.85, 1.9), (-1.7, 2.05),
    # ── back curve ────────────────────────────────────
    (-1.5, 2.2), (-1.3, 2.35), (-1.1, 2.5),
    (-0.9, 2.6), (-0.7, 2.7), (-0.5, 2.8),
    (-0.3, 2.88), (-0.1, 2.95), (0.1, 3.0),
    # ── neck ──────────────────────────────────────────
    (0.25, 3.02), (0.38, 3.06), (0.48, 3.12),
    # ── left ear (outer edge going up) ────────────────
    (0.48, 3.25), (0.45, 3.4), (0.40, 3.55),
    (0.33, 3.72), (0.26, 3.9), (0.22, 4.08),
    (0.22, 4.25),
    (0.26, 4.38),  # ear tip
    # ── left ear (inner edge coming down) ─────────────
    (0.34, 4.22), (0.42, 4.05), (0.50, 3.88),
    (0.57, 3.72), (0.63, 3.58), (0.68, 3.45),
    (0.72, 3.35),
    # ── top of head between ears ──────────────────────
    (0.80, 3.28), (0.90, 3.24), (1.00, 3.22),
    (1.10, 3.24), (1.20, 3.28),
    # ── right ear (inner edge going up) ───────────────
    (1.28, 3.38), (1.34, 3.50), (1.40, 3.65),
    (1.48, 3.82), (1.55, 3.98), (1.62, 4.12),
    (1.68, 4.28),
    (1.74, 4.40),  # ear tip
    # ── right ear (outer edge coming down) ────────────
    (1.78, 4.25), (1.80, 4.08), (1.79, 3.90),
    (1.76, 3.72), (1.72, 3.55), (1.68, 3.42),
    (1.62, 3.30),
    # ── forehead and face ─────────────────────────────
    (1.58, 3.18), (1.56, 3.05), (1.58, 2.92),
    (1.62, 2.80), (1.68, 2.68), (1.76, 2.58),
    # ── nose ──────────────────────────────────────────
    (1.85, 2.52), (1.94, 2.48), (2.00, 2.44),
    (2.04, 2.38), (2.02, 2.30),
    # ── mouth / chin ──────────────────────────────────
    (1.96, 2.24), (1.88, 2.18), (1.80, 2.10),
    (1.72, 2.00), (1.64, 1.88), (1.56, 1.72),
    # ── chest ─────────────────────────────────────────
    (1.44, 1.52), (1.32, 1.30), (1.22, 1.08),
    (1.14, 0.85), (1.08, 0.62), (1.03, 0.40),
    (0.98, 0.18), (0.95, -0.05),
    # ── right front leg ──────────────────────────────
    (0.93, -0.28), (0.92, -0.52), (0.91, -0.76),
    (0.91, -1.00), (0.91, -1.24), (0.90, -1.48),
    (0.88, -1.65),
    # ── right front paw ──────────────────────────────
    (0.84, -1.80), (0.76, -1.90), (0.64, -1.96),
    (0.50, -1.96), (0.38, -1.90), (0.30, -1.80),
    # ── between front legs ───────────────────────────
    (0.26, -1.60), (0.22, -1.38), (0.18, -1.18),
    (0.14, -1.35), (0.10, -1.55), (0.06, -1.72),
    # ── left front paw ───────────────────────────────
    (0.00, -1.85), (-0.10, -1.94), (-0.24, -1.98),
    (-0.38, -1.94), (-0.48, -1.85), (-0.52, -1.72),
    # ── belly rising from front paws ─────────────────
    (-0.55, -1.50), (-0.58, -1.30), (-0.62, -1.12),
    (-0.68, -0.96), (-0.76, -0.82), (-0.86, -0.72),
    # ── belly curve ──────────────────────────────────
    (-1.00, -0.65), (-1.18, -0.60), (-1.36, -0.60),
    (-1.54, -0.64), (-1.68, -0.72), (-1.80, -0.84),
    # ── back haunch ──────────────────────────────────
    (-1.90, -1.00), (-1.98, -1.18), (-2.06, -1.36),
    (-2.12, -1.52), (-2.16, -1.66),
    # ── back paw ─────────────────────────────────────
    (-2.20, -1.80), (-2.28, -1.90), (-2.40, -1.96),
    (-2.54, -1.98), (-2.66, -1.94), (-2.74, -1.84),
    (-2.76, -1.68),
    # ── up from back paw along rump ──────────────────
    (-2.72, -1.45), (-2.64, -1.20), (-2.56, -0.95),
    (-2.50, -0.70), (-2.48, -0.45), (-2.50, -0.20),
    # ── rump to tail base ────────────────────────────
    (-2.55, 0.05), (-2.62, 0.30), (-2.70, 0.55),
    (-2.80, 0.80), (-2.90, 1.05), (-3.00, 1.30),
    # ── tail going up ────────────────────────────────
    (-3.08, 1.52), (-3.16, 1.75), (-3.24, 1.98),
    (-3.32, 2.18), (-3.40, 2.40), (-3.46, 2.60),
]
# fmt: on


def get_contour_complex(
    n_points: int = 500,
) -> np.ndarray:
    """Return contour as an array of complex numbers.

    Interpolates the raw points to *n_points* uniformly
    distributed by arc length.

    :param n_points: number of output points
    :return: complex array z = x + iy
    """
    pts = np.array(CAT_POINTS)
    # close the contour
    pts = np.vstack([pts, pts[0:1]])
    # arc-length parameterization
    diffs = np.diff(pts, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0], np.cumsum(seg_len)])
    total = cum_len[-1]
    # cubic spline interpolation (periodic) for smooth curves
    cs_x = CubicSpline(cum_len, pts[:, 0], bc_type="periodic")
    cs_y = CubicSpline(cum_len, pts[:, 1], bc_type="periodic")
    t_uniform = np.linspace(0, total, n_points, endpoint=False)
    x_interp = cs_x(t_uniform)
    y_interp = cs_y(t_uniform)
    return x_interp + 1j * y_interp
