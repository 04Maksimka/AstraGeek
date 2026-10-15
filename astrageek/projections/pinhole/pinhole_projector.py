"""Module with pinhole projector."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from astrageek.catalogs.constellations import (
    get_available_constellations,
    get_constellation_center,
)
from astrageek.catalogs.hip import Catalog, CatalogConstraints
from astrageek.catalogs.planets import PlanetCatalog, Planets
from astrageek.helpers.constellations.constellation_renderer_pinhole import (
    ConstellationRenderer,
    draw_multiple_constellations,
)
from astrageek.helpers.geometry.geometry import (
    generate_small_circle,
    mag_to_radius,
    make_equatorial_grid_pinhole,
    make_pinhole_projection,
)


@dataclass
class ShotConditions:
    """Shooting conditions."""

    center_direction: NDArray  # direction in ECI (unit vector)
    tilt_angle: float  # angle in degrees on which we rotate a camera

    def __post_init__(self):
        # Normalize direction vector
        self.center_direction /= np.linalg.norm(self.center_direction)


@dataclass
class CameraConfig:
    """Physical camera configurations."""

    width: int  # frame width in pixels
    height: int  # frame height in pixels
    focal_length: float  # camera focal length in pixels

    @classmethod
    def from_fov_and_aspect(
        cls,
        fov_deg: float,
        aspect_ratio: float,
        height_pix: int,
    ):
        """Create CameraConfig from field of view and aspect ratio.

        :param fov_deg: Horizontal field of view in degrees
        :param aspect_ratio: width/height ratio
        :param height_pix: height in pixels
        """
        width_pix = int(height_pix * aspect_ratio)
        fov_rad = np.deg2rad(fov_deg)
        diagonal = np.sqrt(width_pix**2 + height_pix**2)
        focal_length_pix = (diagonal / 2) / np.tan(fov_rad / 2)
        return cls(
            width=width_pix, height=height_pix, focal_length=focal_length_pix
        )


@dataclass
class PinholeConfig:
    """Class of the pinhole projector configurations."""

    local_time: datetime = datetime.now()
    grid_theta_step: float = 10.0
    grid_phi_step: float = 10.0

    # Flags
    add_ecliptic: bool = False
    add_equator: bool = False
    add_galactic_equator: bool = False
    add_planets: bool = False
    add_ticks: bool = False
    add_equatorial_grid: bool = False
    use_dark_mode: bool = False
    add_constellations: bool = False
    add_constellations_names: bool = False


@dataclass
class ProjectionResult:
    """Result of pinhole projection."""

    stars: NDArray  # Structured array with star projections
    planets: NDArray  # Structured array with planet projections


@dataclass
class ConstellationConfig(object):

    constellations_list: Optional[List[str]] = (
        None  # If None, render all available
    )
    constellation_color: str = "gray"
    constellation_linewidth: float = 0.8
    constellation_alpha: float = 0.7
    constellation_color_map: Optional[Dict[str, str]] = None


class Pinhole(object):
    """Pinhole camera projector."""

    def __init__(
        self,
        shot_cond: ShotConditions,
        camera_cfg: CameraConfig,
        config: PinholeConfig,
        catalog: Catalog,
        planet_catalog: PlanetCatalog,
        constellation_config: Optional[ConstellationConfig] = None,
        constellations_renderer: Optional[ConstellationRenderer] = None,
    ):
        self.shot_cond = shot_cond
        self.camera_config = camera_cfg
        self.constellation_config = constellation_config
        self.constellations_renderer = (
            constellations_renderer or ConstellationRenderer()
        )
        self.config = config
        self.catalog = catalog
        self.planets_catalog = planet_catalog
        self._groups = {}
        self._star_projections = None
        self._planets_projections = None

    def _make_pinhole_views(
        self, data: NDArray, object_type: str = "star"
    ) -> NDArray:
        """Returns coordinates in picture plane (pinhole projection).

        :param data: objects equatorial coordinates
        :type data: NDArray
        :param object_type: star or planet
        :type object_type: str:  (Default value = 'star')

        :return:
            array with the structure like
            (magnitude, x, y, point size, right ascension, declination, id)
        :rtype: NDArray
        """

        VIEW_DTYPE = np.dtype(
            [
                ("v_mag", np.float32),
                ("x_pix", np.float32),
                ("y_pix", np.float32),
                ("size", np.float32),
                ("ra", np.float32),
                ("dec", np.float32),
                ("id", np.int32),
            ]
        )

        valid_mask, picture_coords = make_pinhole_projection(
            center_direction=self.shot_cond.center_direction,
            tilt_dec=self.shot_cond.tilt_angle,
            image_width=self.camera_config.width,
            image_height=self.camera_config.height,
            focal_length=self.camera_config.focal_length,
            data=data,
        )

        view_data = np.zeros(np.sum(valid_mask), dtype=VIEW_DTYPE)
        view_data["v_mag"] = data["v_mag"][valid_mask]
        view_data["x_pix"] = picture_coords["x_pix"][valid_mask]
        view_data["y_pix"] = picture_coords["y_pix"][valid_mask]
        view_data["ra"] = data["ra"][valid_mask]
        view_data["dec"] = data["dec"][valid_mask]
        view_data["size"] = mag_to_radius(
            magnitude=view_data["v_mag"],
            max_magnitude=self.catalog.constraints.max_magnitude,
            min_magnitude=self.catalog.constraints.min_magnitude,
        )

        if object_type == "star":
            view_data["id"] = data["hip_id"][valid_mask]
        elif object_type == "planet":
            view_data["id"] = data["planet_id"][valid_mask]

        return view_data

    def generate(
        self, constraints: Optional[CatalogConstraints] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Generate a pinhole projection image.

        :param constraints:
        :type constraints: CatalogConstraints | None

        :return: figure
        :rtype: Tuple[plt.Figure, plt.Axes]
        """

        # Make objects projections
        self.project(constraints=constraints)

        # Add ecliptic
        if self.config.add_ecliptic:
            self._add_ecliptic()

        # Add equator
        if self.config.add_equator:
            self._add_equator()

        # Add galactic equator
        if self.config.add_galactic_equator:
            self._add_galactic_equator()

        # Add equatorial grid
        if self.config.add_equatorial_grid:
            self._add_equatorial_grid()

        # Add constellation lines
        if self.config.add_constellations:
            self._add_constellations()

        # Add constellation names
        if self.config.add_constellations_names:
            self._add_constellations_names()

        # Put a legend to the bottom of the current axis
        self._create_grouped_legend()
        return self._fig, self._ax

    def project(self, constraints: CatalogConstraints):
        """Objects projection maker

        :param constraints: Catalog constraints
        :type constraints: CatalogConstraints
        """

        # Get stars data
        stars_data = self.catalog.get_stars(constraints)
        # From ECI to picture plane, make projection
        star_view_data = self._make_pinhole_views(
            data=stars_data, object_type="star"
        )
        self._star_projections = star_view_data
        # Create picture plane to place stars and other objects
        self._create_picture_plane()

        # Add planets
        if self.config.add_planets:
            planet_data = self.planets_catalog.get_planets(
                self.config.local_time
            )
            planet_view_data = self._make_pinhole_views(
                data=planet_data, object_type="planet"
            )
            self._planets_projections = planet_view_data
            self._add_planets()

    @property
    def projection_result(self) -> ProjectionResult:
        """Pair of the stars and planets projections."""
        return ProjectionResult(
            stars=self._star_projections, planets=self._planets_projections
        )

    def _create_picture_plane(self):
        """Creates the figure and axes.

        ВАЖНО: НЕ используем plt.style.use() — это глобальная операция,
        которая меняет состояние matplotlib для ВСЕХ последующих
        фигур в процессе. После вызова plt.style.use('dark_background') из
        игры, стерео-карты начинают рисоваться с чёрным фоном и невидимыми
        (чёрными) звёздами.

        Вместо этого сбрасываем глобальные стили через rcdefaults() и явно
        устанавливаем цвета только для текущей фигуры и осей.
        """
        # ── Сброс глобального состояния matplotlib
        # Это предотвращает «заражение» от предыдущих рендеров (например, игры
        # с dark_background могут сломать следующую стерео-карту).
        matplotlib.rcdefaults()

        bg_color = "black" if self.config.use_dark_mode else "white"
        star_color = "white" if self.config.use_dark_mode else "black"

        # ── Создаём фигуру с явными цветами
        self._fig = plt.figure(facecolor=bg_color)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(bg_color)

        self._ax.scatter(
            self._star_projections["x_pix"],
            self._star_projections["y_pix"],
            s=self._star_projections["size"],
            c=star_color,
        )

        self._ax.set_ylim(bottom=0, top=self.camera_config.height)
        self._ax.set_xlim(left=0, right=self.camera_config.width)

        if not self.config.add_ticks:
            self._ax.set_xticks([])
            self._ax.set_yticks([])

    def _add_planets(self):
        """Add planets to image."""

        for planet_data in self._planets_projections:
            if planet_data["v_mag"] < self.catalog.constraints.max_magnitude:
                planet = Planets(planet_data["id"])
                name = planet.name.capitalize()
                color = self.planets_catalog.get_planet_color(planet)
                scatter = self._ax.scatter(
                    planet_data["x_pix"],
                    planet_data["y_pix"],
                    c=color,
                    s=max(planet_data["size"] * 3, 0.5),
                    alpha=0.8,
                    linewidth=0.5,
                )
                self._groups["Planets"] = self._groups.get("Planets", []) + [
                    (scatter, name)
                ]

    def _add_ecliptic(self):
        """Add ecliptic on image."""

        RA = 270.0
        DEC = 66.5607

        ecliptic_eci_coords = generate_small_circle(
            spheric_normal_deg=np.array([90.0 - DEC, RA]),
            alpha_deg=90.0,
            num_points=1000,
        )
        valid_mask, picture_coords = make_pinhole_projection(
            center_direction=self.shot_cond.center_direction,
            tilt_dec=self.shot_cond.tilt_angle,
            image_width=self.camera_config.width,
            image_height=self.camera_config.height,
            focal_length=self.camera_config.focal_length,
            data=ecliptic_eci_coords,
        )
        (line,) = self._ax.plot(
            picture_coords["x_pix"][valid_mask],
            picture_coords["y_pix"][valid_mask],
            c="green",
            linewidth=1,
        )
        if np.sum(valid_mask) > 0:
            self._groups["Great circles"] = self._groups.get(
                "Great circles", []
            ) + [(line, "Ecliptic")]

    def _add_equator(self):
        """Add equator on image."""

        equator_eci_coords = generate_small_circle(
            spheric_normal_deg=np.array([0.0, 0.0]),
            alpha_deg=90.0,
            num_points=1000,
        )
        valid_mask, picture_coords = make_pinhole_projection(
            center_direction=self.shot_cond.center_direction,
            tilt_dec=self.shot_cond.tilt_angle,
            image_width=self.camera_config.width,
            image_height=self.camera_config.height,
            focal_length=self.camera_config.focal_length,
            data=equator_eci_coords,
        )
        (line,) = self._ax.plot(
            picture_coords["x_pix"][valid_mask],
            picture_coords["y_pix"][valid_mask],
            c="red",
            linewidth=1,
        )
        if np.sum(valid_mask) > 0:
            self._groups["Great circles"] = self._groups.get(
                "Great circles", []
            ) + [(line, "Celestial equator")]

    def _add_galactic_equator(self):
        """Add galactic equator on image."""

        # Galactical center
        RA = 192.85948
        DEC = 27.12825

        galactic_eci_coords = generate_small_circle(
            spheric_normal_deg=np.array([90.0 - DEC, RA]),
            alpha_deg=90.0,
            num_points=1000,
        )
        valid_mask, picture_coords = make_pinhole_projection(
            center_direction=self.shot_cond.center_direction,
            tilt_dec=self.shot_cond.tilt_angle,
            image_width=self.camera_config.width,
            image_height=self.camera_config.height,
            focal_length=self.camera_config.focal_length,
            data=galactic_eci_coords,
        )
        (line,) = self._ax.plot(
            picture_coords["x_pix"][valid_mask],
            picture_coords["y_pix"][valid_mask],
            c="blue",
            linewidth=1,
        )
        if np.sum(valid_mask) > 0:
            self._groups["Great circles"] = self._groups.get(
                "Great circles", []
            ) + [(line, "Galactic equator")]

    def _add_equatorial_grid(self):
        """Add equatorial grid on image."""

        grid = make_equatorial_grid_pinhole(
            center_direction=self.shot_cond.center_direction,
            tilt_dec=self.shot_cond.tilt_angle,
            image_width=self.camera_config.width,
            image_height=self.camera_config.height,
            focal_length=self.camera_config.focal_length,
            grid_step_ra=self.config.grid_phi_step,
            grid_step_dec=self.config.grid_theta_step,
        )
        self._ax.add_collection(grid)
        self._groups["Grids"] = self._groups.get("Grids", []) + [
            (grid, "Equatorial grid")
        ]

    def _add_constellations(self):
        """Add constellation line patterns to the projection."""

        if self.constellation_config.constellations_list is not None:
            constellations_to_render = (
                self.constellation_config.constellations_list
            )
        else:
            constellations_to_render = get_available_constellations()

        constellation_segments = (
            self.constellations_renderer.get_multiple_constellation_segments(
                constellations=constellations_to_render,
                stars=self._star_projections,
            )
        )

        if not constellation_segments:
            return

        # FIX ME: how to use lcs here?
        _ = draw_multiple_constellations(
            ax=self._ax,
            constellation_segments=constellation_segments,
            color=self.constellation_config.constellation_color,
            linewidth=self.constellation_config.constellation_linewidth,
            alpha=self.constellation_config.constellation_alpha,
            color_map=self.constellation_config.constellation_color_map,
            use_collection=True,
        )

    def _add_constellations_names(self):
        """Adds constellation names on skychart."""
        text_color = "white" if self.config.use_dark_mode else "gray"
        if self.constellation_config is None:
            constellations_to_render = get_available_constellations()
        elif self.constellation_config.constellations_list is not None:
            constellations_to_render = (
                self.constellation_config.constellations_list
            )
        else:
            constellations_to_render = get_available_constellations()

        for constellation in constellations_to_render:
            center = get_constellation_center(constellation)
            center_arr = np.array(center, dtype=np.float32)

            POINT_DTYPE = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
            center_struct = np.array(
                [(center_arr[0], center_arr[1], center_arr[2])],
                dtype=POINT_DTYPE,
            )

            valid_mask, picture_coords = make_pinhole_projection(
                center_direction=self.shot_cond.center_direction,
                tilt_dec=self.shot_cond.tilt_angle,
                image_width=self.camera_config.width,
                image_height=self.camera_config.height,
                focal_length=self.camera_config.focal_length,
                data=center_struct,
            )

            if np.sum(valid_mask) > 0:
                self._ax.annotate(
                    text=constellation,
                    xy=(
                        picture_coords["x_pix"][valid_mask][0],
                        picture_coords["y_pix"][valid_mask][0],
                    ),
                    xytext=(0, 0),
                    fontsize=7,
                    textcoords="offset points",
                    color=text_color,
                    ha="center",
                    va="center",
                )

    def _create_grouped_legend(self):
        """Create a grouped legend for the visualization."""
        if not self._groups:
            return

        group_items = [
            (title, items) for title, items in self._groups.items() if items
        ]
        if not group_items:
            return

        n_groups = len(group_items)
        if n_groups == 0:
            return

        n_columns = min(n_groups, 4)
        n_rows = (n_groups + n_columns - 1) // n_columns

        legend_height = n_rows * 0.08 + 0.05
        fig_size = self._fig.get_size_inches()
        new_height = fig_size[1] + legend_height * fig_size[1]
        self._fig.set_size_inches([fig_size[0], new_height])

        for i, (title, items) in enumerate(group_items):
            row = i // n_columns
            col = i % n_columns

            handles, labels = zip(*items)

            col_width = 0.8 / n_columns
            x_pos = 0.075 + col * col_width + col_width / 2
            y_pos = 0.15 + (n_rows - row - 1) * 0.08

            legend = self._ax.legend(
                handles,
                labels,
                title=title,
                loc="upper center",
                bbox_to_anchor=(x_pos, y_pos),
                bbox_transform=self._fig.transFigure,
                ncol=1,
                frameon=True,
                fancybox=True,
                borderaxespad=0.3,
                fontsize=8,
                handlelength=1.5,
            )

            legend.get_title().set_fontweight("bold")
            legend.get_title().set_fontsize(10)

            self._ax.add_artist(legend)
