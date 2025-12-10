"""Module implementing main stereographic projection."""
from dataclasses import dataclass
from datetime import datetime
from numpy.typing import NDArray
import numpy as np
from matplotlib import pyplot as plt
from stereographic_projection.helpers.pdf_helpers.figure2pdf import save_figure
from stereographic_projection.hip_catalog.hip_catalog import Catalog, Star


@dataclass
class StereoProjConfig(object):
    """Class of configuration of the StereoProjector."""

    add_ecliptic: bool
    utc_time: datetime
    latitude: float
    longitude: float


@dataclass
class HorizontalCoords(object):
    """Class of horizontal coordinates."""

    zenith_dist: float
    azimuth: float


@dataclass
class StarView(object):
    """Class of star view."""

    v_mag : float
    hor_coords : HorizontalCoords


@dataclass
class PointProjection(object):
    """Class of point projection."""

    radius : float
    rho: float
    phi: float


class StereoProjector(object):
    """Class of the stereographic projector."""

    def __init__(self, config: StereoProjConfig, catalog: Catalog):
        self.config = config
        self.catalog = catalog

    def generate(self, need_pdf=False) -> NDArray[PointProjection]:
        """
        Generate a projection.
        """

        # get catalog
        catalog_data = catalog.parse_data()
        # from equatorial to horizontal
        star_view_data = self._make_star_views(catalog_data)
        # make projections
        points_data = self._make_point_projections(star_view_data)

        self._stars_with_logo(points_data)

    def _make_point_projections(self, star_view_data: NDArray[StarView]) -> NDArray[PointProjection]:
        """
        Returns star point projections array.
        :param star_view_data: observed stars parameters in horizontal coordinates
        :return: star image point parameters

        TODO: check if it can be implemented using np.where or smth more fast and robust
        """

        points_data = np.array(
            [
                PointProjection(
                    radius=self._mag_to_radius(star_view.v_mag),
                    rho=2 * np.tan(star_view.hor_coords.zenith_dist / 2),
                    phi=star_view.hor_coords.azimuth
                )
                for star_view in star_view_data
                if star_view.hor_coords.zenith_dist < np.pi / 2
            ]
        )
        return points_data

    @staticmethod
    def _make_star_views(catalog_data: NDArray[Star]) -> NDArray[StarView]:
        """
        Returns horizontal coordinates.
        :param catalog_data: catalog of the stars
        :return: star view parameters

        TODO: implement equatorial using config parameters (longitude, latitude and local time)
        """
        star_view_data = np.array(
            [
                StarView(
                    v_mag=star.v_mag,
                    hor_coords=HorizontalCoords(
                        zenith_dist=(np.pi / 2 - star.eq_coords.declination),
                        azimuth=star.eq_coords.right_ascension
                    )
                )
                for star in catalog_data
            ]
        )
        return star_view_data

    @staticmethod
    def _mag_to_radius(magnitude: float) -> float:
        """
        Returns radius of the point corresponds to given star magnitude.
        :param magnitude: star magnitude
        :return: radius: star image radius

        TODO: implement this function
        """
        return 8 - magnitude

    def _stars_with_logo(self, data: NDArray[PointProjection], need_pdf: bool = False):
        """Example with local logo file."""
        # Create polar scatter plot
        fig, ax = self._create_polar_scatter(data)

        # Footer text
        footer = "© 2025 AstraGeek. All rights reserved."

        if need_pdf:
            save_figure(
                fig=fig,
                filename="polar_scatter_local_logo.pdf",
                logo_path="logo_astrageek.png",
                footer_text=footer,
                logo_position=(0.15, 0.97),
                text_position=(0.5, 0.01),
            )

        plt.show()

    def _create_polar_scatter(self, data: NDArray[PointProjection]):
        """Create a scatter plot in polar coordinates."""
        # Set up the figure with polar projection
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')

        sizes = [2 * point.radius for point in data]
        phi = [point.phi for point in data]
        rho = [point.rho for point in data]

        ax.scatter(
            phi,
            rho,
            c='black',
            s=sizes,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
        )

        ax.set_title("Polar Scatter Plot", va='bottom', fontsize=14, pad=20)
        ax.set_xlabel("Angle (θ)", labelpad=15)
        ax.grid(False)
        ax.set_yticks([])

        return fig, ax


if __name__ == "__main__":
    cfg = StereoProjConfig(add_ecliptic=True, utc_time=datetime(2021, 3, 1), latitude=55, longitude=-70)
    catalog = Catalog()
    proj = StereoProjector(cfg, catalog)

    proj.generate()