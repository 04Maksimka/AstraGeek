"""Module implementing main stereographic projection."""
from dataclasses import dataclass
from datetime import datetime
import dateutil.tz
from numpy.typing import NDArray
import numpy as np
from matplotlib import pyplot as plt
from stereographic_projection.hip_catalog.hip_catalog import Catalog, Star, ECICoords


@dataclass
class StereoProjConfig(object):
    """Class of configuration of the StereoProjector."""

    add_ecliptic: bool
    utc_time: datetime
    latitude: float
    longitude: float

    def __post_init__(self):
        self.latitude *= np.pi / 180.0
        self.longitude *= np.pi / 180.0


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
    """
    Class of point projection.
    radius: star image circle radius
    phi: star image azimuth
    rho: star image polar radius
    """

    radius : float
    rho: float
    phi: float


class StereoProjector(object):
    """Class of the stereographic projector."""

    def __init__(self, config: StereoProjConfig, catalog: Catalog):
        self.config = config
        self.catalog = catalog

    def generate(self) -> plt.Figure:
        """
        Generate a projection.
        """

        # Get catalog
        catalog_data = self.catalog.parse_data()
        # From equatorial to horizontal
        star_view_data = self._make_star_views(catalog_data)
        # Make projections
        points_data = self._make_point_projections(star_view_data)
        # Make figure with projections
        fig, _ = self._create_polar_scatter(points_data)

        return fig

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
                if star_view.hor_coords.zenith_dist <= np.pi / 2
            ]
        )
        return points_data

    def _make_star_views(self, catalog_data: NDArray[Star]) -> NDArray[StarView]:
        """
        Returns horizontal coordinates.
        :param catalog_data: catalog of the stars
        :return: star view parameters

        TODO: it's awful, but working, need to refactor. need to calculate sidereal_time accurately
        """

        eci_coords = [star.eci_coords for star in catalog_data]

        # Shift from UTC
        timeshift = self.config.longitude * 3600 / 15.0

        # Just time from 1970-1-1
        sidereal_time = self.config.utc_time.replace(tzinfo=dateutil.tz.tzoffset(None, timeshift)).timestamp() * np.pi / (12 * 3600)

        # Rotate ECI (XYZ) to "cartesian" equatorial system (X'Y'Z'),
        # so Z' is directed to the North celestial pole, X' --- to the West point
        latitude = self.config.latitude
        rotation_matrix = np.array(
            [
                [np.sin(sidereal_time), np.sin(latitude) * np.cos(sidereal_time), -np.cos(latitude) * np.cos(sidereal_time)],
                [-np.cos(sidereal_time), np.sin(latitude) * np.sin(sidereal_time), -np.cos(latitude) * np.sin(sidereal_time)],
                [0.0, np.cos(latitude), np.sin(latitude)]
            ]
        )
        cartesian_hor_coords = np.array([rotation_matrix @ np.array(list(eci_coord)) for eci_coord in eci_coords])

        # 0, 1, 2 is x, y, z below
        azimuths = -np.atan2(cartesian_hor_coords[:, 1], cartesian_hor_coords[:, 0])
        zeniths = np.arccos(cartesian_hor_coords[:, 2])
        magnitudes = np.array([star.v_mag for star in catalog_data])

        star_view_data = np.array(
            [
                StarView(
                    v_mag=m,
                    hor_coords=HorizontalCoords(
                        zenith_dist=z,
                        azimuth=a
                    )
                )
                for m, z, a in zip(magnitudes, zeniths, azimuths)
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

        return np.max([6 - magnitude, 0])

    @staticmethod
    def _create_polar_scatter(data: NDArray[PointProjection]):
        """Create a scatter plot in polar coordinates."""
        # Set up the figure with polar projection
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='polar')

        # Get parameters from projections data array
        sizes = [point.radius for point in data]
        phi = [point.phi for point in data]
        rho = [point.rho for point in data]

        # Make scatter
        ax.scatter(
            phi,
            rho,
            c='black',
            s=sizes,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
        )

        ax.set_title("Skychart", va='bottom', fontsize=14, pad=20)
        # ax.set_xlabel("Angle (θ)", labelpad=15)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim((0.0, 2.0))

        return fig, ax
