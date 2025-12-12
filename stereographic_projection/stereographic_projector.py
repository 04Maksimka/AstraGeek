"""Module implementing main stereographic projection."""
from dataclasses import dataclass
from datetime import datetime
from numpy.typing import NDArray
import numpy as np
from matplotlib import pyplot as plt
from stereographic_projection.hip_catalog.hip_catalog import Catalog, Star
from stereographic_projection.helpers.geometry import get_horizontal_coords, make_point_projections, StarView, PointProjection, HorizontalCoords


@dataclass
class StereoProjConfig(object):
    """Class of configuration of the StereoProjector."""

    add_ecliptic: bool
    local_time: datetime
    latitude: float
    longitude: float

    def __post_init__(self):
        self.latitude = np.deg2rad(self.latitude)
        self.longitude = np.deg2rad(self.longitude)



class StereoProjector(object):
    """Class of the stereographic projector."""

    def __init__(self, config: StereoProjConfig, catalog: Catalog):
        self.config = config
        self.catalog = catalog

    def generate(self) -> plt.Figure:
        """
        Generate a projection.
        :return: figure
        """

        # Get catalog
        catalog_data = self.catalog.parse_data()
        # From equatorial to horizontal
        star_view_data = self._make_star_views(catalog_data)
        # Make projections
        points_data = make_point_projections(star_view_data, self.catalog.mag_criteria)
        # Make figure with projections
        fig, _ = self._create_polar_scatter(points_data)

        return fig


    def _make_star_views(self, catalog_data: NDArray[Star]) -> NDArray[StarView]:
        """
        Returns horizontal coordinates.
        :param catalog_data: catalog of the stars
        :return: star view parameters
        """

        _, azimuths, zeniths = get_horizontal_coords(self.config.__dict__, catalog_data=catalog_data)

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
    def _create_polar_scatter(data: NDArray[PointProjection]):
        """
        Create a scatter plot in polar coordinates.
        :param data: projection points to place on figure
        """

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
        ax.set_xlabel("Angle (θ)", labelpad=15)
        ax.grid(True)
        # ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim((0.0, 2.0))

        return fig, ax
