"""Module with astronomical geometrical functions."""

from stereographic_projection.hip_catalog.hip_catalog import Star
from stereographic_projection.helpers.time import vequinox_hour_angle
from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass


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


def get_horizontal_coords(config: dict, catalog_data: NDArray[Star]):
    """
    Get horizontal coordinates from catalog data.
    :param config: place configuration
    :param catalog_data: star catalog data
    :return: horizontal coordinates, zenith distances and azimuths
    """

    # Get ECI star coordinates
    eci_coords = np.array([np.array(list(star.eci_coords)) for star in catalog_data]).T

    # Calculate local sidereal time
    sidereal_time = vequinox_hour_angle(
        longitude=config['longitude'],
        local=config['local_time']
    )

    # Rotate ECI (XYZ) to "cartesian" equatorial system (X'Y'Z'),
    # so Z' is directed to the North celestial pole, X' --- to the West point
    latitude = config['latitude']
    sl = np.sin(latitude)
    cl = np.cos(latitude)
    st = np.sin(sidereal_time)
    ct = np.cos(sidereal_time)
    rotation_matrix = np.array(
        [
            [st, -ct, 0],
            [sl * ct, sl * st, -cl],
            [cl * ct, cl * st, sl]
        ]
    )
    cartesian_hor_coords = rotation_matrix @ eci_coords

    # 0, 1, 2 is x, y, z below
    azimuths = np.atan2(cartesian_hor_coords[0, :], cartesian_hor_coords[1, :]) - np.pi / 2
    zeniths = np.arccos(cartesian_hor_coords[2, :])

    return cartesian_hor_coords, azimuths, zeniths


def mag_to_radius(magnitude: float, mag_criteria: float) -> float:
    """
    Returns radius of the point corresponds to given star magnitude.
    :param magnitude: star magnitude
    :param mag_criteria: max radius criteria
    :return: radius: star image radius
    """
    return 1.5*np.max([mag_criteria - magnitude, 0])


def make_point_projections(star_view_data: NDArray[StarView], mag_criteria: float) -> NDArray[PointProjection]:
    """
    Returns star point projections array.
    :param star_view_data: observed stars parameters in horizontal coordinates
    :param mag_criteria: max star catalog magnitude
    :return: star image point parameters

    TODO: check if it can be implemented using np.where or smth more fast and robust
    """

    points_data = np.array(
        [
            PointProjection(
                radius=mag_to_radius(star_view.v_mag, mag_criteria),
                rho=2 * np.tan(star_view.hor_coords.zenith_dist / 2),
                phi=star_view.hor_coords.azimuth
            )
            for star_view in star_view_data
            if star_view.hor_coords.zenith_dist <= np.pi / 2
        ]
    )
    return points_data