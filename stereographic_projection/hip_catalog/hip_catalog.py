"""Implementation of Hipparchus catalog."""
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np


@dataclass
class EquatorialCoords(object):

    right_ascension: float
    declination: float

    def __init__(self, right_ascension: float, declination: float):
        self.right_ascension = right_ascension
        self.declination = declination

    def __repr__(self):
        return f'(alpha:{np.rad2deg(self.right_ascension):10.4f}, delta:{np.rad2deg(self.declination):10.4f})'


@dataclass
class ECICoords(object):

    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f'(x: {self.x:7.4f}, y: {self.y:7.4f}, z: {self.z:7.4f})'


@dataclass
class Star(object):
    """Class of a singular star."""

    v_mag: float                    # visual magnitude
    eq_coords: EquatorialCoords     # equatorial coordinates (alpha, delta)
    eci_coords: ECICoords           # eci coordinates (x, y, z)


    def __init__(self, v_mag: float, right_ascension: float, declination: float):
        self.eq_coords = EquatorialCoords(right_ascension=right_ascension,
                                          declination=declination)
        self.init_eci()
        self.v_mag = v_mag

    def __repr__(self):
        return f'eq={self.eq_coords}, eci={self.eci_coords}, m={self.v_mag}'

    def init_eci(self):
        """Initialize ECI coordinates."""
        # short names for usage
        alpha = self.eq_coords.right_ascension
        delta = self.eq_coords.declination

        # to spherical coordinate system
        x = np.cos(alpha) * np.cos(delta)
        y = np.sin(alpha) * np.cos(delta)
        z = np.sin(delta)

        self.eci_coords = ECICoords(x=x, y=y, z=z)


@dataclass
class Catalog(object):
    """Hipparchus catalog."""

    mag_criteria: float = 5.5
    catalog_path: str = './hip_catalog/hip_data.tsv'

    def get_data(self) -> NDArray[Star]:
        """
        Returns list of stars with given constrains.

        TODO: add correct type assigning in the data array
        """

        # read data from file
        raw_data = np.genfromtxt(self.catalog_path, delimiter=';',
                                 dtype=None,
                                 names=True,
                                 encoding='utf-8',
                                 missing_values='',
                                 filling_values=None)

        # masking stars with missing coordinates
        mask = []
        for row in raw_data:
            if (row['_RAJ2000'] is not None and row['_RAJ2000'] != '') \
                    and (row['_DEJ2000'] is not None and row['_DEJ2000'] != ''):
                mask.append(True)
            else:
                mask.append(False)
        clean_data = raw_data[mask]

        # make NDarray of Stars
        data = np.array([
            Star(v_mag=float(line['Vmag']),
                 right_ascension=np.deg2rad(float(line['_RAJ2000'])),
                 declination=np.deg2rad(float(line['_DEJ2000'])))
            for line in clean_data[1:]
        ], dtype=Star)

        return data
