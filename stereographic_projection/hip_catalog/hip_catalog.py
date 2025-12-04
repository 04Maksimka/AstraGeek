"""Implementation of Hipparchus catalog."""
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np


@dataclass
class EquatorialCoords(object):
    """Data class of equatorial coordinates."""

    right_ascension: float
    declination: float

    def __repr__(self):
        return f'(alpha:{np.rad2deg(self.right_ascension):8.2f}, delta:{np.rad2deg(self.declination):8.2f})'


@dataclass
class ECICoords(object):
    """Data class of ECI coordinates."""

    x: float
    y: float
    z: float

    def __repr__(self):
        return f'(x: {self.x:6.2f}, y: {self.y:6.2f}, z: {self.z:6.2f})'


@dataclass
class Star(object):
    """Class of a singular star."""

    v_mag: float                    # visual magnitude
    eq_coords: EquatorialCoords     # equatorial coordinates (alpha, delta)
    eci_coords: ECICoords = None    # eci coordinates (x, y, z)


    def __post_init__(self):
        self.eci_coords = self.eci

    def __repr__(self):
        return f'eq={self.eq_coords}, eci={self.eci_coords}, m={self.v_mag:7.2f}'

    @property
    def eci(self):
        """Initialize ECI coordinates."""
        # short names for usage
        alpha = self.eq_coords.right_ascension
        delta = self.eq_coords.declination

        # to spherical coordinate system
        x = np.cos(alpha) * np.cos(delta)
        y = np.sin(alpha) * np.cos(delta)
        z = np.sin(delta)

        assert abs(x ** 2 + y ** 2 + z ** 2 - 1) < 1e-15

        return ECICoords(x=x, y=y, z=z)


@dataclass
class Catalog(object):
    """
    Hipparchus catalog parsing.

    TODO: correct data import, path is need to be relative to the module, not to a main
    """

    mag_criteria: float = 5.5
    catalog_path: str = './hip_data.tsv'

    def get_data(self) -> NDArray[Star]:
        """
        Returns list of stars with given constrains.
        """

        # read data from file
        raw_data = np.genfromtxt(fname=self.catalog_path, delimiter=';',
                                 dtype=None,
                                 names=True,
                                 encoding='utf-8',
                                 missing_values='',
                                 filling_values=None)

        raw_data = raw_data[1:]     # without units
        mask = (raw_data['_RAJ2000'] != '') & (raw_data['_DEJ2000'] != '')
        clean_data = raw_data[mask]

        # make numpy array of Stars
        data = np.array([
            Star(v_mag=float(line['Vmag']),
                 eq_coords=EquatorialCoords(right_ascension=np.deg2rad(float(line['_RAJ2000'])),
                                            declination=np.deg2rad(float(line['_DEJ2000']))))
            for line in clean_data
        ], dtype=Star)

        return data

def main():
    """Main function."""

    # test catalog print
    catalog = Catalog()
    data = catalog.get_data()
    print(data)

if __name__ == "__main__":
    main()