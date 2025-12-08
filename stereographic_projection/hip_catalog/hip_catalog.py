"""Implementation of Hipparchus catalog."""
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import pathlib


@dataclass
class EquatorialCoords(object):
    """Data class of equatorial coordinates."""

    right_ascension: float
    declination: float

    def __repr__(self):
        return f'(alpha:{np.rad2deg(self.right_ascension):8.2f}, ' \
               f'delta:{np.rad2deg(self.declination):8.2f})'


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

    def __repr__(self):
        return f'eq={self.eq_coords}, eci={self.eci_coords}, m={self.v_mag:7.2f}'

    @property
    def eci_coords(self) -> ECICoords:
        """Initialize ECI coordinates."""

        # short names for usage
        alpha = self.eq_coords.right_ascension
        delta = self.eq_coords.declination

        # to spherical coordinate system
        x = np.cos(alpha) * np.cos(delta)
        y = np.sin(alpha) * np.cos(delta)
        z = np.sin(delta)

        return ECICoords(x=x, y=y, z=z)


@dataclass
class Catalog(object):
    """
    Hipparchus catalog.
    """

    mag_criteria: float = 5.5
    catalog_name: str = 'hip_data.tsv'

    def parse_data(self) -> NDArray[Star]:
        """
        Returns list of stars with given constrains.
        """

        catalog_path = pathlib.Path(__file__).parent.absolute() / self.catalog_name

        # read data from file
        raw_data = np.genfromtxt(
            fname=catalog_path,
            delimiter=';',
            dtype=None,
            names=True,
            encoding='utf-8',
            missing_values='',
            filling_values=None
        )

        cleaned_data = self._clean_raw_data(raw_data)

        # make numpy array of Stars
        data = np.array(
            [
                Star(
                    v_mag=float(line['Vmag']),
                    eq_coords=EquatorialCoords(
                        right_ascension=np.deg2rad(float(line['_RAJ2000'])),
                        declination=np.deg2rad(float(line['_DEJ2000']))
                    )
                )
                for line in cleaned_data
            ],
            dtype=Star
        )
        return data

    @staticmethod
    def _clean_raw_data(raw_data):
        """
        function removing units and rows with missing values
        in right ascension and declinations columns

        :param raw_data: source catalog data
        :return: cleaned catalog data
        """
        raw_data = raw_data[1:]
        mask = (raw_data['_RAJ2000'] != '') & (raw_data['_DEJ2000'] != '')
        clean_data = raw_data[mask]

        return clean_data


def main():
    """Main function."""

    # test catalog print
    catalog = Catalog()
    data = catalog.parse_data()
    print(data)

if __name__ == "__main__":
    main()