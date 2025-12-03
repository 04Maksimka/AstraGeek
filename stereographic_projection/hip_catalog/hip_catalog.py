"""Implementation of Hipparchus catalog."""
from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class Star(object):
    """Class of a singular star."""
    v_mag: float
    eci_coords: NDArray


@dataclass
class Catalog(object):
    """Hipparchus catalog."""
    mag_criteria: float = 5.5

    def get_data(self) -> NDArray[Star]:
        """Returns list of stars with given constrains."""
        raise NotImplemented
