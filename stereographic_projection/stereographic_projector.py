"""Module implementing main stereographic projection."""
from dataclasses import dataclass


@dataclass
class StereoProjCfg(object):
    """Class of configuration of the StereoProjector."""
    add_ecliptic: bool


class StereoProjector(object):
    """Class of the stereographic projector."""

    def __init__(self, zenith_coords):
        raise NotImplemented

    def generate(self):
        """Generate a projection."""
        raise NotImplemented
