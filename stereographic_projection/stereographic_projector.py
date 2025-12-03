"""Module implementing main stereographic projection."""
from dataclasses import dataclass
from datetime import datetime

from stereographic_projection.hip_catalog.hip_catalog import Catalog


@dataclass
class StereoProjCfg(object):
    """Class of configuration of the StereoProjector."""

    add_ecliptic: bool
    utc_time: datetime
    latitude: float
    longitude: float


class StereoProjector(object):
    """Class of the stereographic projector."""

    def __init__(self, cfg: StereoProjCfg, catalog: Catalog):
        raise NotImplemented

    def generate(self) -> None:
        """Generate a projection."""
        raise NotImplemented
