from stereographic_projection.helpers.time import get_sidereal_time
from datetime import datetime
import numpy as np

print(
    get_sidereal_time(
        longitude=np.deg2rad(90),
        local=datetime(
            year=1980,
            month=4,
            day=22,
            hour=14,
            minute=36,
            second=52
        )
    )
)

