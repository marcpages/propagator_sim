from datetime import datetime
from typing import Protocol

import numpy as np
import numpy.typing as npt

from propagator.models import PropagatorStats


class RasterWriterProtocol(Protocol):
    def write_raster(
        self,
        values: npt.NDArray[np.floating] | npt.NDArray[np.integer],
        c_time: int,
        ref_date: datetime,
    ) -> None:
        ...

class MetadataWriterProtocol(Protocol):
    def write_metadata(
        self,
        stats: PropagatorStats,
        c_time: int,
        ref_date: datetime,
    ) -> None:
        ...

class IsochronesWriterProtocol(Protocol):
    def write_isochrones(
        self,
        isochrones: dict[int, npt.NDArray[np.floating]],
        c_time: int,
        ref_date: datetime,
    ) -> None:
        ...