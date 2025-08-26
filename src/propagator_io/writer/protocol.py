from datetime import datetime
from typing import Protocol

import numpy as np
import numpy.typing as npt
import geopandas as gpd

from propagator.models import PropagatorStats


class RasterWriterProtocol(Protocol):
    def write_raster(
        self,
        variable: str,
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
        isochrones: gpd.GeoDataFrame,
        c_time: int,
        ref_date: datetime,
    ) -> None:
        ...