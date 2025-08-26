
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio as rio
from pyproj import CRS
from rasterio.transform import Affine

from .protocol import RasterWriterProtocol


def write_geotiff(
    filename: str|Path,
    values: npt.NDArray[np.floating] | npt.NDArray[np.integer],
    dst_trans,
    dst_crs,
    dtype: npt.DTypeLike = np.uint8,
) -> None:
    """Write a single-band GeoTIFF with provided transform and CRS."""
    with rio.Env():
        with rio.open(
            filename,
            "w",
            driver="GTiff",
            width=values.shape[1],
            height=values.shape[0],
            count=1,
            dtype=dtype,
            nodata=0,
            transform=dst_trans,
            crs=dst_crs,
        ) as f:
            f.write(values.astype(dtype), indexes=1)


@dataclass
class GeoTiffWriter(RasterWriterProtocol):
    dst_trans: Affine
    dst_crs: CRS
    output_folder: Path
    prefix: str
    
    def write_raster(
        self, 
        values: npt.NDArray[np.floating] | npt.NDArray[np.integer],
        c_time: int,
        ref_date: datetime,
    ) -> None:


        tiff_file = self.output_folder / f"{self.prefix}_{c_time}.tiff"
        # now it returns the RoS in m/h
        write_geotiff(tiff_file, values, self.dst_trans, self.dst_crs, values.dtype)
