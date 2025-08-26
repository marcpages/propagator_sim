
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio as rio
from pyproj import CRS
from rasterio import enums, transform, warp
from rasterio.transform import Affine

from .protocol import RasterWriterProtocol




def reproject(
    values: npt.NDArray[np.floating],
    src_trans,
    src_crs,
    dst_crs,
    trim: bool = True,
):
    """Reproject a raster (optionally trimmed) to a different CRS.

    Returns `(dst, dst_trans)` with the new raster array and affine transform.
    """
    if trim:
        values, src_trans = trim_values(values, src_trans)

    rows, cols = values.shape
    (west, east), (north, south) = transform.xy(
        src_trans, [0, rows], [0, cols], offset="ul"
    )

    with rio.Env():
        dst_trans, dw, dh = warp.calculate_default_transform(
            src_crs=src_crs,
            dst_crs=dst_crs,
            width=cols,
            height=rows,
            left=west,
            bottom=south,
            right=east,
            top=north,
            resolution=None,
        )
        dst = np.empty(shape=(dh, dw))  # type: ignore # warp calculate_default_transform returns inconsistent types

        warp.reproject(
            source=np.ascontiguousarray(values),
            destination=dst,
            src_crs=src_crs,
            dst_crs=dst_crs,
            dst_transform=dst_trans,
            src_transform=src_trans,
            resampling=enums.Resampling.nearest,
            num_threads=1,
        )

    return dst, dst_trans


def trim_values(
    values: npt.NDArray[np.floating],
    src_trans,
):
    """Trim a values raster around non-zero area and return new transform."""
    rows, cols = values.shape
    min_row, max_row = int(rows / 2 - 1), int(rows / 2 + 1)
    min_col, max_col = int(cols / 2 - 1), int(cols / 2 + 1)

    v_rows = np.where(values.sum(axis=1) > 0)[0]
    if len(v_rows) > 0:
        min_row, max_row = v_rows[0] - 1, v_rows[-1] + 2

    v_cols = np.where(values.sum(axis=0) > 0)[0]
    if len(v_cols) > 0:
        min_col, max_col = v_cols[0] - 1, v_cols[-1] + 2

    trim_values = values[min_row:max_row, min_col:max_col]
    rows, cols = trim_values.shape

    (west, east), (north, south) = transform.xy(
        src_trans, [min_row, max_row], [min_col, max_col], offset="ul"
    )
    trim_trans = transform.from_bounds(west, south, east, north, cols, rows)
    return trim_values, trim_trans

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
