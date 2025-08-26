from typing import Tuple
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import rasterio as rio
from pyproj import Proj
from rasterio import enums, transform, warp, CRS
from rasterio.transform import Affine


def reproject(
    values: npt.NDArray[np.floating],
    src_trans: Affine,
    src_prj: Proj,
    dst_prj: Proj,
    trim: bool = True,
) -> Tuple[npt.NDArray[np.floating], Affine]:
    """Reproject a raster (optionally trimmed) to a different CRS.

    Returns `(dst, dst_trans)` with the new raster array and affine transform.
    """
    if trim:
        values, src_trans = trim_values(values, src_trans)

    rows, cols = values.shape
    (west, east), (north, south) = transform.xy(
        src_trans, [0, rows], [0, cols], offset="ul"
    )

    src_crs = CRS.from_proj4(src_prj.srs)
    dst_crs = CRS.from_proj4(dst_prj.srs)

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


@dataclass(frozen=True)
class GeographicInfo:
    prj: Proj
    trans: transform.Affine
    bounds: tuple[float, float, float, float]
    shape: tuple[int, int]

    def get_stepx_stepy(self) -> tuple[float, float]:
        step_x = (self.bounds[2] - self.bounds[0]) / self.shape[1]
        step_y = (self.bounds[3] - self.bounds[1]) / self.shape[0]
        return step_x, step_y

    @staticmethod
    def from_bounds(
        west: float,
        south: float,
        east: float,
        north: float,
        rows: int,
        cols: int,
        zone: int,
        proj: str = "utm",
        datum: str = "WGS84",
    ) -> "GeographicInfo":
        """
        Create a GeographicInfo object from bounds and projection parameters.
        :param west: West bound
        :param south: South bound
        :param east: East bound
        :param north: North bound
        :param rows: Number of rows
        :param cols: Number of columns
        :param zone: UTM zone number
        :param proj: Projection type (default is UTM)
        :param datum: Datum (default is WGS84)
        :return: GeographicInfo object
        """
        prj = Proj(proj=proj, zone=zone, datum=datum)
        trans = transform.from_bounds(west, south, east, north, cols, rows)
        bounds = (west, south, east, north)
        shape = (rows, cols)

        return GeographicInfo(prj=prj, trans=trans, bounds=bounds, shape=shape)

    @staticmethod
    def from_file(rio_file: rio.DatasetReader) -> "GeographicInfo":
        """
        Create a GeographicInfo object from a raster file.
        :param file: Path to the raster file
        :return: GeographicInfo object
        """
        bounds = rio_file.bounds
        cols, rows = rio_file.width, rio_file.height
        west, south, east, north = (
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )

        proj = rio_file.crs.to_proj4()
        transform = rio_file.transform

        prj = Proj(proj)
        return GeographicInfo(
            prj=prj,
            trans=transform,
            bounds=(west, south, east, north),
            shape=(rows, cols),
        )
