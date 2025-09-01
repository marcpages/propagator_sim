from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import rasterio as rio

from propagator.models import PropagatorOutput
from propagator_io.geo import reproject, trim_values, GeographicInfo
from pyproj import Proj
from rasterio.transform import Affine
from rasterio.crs import CRS

from .protocol import RasterWriterProtocol


def write_geotiff(
    filename: str | Path,
    values: npt.NDArray[np.floating] | npt.NDArray[np.integer],
    dst_trans: Affine,
    dst_prj: Proj,
    dtype: npt.DTypeLike = np.uint8,
) -> None:
    """Write a single-band GeoTIFF with provided transform and CRS."""
    dst_crs = CRS.from_proj4(dst_prj.srs)
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
    start_date: datetime
    output_folder: Path
    raster_variables_mapping: dict[
        str,
        Callable[[PropagatorOutput], npt.NDArray[np.floating]],
    ]
    geo_info: GeographicInfo
    dst_prj: Proj

    trim: bool = True

    def write_rasters(self, output: PropagatorOutput) -> None:
        for key, fun in self.raster_variables_mapping.items():
            values = fun(output)
            dst_trans = self.geo_info.trans
            dst_prj = self.geo_info.prj

            if self.geo_info.prj != self.dst_prj:
                dst_prj = self.dst_prj
                values, dst_trans = reproject(
                    values,
                    self.geo_info.trans,
                    self.geo_info.prj,
                    dst_prj,
                    trim=self.trim,
                )

            elif self.trim:
                values, dst_trans = trim_values(values, dst_trans)

            tiff_file = self.output_folder / f"{key}_{output.time}.tiff"
            # now it returns the RoS in m/h
            write_geotiff(tiff_file, values, dst_trans, dst_prj, values.dtype)
