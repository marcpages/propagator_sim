from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.signal.signaltools import medfilt2d
from shapely.geometry import LineString, MultiLineString, mapping, shape

from propagator.models import PropagatorOutput
from propagator_io.geo import reproject, GeographicInfo
from pyproj import Proj

from propagator_io.writer.protocol import IsochronesWriterProtocol

TIME_TAG = "time"


def smooth_linestring(linestring, smooth_sigma):
    """
    Uses a gauss filter to smooth out the LineString coordinates.
    """
    smooth_x = np.array(gaussian_filter1d(linestring.xy[0], smooth_sigma))  # type: ignore # gaussian_filter1d has None typing in library
    smooth_y = np.array(gaussian_filter1d(linestring.xy[1], smooth_sigma))  # type: ignore

    # close the linestring
    smooth_y[-1] = smooth_y[0]
    smooth_x[-1] = smooth_x[0]

    smoothed_coords = np.hstack((smooth_x, smooth_y))
    smoothed_coords = zip(smooth_x, smooth_y)

    linestring_smoothed = LineString(smoothed_coords)

    return linestring_smoothed


def extract_isochrone(
    values: npt.NDArray[np.floating],
    transf: Affine,
    thresholds=[0.5, 0.75, 0.9],
    med_filt_val=9,
    min_length=0.0001,
    smooth_sigma=0.8,
    simp_fact=0.00001,
) -> dict[float, MultiLineString]:
    """
    extract isochrone from the propagation probability map values at the probanilities thresholds,
     applying filtering to smooth out the result
    :param values:
    :param transf:
    :param thresholds:
    :param med_filt_val:
    :param min_length:
    :param smooth_sigma:
    :param simp_fact:
    :return:
    """

    # if the dimension of the burned area is low, we do not filter it
    if np.sum(values > 0) <= 100:
        filt_values = values
    else:
        filt_values = medfilt2d(values, med_filt_val)  # type: ignore # medfilt2d has None typing in library
    results = {}

    for t in thresholds:
        over_t_ = (filt_values >= t).astype("uint8")
        over_t = binary_dilation(binary_erosion(over_t_).astype("uint8")).astype(  # type: ignore #binary_erosion has None typing in library
            "uint8"
        )
        if np.any(over_t):
            for s, v in shapes(over_t, transform=transf):
                sh = shape(s)

                ml = [
                    smooth_linestring(
                        interior_line, smooth_sigma
                    )  # .simplify(simp_fact)
                    for interior_line in sh.interiors  # type: ignore # sh.interiors is missing in typing
                    if interior_line.length > min_length
                ]

                results[t] = MultiLineString(ml)

    return results


def save_isochrones(results, filename: str, format: str = "geojson") -> None:
    """Serialize extracted isochrones to GeoJSON or ESRI Shapefile."""
    if format == "shp":
        schema = {
            "geometry": "MultiLineString",
            "properties": {"value": "float", TIME_TAG: "int"},
        }
        # Write a new Shapefile
        with fiona.open(filename, "w", "ESRI Shapefile", schema) as c:
            for t in results:
                for p in results[t]:
                    if results[t][p].type == "MultiLineString":
                        c.write(
                            {
                                "geometry": mapping(results[t][p]),
                                "properties": {"value": p, TIME_TAG: t},
                            }
                        )

    if format == "geojson":
        import json

        features = []
        geojson_obj = dict(type="FeatureCollection", features=features)
        for t in results:
            for p in results[t]:
                if results[t][p].geom_type == "MultiLineString":
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": mapping(results[t][p]),
                            "properties": {"value": p, TIME_TAG: t},
                        }
                    )
        with open(filename, "w") as f:
            f.write(json.dumps(geojson_obj))


@dataclass
class IsochronesGeoJSONWriter(IsochronesWriterProtocol):
    start_date: datetime
    output_folder: Path
    prefix: str
    geo_info: GeographicInfo
    dst_prj: Proj


    thresholds: list[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])
    med_filt_val: int = 9
    min_length: float = 0.0001
    smooth_sigma: float = 0.8
    simp_fact: float = 0.00001

    _isochrones: gpd.GeoDataFrame = field(init=False)

    def __post_init__(self):
        self._isochrones = gpd.GeoDataFrame(
            crs=self.dst_prj.to_proj4(),
            columns=["geometry", "date"],
            geometry="geometry",
            index=pd.MultiIndex.from_arrays([[], []], names=["threshold", "time"]),
        )

    def write_isochrones(self, output: PropagatorOutput) -> None:
        json_file = self.output_folder / f"{self.prefix}_{output.time}.json"
        ref_date = self.ref_date(output)

        values = output.fire_probability
        dst_trans = self.geo_info.trans
        prj = self.geo_info.prj
        if prj != self.dst_prj:
            values, dst_trans = reproject(
                values,
                self.geo_info.trans,
                self.geo_info.prj,
                self.dst_prj,
            )

        isochrones_geoms = extract_isochrone(
            values,
            dst_trans,
            thresholds=self.thresholds,
            med_filt_val=self.med_filt_val,
            min_length=self.min_length,
            smooth_sigma=self.smooth_sigma,
            simp_fact=self.simp_fact,
        )

        # iterate over threshold/geometry and add it to the _isochrones
        for threshold, geom in isochrones_geoms.items():
            self._isochrones = gpd.GeoDataFrame(
                pd.concat(
                    [
                        self._isochrones,
                        pd.DataFrame(
                            {
                                "geometry": geom,
                                "date": ref_date.isoformat(),
                            },
                            index=pd.MultiIndex.from_tuples(
                                [(threshold, output.time)], names=["threshold", "time"]
                            ),
                        ),
                    ]
                ),
                geometry="geometry",
            )

        self._isochrones.to_file(json_file, driver="GeoJSON")
