from __future__ import annotations

from typing import List, Optional, Union, Any, Sequence
import re
import math
from enum import Enum
from pydantic import BaseModel, ConfigDict, model_validator
from typing import Tuple
import numpy as np
from pyproj import CRS, Transformer
from rasterio.features import rasterize
import rasterio.enums as rio_enums

from propagator_io.geo import GeographicInfo

DEFAULT_EPSG_GEOMETRY = 4326  # default to WGS84


# ---- geometry models --------------------------------------------------------
class GeometryKind(str, Enum):
    POINT = "point"
    LINE = "line"
    POLYGON = "polygon"


class GeometryBase(BaseModel):
    """Common fields/behavior for all geometries."""
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,   # <-- allow np.ndarray, CRS
    )

    kind: GeometryKind
    ys: np.ndarray
    xs: np.ndarray
    crs: CRS = CRS.from_epsg(4326)  # default to WGS84

    @model_validator(mode="before")
    @classmethod
    def _check_geometry(cls, data: dict) -> dict:
        if len(data["xs"]) != len(data["ys"]):
            raise ValueError("coordinates must have the same length")
        return data

    # to be implemented by subclasses: returns (xs, ys)
    def _x_y(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.xs, self.ys


class GeoPoint(GeometryBase):
    kind: GeometryKind = GeometryKind.POINT


class GeoLine(GeometryBase):
    kind: GeometryKind = GeometryKind.LINE

    @model_validator(mode="before")
    @classmethod
    def _check_line(cls, data: dict):
        if len(data["xs"]) < 2:
            raise ValueError("Line must have at least 2 points")
        return data


class GeoPolygon(GeometryBase):
    kind: GeometryKind = GeometryKind.POLYGON

    @model_validator(mode="before")
    @classmethod
    def _check_poly(cls, data):
        if len(data["xs"]) < 4:  # because the polygon must be closed
            raise ValueError("Polygon must have at least 4 points")
        if not (math.isclose(data["xs"][0], data["xs"][-1]) and
                math.isclose(data["ys"][0], data["ys"][-1])):
            raise ValueError("Polygon must be closed")
        return data


# super-class for all geometry types
Geometry = Union[GeoPoint, GeoLine, GeoPolygon]


# --- geometry parsing ----

_POINT_RE = re.compile(
    r"""^POINT:\[\s*
        (?P<y>-?\d+(?:\.\d+)?)\s*;\s*
        (?P<x>-?\d+(?:\.\d+)?)\s*
        \]\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

_SERIES_RE = re.compile(
    r"""^(?P<kind>LINE|POLYGON):\[\s*
        (?P<ys>-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?)+)\s*
        \];\[\s*
        (?P<xs>-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?)+)\s*
        \]\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _split_floats(s: str) -> List[float]:
    return [float(x) for x in s.strip().split() if x.strip()]


def parse_geometry_string(s: str, epsg: int) -> Geometry:
    """Parse POINT/LINE/POLYGON strings into geometry objects (with CRS)."""
    s = s.strip()
    crs = CRS.from_epsg(epsg)
    m_pt = _POINT_RE.match(s)
    if m_pt:
        y = float(m_pt.group("y"))
        x = float(m_pt.group("x"))
        return GeoPoint(
            ys=np.asarray([y], dtype=float),
            xs=np.asarray([x], dtype=float),
            crs=crs,
        )
    m_series = _SERIES_RE.match(s)
    if m_series:
        kind = m_series.group("kind").upper()
        ys_list = _split_floats(m_series.group("ys"))
        xs_list = _split_floats(m_series.group("xs"))
        if len(ys_list) != len(xs_list):
            raise ValueError(f"{kind}: y/x counts differ \
                ({len(ys_list)} vs {len(xs_list)})")
        ys_arr = np.asarray(ys_list, dtype=float)
        xs_arr = np.asarray(xs_list, dtype=float)
        if kind == "LINE":
            return GeoLine(ys=ys_arr, xs=xs_arr, crs=crs)
        elif kind == "POLYGON":
            return GeoPolygon(ys=ys_arr, xs=xs_arr, crs=crs)
        else:
            raise ValueError(f"Unsupported geometry kind: {kind!r}")
    raise ValueError(f"Unsupported geometry string: {s!r}")


def parse_geometry_list(v: Any, allowed: set[str],
                        epsg: int) -> Optional[List[Geometry]]:
    if v is None:
        return None
    if not isinstance(v, list):
        raise ValueError("expected a list")
    out: List[Geometry] = []
    for item in v:
        if isinstance(item, str):
            g = parse_geometry_string(item, epsg)
        elif isinstance(item, dict) and "kind" in item:
            k = str(item["kind"]).lower()
            if k == "point":
                g = GeoPoint(**item)
            elif k == "line":
                g = GeoLine(**item)
            elif k == "polygon":
                g = GeoPolygon(**item)
            else:
                raise ValueError(f"\
                    unsupported kind {item['kind']!r}")
        else:
            raise ValueError(f"unsupported entry {item!r}")
        # check on allowed geometry types
        if g.__class__ is GeoPoint and "point" not in allowed:
            raise ValueError("POINT not allowed")
        if g.__class__ is GeoLine and "line" not in allowed:
            raise ValueError("LINE not allowed")
        if g.__class__ is GeoPolygon and "polygon" not in allowed:
            raise ValueError("POLYGON not allowed")
        out.append(g)
    return out


# --- rasterization ---

def _reproject_xy(
    xs: np.ndarray,
    ys: np.ndarray,
    src_crs: CRS,
    dst_crs: CRS,
) -> Tuple[np.ndarray, np.ndarray]:
    tfm = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    X, Y = tfm.transform(xs, ys)
    return X, Y


def geometry_to_geojson(
    g: Geometry,
    dst_crs: Optional[CRS] = None,
) -> dict:
    xs, ys = g._x_y()

    # ensure float arrays
    if not np.issubdtype(xs.dtype, np.floating):
        xs = xs.astype(float)
    if not np.issubdtype(ys.dtype, np.floating):
        ys = ys.astype(float)

    # reproject if needed
    if dst_crs is not None:
        if g.crs != CRS(dst_crs):
            xs, ys = _reproject_xy(xs, ys, g.crs, dst_crs)

    # convert to pure Python lists of [x, y]
    coords = [[float(x), float(y)] for x, y in zip(xs.tolist(), ys.tolist())]

    if g.kind == GeometryKind.POINT:
        # GeoJSON point: [x, y]
        return {"type": "Point", "coordinates": coords[0]}

    if g.kind == GeometryKind.LINE:
        # GeoJSON line: [[x, y], ...]
        return {"type": "LineString", "coordinates": coords}

    if g.kind == GeometryKind.POLYGON:
        # GeoJSON polygon: [ exterior_ring, hole1, ... ]
        # 'coords' should already be a closed ring per your validator
        return {"type": "Polygon", "coordinates": [coords]}

    raise ValueError(f"Unsupported geometry kind: {g.kind}")


def rasterize_geometries(
    geometries: Sequence[Geometry],
    geo_info: GeographicInfo,
    fill: int = 0,
    default_value: Union[int, float] = 1,
    values: Optional[Sequence[Union[int, float]]] = None,
    all_touched: bool = True,
    dtype: str = "uint8",
    merge_alg: str = "replace",   # "replace" | "add"
) -> np.ndarray:
    """
    Rasterize a sequence of Geometry objects into a numpy array.

    Parameters
    ----------
    geometries : list of Geometry
        Geometry objects in the same CRS `src_crs`.
    geo_info: GeographicInfo
        Geographic information for the output raster.
    src_crs : CRS
        CRS of input geometries.
    fill : scalar
        Background value.
    default_value : scalar
        Burn value when `values` not provided.
    values : optional sequence
        Per-geometry burn values; if provided, must match `len(geometries)`.
    all_touched : bool
        Pass-through to rasterize(); include all touched pixels if True.
    dtype : numpy dtype string
        Output dtype.
    merge_alg : str
        "replace" (last wins) or "add" (sum overlaps).

    Returns
    -------
    np.ndarray
        Rasterized array of shape `out_shape` and dtype `dtype`.
    """

    if values is not None and len(values) != len(geometries):
        raise ValueError("`values` length must match `geometries` length")

    # Prepare shapes in destination CRS
    shapes: List[Tuple[dict, Union[int, float]]] = []
    for i, g in enumerate(geometries):
        gj = geometry_to_geojson(g, dst_crs=geo_info.prj.crs)
        val = values[i] if values is not None else default_value
        shapes.append((gj, val))

    if merge_alg not in {"replace", "add"}:
        raise ValueError("merge_alg must be 'replace' or 'add'")

    # Rasterize
    out = rasterize(
        shapes=shapes,
        out_shape=geo_info.shape,
        transform=geo_info.trans,
        fill=fill,
        all_touched=all_touched,
        dtype=dtype,
        merge_alg=rio_enums.MergeAlg.add
        if merge_alg == "add"
        else rio_enums.MergeAlg.replace,
    )
    return out
