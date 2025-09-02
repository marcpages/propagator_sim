from __future__ import annotations

from typing import List, Optional, Union, Sequence
import re
import math
from enum import Enum
from pydantic import BaseModel, ConfigDict, model_validator, field_validator
from typing import Tuple
import numpy as np
import numpy.typing as npt
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
        arbitrary_types_allowed=True,  # <-- allow np.ndarray, CRS
    )

    kind: GeometryKind
    ys: npt.NDArray[np.floating]
    xs: npt.NDArray[np.floating]
    crs: CRS = CRS.from_epsg(4326)  # default to WGS84

    # Coerce xs/ys to 1D float arrays
    @field_validator("xs", "ys", mode="before")
    @classmethod
    def _coerce_array(cls, v):
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 1:
            raise ValueError("xs/ys must be 1D sequences")
        return arr

    # Coerce CRS from int/str/etc.
    @field_validator("crs", mode="before")
    @classmethod
    def _coerce_crs(cls, v):
        return v if isinstance(v, CRS) else CRS.from_user_input(v)

    @model_validator(mode="before")
    @classmethod
    def _check_geometry(cls, data):
        if not isinstance(data, dict):
            return data
        xs = data.get("xs")
        ys = data.get("ys")
        if xs is not None and ys is not None:
            # don’t assume arrays yet; just check lengths
            if len(xs) != len(ys):
                raise ValueError("coordinates must have the same length")
        return data

    def _x_y(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.xs, self.ys

    def _reproject_x_y(self, dst_crs: CRS) -> Tuple[np.ndarray, np.ndarray]:
        tfm = Transformer.from_crs(self.crs, dst_crs, always_xy=True)
        xs, ys = self._x_y()
        X, Y = tfm.transform(xs, ys)
        return X, Y

    def _shape_rasterize(
        self,
        dst_crs: Optional[CRS] = None,
    ) -> dict:
        xs, ys = self._x_y()
        # reproject if needed
        if dst_crs is not None:
            if self.crs != dst_crs:
                xs, ys = self._reproject_x_y(dst_crs)
        # convert to pure Python lists of [x, y]
        coords = [
            [float(x), float(y)] for x, y in zip(xs.tolist(), ys.tolist())
        ]
        if self.kind == GeometryKind.POINT:
            # GeoJSON point: [x, y]
            return {"type": "Point", "coordinates": coords[0]}
        if self.kind == GeometryKind.LINE:
            # GeoJSON line: [[x, y], ...]
            return {"type": "LineString", "coordinates": coords}
        if self.kind == GeometryKind.POLYGON:
            # GeoJSON polygon: [ exterior_ring, hole1, ... ]
            # 'coords' should already be a closed ring per your validator
            return {"type": "Polygon", "coordinates": [coords]}
        raise ValueError(f"Unsupported geometry kind: {self.kind}")

    def get_middle_point(self) -> Optional[Tuple[float, float]]:
        if self.ys.size == 0 or self.xs.size == 0:
            return None
        return float(np.mean(self.xs)), float(np.mean(self.ys))


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
        if not (
            math.isclose(data["xs"][0], data["xs"][-1])
            and math.isclose(data["ys"][0], data["ys"][-1])
        ):
            raise ValueError("Polygon must be closed")
        return data


# super-class for all geometry types
Geometry = Union[GeoPoint, GeoLine, GeoPolygon]

# ---- parsing ---------------------------------------------------------------
_POINT_RE = re.compile(
    r"""^POINT:\s*(?P<y>-?\d+(?:\.\d+)?)\s*;\s*(?P<x>-?\d+(?:\.\d+)?)\s*$""",
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


class GeometryParser:
    @staticmethod
    def parse_geometry_string(s: str, epsg: int) -> Geometry:
        """Parse POINT/LINE/POLYGON strings into geometry objects."""
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
            ys_arr = np.asarray(
                _split_floats(m_series.group("ys")), dtype=float
            )
            xs_arr = np.asarray(
                _split_floats(m_series.group("xs")), dtype=float
            )
            if ys_arr.size != xs_arr.size:
                raise ValueError(
                    f"{kind}: y/x counts differ \
                    ({ys_arr.size} vs {xs_arr.size})"
                )
            if kind == "LINE":
                return GeoLine(ys=ys_arr, xs=xs_arr, crs=crs)
            elif kind == "POLYGON":
                return GeoPolygon(ys=ys_arr, xs=xs_arr, crs=crs)
            else:
                raise ValueError(f"Unsupported geometry kind: {kind!r}")
        raise ValueError(f"Unsupported geometry string: {s!r}")

    @staticmethod
    def parse_geometry_list(
        v: list, allowed: set[str], epsg: int
    ) -> List[Geometry]:
        if not isinstance(v, list):
            raise ValueError("expected a list")
        out: List[Geometry] = []
        for item in v:
            if isinstance(item, str):
                g = GeometryParser.parse_geometry_string(item, epsg)
            else:
                raise ValueError(f"unsupported entry {item!r}")
            # allowed-kind check
            if isinstance(g, GeoPoint) and "point" not in allowed:
                raise ValueError("POINT not allowed")
            if isinstance(g, GeoLine) and "line" not in allowed:
                raise ValueError("LINE not allowed")
            if isinstance(g, GeoPolygon) and "polygon" not in allowed:
                raise ValueError("POLYGON not allowed")
            out.append(g)
        return out


# --- rasterization ---


def rasterize_geometries(
    geometries: Sequence[Geometry],
    geo_info: GeographicInfo,
    fill: int = 0,
    default_value: Union[int, float] = 1,
    values: Optional[Sequence[Union[int, float]]] = None,
    all_touched: bool = True,
    dtype: str = "uint8",
    merge_alg: str = "replace",  # "replace" | "add"
) -> np.ndarray:
    """
    Rasterize a sequence of Geometry objects into a numpy array.

    Parameters
    ----------
    geometries : list of Geometry
        Geometry objects in the same CRS `src_crs`.
    geo_info: GeographicInfo
        Geographic information for the output raster.
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
        gj = g._shape_rasterize(dst_crs=geo_info.prj.crs)
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
