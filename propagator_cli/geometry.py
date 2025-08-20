from __future__ import annotations

from typing import List, Optional, Union, Literal, Any
import re
from pydantic import (BaseModel, field_validator)

# ---- geometry parsing -------------------------------------------------------
_POINT_RE = re.compile(
    r"""^POINT:\[\s*
        (?P<lat>-?\d+(?:\.\d+)?)\s*;\s*
        (?P<lon>-?\d+(?:\.\d+)?)\s*
        \]\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

_SERIES_RE = re.compile(
    r"""^(?P<kind>LINE|POLYGON):\[\s*
        (?P<lats>-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?)+)\s*
        \];\[\s*
        (?P<lons>-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?)+)\s*
        \]\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _split_floats(s: str) -> List[float]:
    return [float(x) for x in s.strip().split() if x.strip()]


# ---- geometry models --------------------------------------------------------
class GeoPoint(BaseModel):
    kind: Literal["point"] = "point"
    # lat/lon = (y, x) in CRS given by geometry_epsg
    lat: float
    lon: float


class GeoLine(BaseModel):
    kind: Literal["line"] = "line"
    points: List[GeoPoint]

    @field_validator("points")
    @classmethod
    def _min_two(cls, v):
        if len(v) < 2:
            raise ValueError("Line must have at least 2 points")
        return v


class GeoPolygon(BaseModel):
    kind: Literal["polygon"] = "polygon"
    points: List[GeoPoint]

    @field_validator("points")
    @classmethod
    def _min_three(cls, v):
        if len(v) < 3:
            raise ValueError("Polygon must have at least 3 points")
        return v


Geometry = Union[GeoPoint, GeoLine, GeoPolygon]


def parse_geometry_string(s: str) -> Geometry:
    """Parse POINT/LINE/POLYGON strings into geometry objects"""
    s = s.strip()
    m_pt = _POINT_RE.match(s)
    if m_pt:
        return GeoPoint(lat=float(m_pt.group("lat")),
                        lon=float(m_pt.group("lon")))
    m_series = _SERIES_RE.match(s)
    if m_series:
        kind = m_series.group("kind").upper()
        lats = _split_floats(m_series.group("lats"))
        lons = _split_floats(m_series.group("lons"))
        if len(lats) != len(lons):
            raise ValueError(f"{kind}: lat/lon counts differ \
                ({len(lats)} vs {len(lons)})")
        pts = [GeoPoint(lat=la, lon=lo) for la, lo in zip(lats, lons)]
        if kind == "LINE":
            return GeoLine(points=pts)
        else:  # POLYGON
            # (optionally) auto-close polygon
            if pts[0].lat != pts[-1].lat or pts[0].lon != pts[-1].lon:
                pts.append(pts[0])
            return GeoPolygon(points=pts)
    raise ValueError(f"Unsupported geometry string: {s!r}")


def _coerce_geometry_list(v: Any, allowed: set[str],
                          field_name: str) -> Optional[List[Geometry]]:
    if v is None:
        return None
    if not isinstance(v, list):
        raise ValueError(f"{field_name}: expected a list")
    out: List[Geometry] = []
    for item in v:
        if isinstance(item, str):
            g = parse_geometry_string(item)
        elif isinstance(item, dict) and "kind" in item:
            k = str(item["kind"]).lower()
            if k == "point":
                g = GeoPoint(**item)
            elif k == "line":
                g = GeoLine(**item)
            elif k == "polygon":
                g = GeoPolygon(**item)
            else:
                raise ValueError(f"{field_name}: \
                    unsupported kind {item['kind']!r}")
        else:
            raise ValueError(f"{field_name}: unsupported entry {item!r}")
        if g.__class__ is GeoPoint and "point" not in allowed:
            raise ValueError(f"{field_name}: POINT not allowed")
        if g.__class__ is GeoLine and "line" not in allowed:
            raise ValueError(f"{field_name}: LINE not allowed")
        if g.__class__ is GeoPolygon and "polygon" not in allowed:
            raise ValueError(f"{field_name}: POLYGON not allowed")
        out.append(g)
    return out
