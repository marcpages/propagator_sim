from __future__ import annotations

from typing import List, Optional, Union, Literal, Any
import re
from pydantic import BaseModel, ConfigDict, field_validator
from typing import Tuple

# ---- geometry parsing -------------------------------------------------------
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


# ---- geometry models --------------------------------------------------------

class GeometryBase(BaseModel):
    """Common fields/behavior for all geometries."""
    model_config = ConfigDict(extra="forbid")

    kind: Literal["point", "line", "polygon"]

    # to be implemented by subclasses: returns (xs, ys)
    def _x_y_lists(self) -> Tuple[List[float], List[float]]:
        raise NotImplementedError

    # to be implemented by subclasses: returns geojson representation
    def _as_geojson(self) -> dict:
        raise NotImplementedError


class GeoPoint(GeometryBase):
    kind: Literal["point"] = "point"
    y: float
    x: float

    def _x_y_lists(self) -> Tuple[List[float], List[float]]:
        return [self.x], [self.y]

    def _as_geojson(self) -> dict:
        """Return GeoJSON representation of the point."""
        x, y = self._x_y_lists()
        return {"type": "Point", "coordinates": (x[0], y[0])}


class GeoLine(GeometryBase):
    kind: Literal["line"] = "line"
    points: List[GeoPoint]

    @field_validator("points")
    @classmethod
    def _min_two(cls, v):
        if len(v) < 2:
            raise ValueError("Line must have at least 2 points")
        return v

    def _x_y_lists(self) -> Tuple[List[float], List[float]]:
        return [p.x for p in self.points], [p.y for p in self.points]

    def _geojson_line(self) -> dict:
        xs, ys = self._x_y_lists()
        return {"type": "LineString", "coordinates": list(zip(xs, ys))}


class GeoPolygon(GeometryBase):
    kind: Literal["polygon"] = "polygon"
    points: List[GeoPoint]

    @field_validator("points")
    @classmethod
    def _min_three(cls, v):
        if len(v) < 3:
            raise ValueError("Polygon must have at least 3 points")
        return v

    def _x_y_lists(self) -> Tuple[List[float], List[float]]:
        return [p.x for p in self.points], [p.y for p in self.points]

    def _geojson_polygon(self) -> dict:
        xs, ys = self._x_y_lists()
        coords = list(zip(xs, ys))
        # if coords[0] != coords[-1]:  > to be moved in the check of the object
        #     coords.append(coords[0])
        return {"type": "Polygon", "coordinates": [coords]}


# super-class for all geometry types
Geometry = Union[GeoPoint, GeoLine, GeoPolygon]


def parse_geometry_string(s: str) -> Geometry:
    """Parse POINT/LINE/POLYGON strings into geometry objects"""
    s = s.strip()
    m_pt = _POINT_RE.match(s)
    if m_pt:
        return GeoPoint(y=float(m_pt.group("y")),
                        x=float(m_pt.group("x")))
    m_series = _SERIES_RE.match(s)
    if m_series:
        kind = m_series.group("kind").upper()
        ys = _split_floats(m_series.group("ys"))
        xs = _split_floats(m_series.group("xs"))
        if len(ys) != len(xs):
            raise ValueError(f"{kind}: y/x counts differ \
                ({len(ys)} vs {len(xs)})")
        pts = [GeoPoint(y=y, x=x) for y, x in zip(ys, xs)]
        if kind == "LINE":
            return GeoLine(points=pts)
        else:  # POLYGON
            # (optionally) auto-close polygon
            if pts[0].y != pts[-1].y or pts[0].x != pts[-1].x:
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
