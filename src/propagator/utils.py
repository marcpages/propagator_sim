"""Utility helpers for raster/vector IO and small geometry ops."""

from typing import TypeVar, Union

import fiona
import numpy as np
import numpy.typing as npt
import rasterio as rio
import utm
from numpy import pi
from rasterio import enums, transform, warp
from rasterio.features import shapes
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.signal.signaltools import medfilt2d
from shapely.geometry import LineString, MultiLineString, mapping, shape

TIME_TAG = 'time'

# Define a type variable that can be either a float or a numpy array of floats
T = TypeVar("T", bound=Union[float, npt.NDArray[np.floating]])

def normalize(angle_to_norm: T) -> T:
    """Normalize an angle to the interval [-pi, pi)."""
    return (angle_to_norm + pi) % (2 * pi) - pi  # type: ignore[return-value]

def add_point(
    img: npt.NDArray[np.floating], c: int, r: int, val: float
) -> list[tuple[int, int]]:
    """Set a single pixel to `val` if inside bounds.

    Returns the list with the written coordinate `(r, c)` when in-bounds, or
    still returns the attempted coordinate for consistency.
    """
    if 0 <= c < img.shape[1] and 0 <= r < img.shape[0]:
        img[r, c] = val
    return [(r, c)]


def add_segment(
    img: npt.NDArray[np.floating],
    c0: int,
    r0: int,
    c1: int,
    r1: int,
    value: float,
) -> list[tuple[int, int]]:
    """Rasterize a discrete line segment using Bresenham and write `value`.

    Returns the list of `(r, c)` points touched along the segment.
    Coordinates are expressed as column-first (`c`) and row (`r`).
    """
    dc = abs(c1 - c0)
    dr = abs(r1 - r0)
    if c0 < c1:
        sc = 1
    else:
        sc = -1

    if r0 < r1:
        sr = 1
    else:
        sr = -1

    err = dc - dr
    points = []
    while True:
        if 0 <= c0 < img.shape[1] and 0 <= r0 < img.shape[0]:
            img[r0, c0] = value
            points.append((r0, c0))

        if c0 == c1 and r0 == r1:
            break

        e2 = 2 * err
        if e2 > -dr:
            err = err - dr
            c0 += sc

        if e2 < dc:
            err = err + dc
            r0 += sr

    return points


def add_line(
    img: npt.NDArray[np.floating], cs: npt.NDArray[np.integer], rs: npt.NDArray[np.integer], val: float
) -> list[tuple[int, int]]:
    """Rasterize a polyline defined by sequences `cs` and `rs` into `img`.

    The last point is not connected back to the first. Returns the contour
    list of touched points.
    """
    contour = []
    img_temp = np.zeros(img.shape)

    for idx in range(len(cs) - 1):
        points = add_segment(img_temp, cs[idx], rs[idx], cs[idx + 1], rs[idx + 1], 1)
        if idx > 0:
            contour.extend(points[1:])
        else:
            contour.extend(points)

    img[img_temp == 1] = val

    return contour


def add_poly(
    img: npt.NDArray[np.floating], cs: npt.NDArray[np.integer], rs: npt.NDArray[np.integer], val: float
) -> list[tuple[int, int]]:
    """Fill a polygon defined by `cs` and `rs` into `img` with `val`.

    The polygon is closed automatically and a flood fill is used to write the
    interior. Returns the contour list of touched boundary points.
    """
    img_temp = np.ones_like(img)
    contour = []

    for idx in range(len(cs) - 1):
        points = add_segment(img_temp, cs[idx], rs[idx], cs[idx + 1], rs[idx + 1], 2)
        if idx > 0:
            contour.extend(points[1:])
        else:
            contour.extend(points)

    points = add_segment(img_temp, cs[-1], rs[-1], cs[0], rs[0], 2)
    contour.extend(points[1:])

    pp = [(0, 0)]
    dim_y, dim_x = img_temp.shape

    while len(pp) > 0:
        pp_n = []
        for x, y in pp:
            if y < dim_y - 1 and img_temp[y + 1, x] == 1:
                img_temp[y + 1, x] = 0
                pp_n.append((x, y + 1))

            if x < dim_x - 1 and img_temp[y, x + 1] == 1:
                img_temp[y, x + 1] = 0
                pp_n.append((x + 1, y))

            if y > 0 and img_temp[y - 1, x] == 1:
                img_temp[y - 1, x] = 0
                pp_n.append((x, y - 1))

            if x > 0 and img_temp[y, x - 1] == 1:
                img_temp[y, x - 1] = 0
                pp_n.append((x - 1, y))

        pp = pp_n

    img[img_temp > 0] = val
    return contour


def read_actions(
    imp_points_string: str,
) -> tuple[
    float,
    float,
    list[tuple[list[float], list[float]]],
    list[tuple[list[float], list[float]]],
    list[tuple[float, float]],
]:
    """Parse action strings into polygons, lines, and points.

    Input format examples:
    - "LINE:[lat lat ...];[lon lon ...]"
    - "POLYGON:[lat lat ...];[lon lon ...]"
    - "POINT:lat;lon"

    Returns `(mid_lat, mid_lon, polys, lines, points)`.
    """
    strings = imp_points_string.split("\n")

    polys, lines, points = [], [], []
    max_lat, max_lon, min_lat, min_lon = -np.inf, -np.inf, np.inf, np.inf

    for s in strings:
        f_type, values = s.split(":")
        values = values.replace("[", "").replace("]", "")
        lats = None
        lons = None
        if f_type == "POLYGON":
            s_lats, s_lons = values.split(";")
            lats = [float(sv) for sv in s_lats.split()]
            lons = [float(sv) for sv in s_lons.split()]
            polys.append((lats, lons))

        elif f_type == "LINE":
            s_lats, s_lons = values.split(";")
            lats = [float(sv) for sv in s_lats.split()]
            lons = [float(sv) for sv in s_lons.split()]
            lines.append((lats, lons))

        elif f_type == "POINT":
            s_lat, s_lon = values.split(";")
            lat, lon = float(s_lat), float(s_lon)
            lats = [lat]
            lons = [lon]
            points.append((lat, lon))

        if lats is None or lons is None:
            raise ValueError("No valid actions found in input string")

        max_lat = max(max(lats), max_lat)
        min_lat = min(min(lats), min_lat)
        max_lon = max(max(lons), max_lon)
        min_lon = min(min(lons), min_lon)
  
    mid_lat = (max_lat + min_lat) / 2
    mid_lon = (max_lon + min_lon) / 2

    return mid_lat, mid_lon, polys, lines, points


def rasterize_actions(
    dim: tuple[int, int],
    points: list[tuple[float, float]],
    lines: list[tuple[list[float], list[float]]],
    polys: list[tuple[list[float], list[float]]],
    lonmin: float,
    latmax: float,
    stepx: float,
    stepy: float,
    zone_number: int,
    base_value: float = 0,
    value: float = 1,
) -> tuple[npt.NDArray[np.floating], list[tuple[int, int]]]:
    """Rasterize points, lines, and polygons into an image of shape `dim`.

    Coordinates are provided in lat/lon and converted using a fixed UTM zone.
    Returns `(img, active_points)` where `active_points` are grid coordinates
    that were touched when drawing.
    """
    img = np.ones(dim) * base_value
    active_points = []
    for line in lines:
        xs, ys, _, _ = zip(
            *[
                utm.from_latlon(p[0], p[1], force_zone_number=zone_number)
                for p in zip(*line)
            ]
        )
        x = np.floor((np.array(xs) - lonmin) / stepx).astype("int")
        y = np.floor((latmax - np.array(ys)) / stepy).astype("int")
        active = add_line(img, x, y, 1)
        active_points.extend(active)
    for point in points:
        xs, ys, _, _ = utm.from_latlon(
            point[0], point[1], force_zone_number=zone_number
        )
        x = int(np.floor((xs - lonmin) / stepx))
        y = int(np.floor((latmax - ys) / stepy))
        active = add_point(img, x, y, 1)
        active_points.extend(active)
    for poly in polys:
        xs, ys, _, _ = zip(
            *[
                utm.from_latlon(p[0], p[1], force_zone_number=zone_number)
                for p in zip(*poly)
            ]
        )
        x = np.floor((np.array(xs) - lonmin) / stepx).astype("int")
        y = np.floor((latmax - np.array(ys)) / stepy).astype("int")
        active = add_poly(img, x, y, 1)
        active_points.extend(active)

    return img, active_points


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


def write_geotiff(
    filename: str,
    values: npt.NDArray[np.floating] | npt.NDArray[np.integer],
    dst_trans,
    dst_crs,
    dtype=np.uint8,
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
    values,
    transf,
    thresholds=[0.5, 0.75, 0.9],
    med_filt_val=9,
    min_length=0.0001,
    smooth_sigma=0.8,
    simp_fact=0.00001,
):
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
        over_t = binary_dilation(binary_erosion(over_t_).astype("uint8")).astype(       # type: ignore #binary_erosion has None typing in library
            "uint8"
        )
        if np.any(over_t):
            for s, v in shapes(over_t, transform=transf):
                sh = shape(s)

                ml = [
                    smooth_linestring(interior_line, smooth_sigma)  # .simplify(simp_fact)
                    for interior_line in sh.interiors                   # type: ignore # sh.interiors is missing in typing
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

