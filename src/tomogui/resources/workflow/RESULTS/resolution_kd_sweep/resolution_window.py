"""Fast local-window sums for checkerboard resolution scoring."""

from __future__ import annotations

import numpy as np


def bounds_for_window(values: np.ndarray, half_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Return inclusive/exclusive index bounds for coordinate-centered windows.

    The coordinate arrays may be ascending or descending. The output matches the
    older implementation that searched all coordinates around each point and
    then used the minimum and maximum matching indices as a contiguous box.
    """
    coords = np.asarray(values, dtype=np.float64)
    lo = np.zeros(coords.size, dtype=np.int64)
    hi = np.zeros(coords.size, dtype=np.int64)
    for idx, value in enumerate(coords):
        inside = np.flatnonzero((coords >= value - half_width) & (coords <= value + half_width))
        if inside.size == 0:
            lo[idx] = idx
            hi[idx] = idx + 1
        else:
            lo[idx] = int(inside.min())
            hi[idx] = int(inside.max()) + 1
    return lo, hi


def summed_volume(values: np.ndarray) -> np.ndarray:
    """Return a 1-cell padded 3D prefix-sum volume."""
    csum = np.asarray(values, dtype=np.float64).cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    return np.pad(csum, ((1, 0), (1, 0), (1, 0)), mode="constant")


def windowed_sum(
    values: np.ndarray,
    dep: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    window_depth_km: float,
    window_lat_deg: float,
    window_lon_deg: float,
) -> np.ndarray:
    """Sum values inside the local 3D window around every model node.

    This is equivalent to calling the old scalar ``box_sum`` for every
    depth/lat/lon cell, but performs the eight prefix-sum lookups with NumPy
    broadcasting.
    """
    z_lo, z_hi = bounds_for_window(dep, window_depth_km * 0.5)
    y_lo, y_hi = bounds_for_window(lat, window_lat_deg * 0.5)
    x_lo, x_hi = bounds_for_window(lon, window_lon_deg * 0.5)
    prefix = summed_volume(np.nan_to_num(values, nan=0.0))

    z0 = z_lo[:, None, None]
    z1 = z_hi[:, None, None]
    y0 = y_lo[None, :, None]
    y1 = y_hi[None, :, None]
    x0 = x_lo[None, None, :]
    x1 = x_hi[None, None, :]

    return (
        prefix[z1, y1, x1]
        - prefix[z0, y1, x1]
        - prefix[z1, y0, x1]
        - prefix[z1, y1, x0]
        + prefix[z0, y0, x1]
        + prefix[z0, y1, x0]
        + prefix[z1, y0, x0]
        - prefix[z0, y0, x0]
    )


def windowed_resolution(
    numerator: np.ndarray,
    denominator_terms: np.ndarray,
    dep: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    window_depth_km: float,
    window_lat_deg: float,
    window_lon_deg: float,
) -> np.ndarray:
    """Compute the paper-style local resolution score."""
    num = windowed_sum(
        numerator,
        dep,
        lat,
        lon,
        window_depth_km=window_depth_km,
        window_lat_deg=window_lat_deg,
        window_lon_deg=window_lon_deg,
    )
    den_terms = windowed_sum(
        denominator_terms,
        dep,
        lat,
        lon,
        window_depth_km=window_depth_km,
        window_lat_deg=window_lat_deg,
        window_lon_deg=window_lon_deg,
    )
    den = 2.0 * den_terms
    ratio = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den, out=ratio, where=den > 0.0)
    return np.where(den > 0.0, 1.0 - ratio, 0.0)
