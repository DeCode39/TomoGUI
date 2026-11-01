from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


KNOWN_FIELDS = (
    "vel",
    "xi",
    "eta",
    "zeta",
    "dlnv",
    "epsilon",
    "phi",
)


def compute_initial_window_size(
    available_width: int,
    available_height: int,
    preferred: tuple[int, int] = (1460, 930),
    max_fraction: tuple[float, float] = (0.94, 0.90),
) -> tuple[int, int]:
    if available_width <= 0 or available_height <= 0:
        return preferred

    width_limit = max(640, int(available_width * max_fraction[0]))
    height_limit = max(480, int(available_height * max_fraction[1]))

    width = min(preferred[0], width_limit, available_width)
    height = min(preferred[1], height_limit, available_height)
    return width, height


def map_box_aspect_from_extent(extent: list[float]) -> float | None:
    lon0, lon1, lat0, lat1 = [float(value) for value in extent]
    lon_span = abs(lon1 - lon0)
    lat_span = abs(lat1 - lat0)
    if lon_span <= 0.0 or lat_span <= 0.0:
        return None

    mean_lat = math.radians(0.5 * (lat0 + lat1))
    km_per_lon_degree = 111.320 * max(abs(math.cos(mean_lat)), 1.0e-6)
    km_per_lat_degree = 110.574
    width_km = lon_span * km_per_lon_degree
    height_km = lat_span * km_per_lat_degree
    if width_km <= 0.0 or height_km <= 0.0:
        return None
    return height_km / width_km


def map_data_aspect_from_extent(extent: list[float]) -> float | None:
    _lon0, _lon1, lat0, lat1 = [float(value) for value in extent]
    mean_lat = math.radians(0.5 * (lat0 + lat1))
    km_per_lon_degree = 111.320 * max(abs(math.cos(mean_lat)), 1.0e-6)
    km_per_lat_degree = 110.574
    if km_per_lon_degree <= 0.0:
        return None
    return km_per_lat_degree / km_per_lon_degree


def is_numeric_dataset(obj: object) -> bool:
    return isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number)


def safe_identifier(text: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", text.strip("/"))
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "field"
    if name[0].isdigit():
        name = f"f_{name}"
    return name


def unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    idx = 2
    while f"{base}_{idx}" in used:
        idx += 1
    name = f"{base}_{idx}"
    used.add(name)
    return name


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_stem(text: str, max_len: int = 80) -> str:
    cleaned = safe_identifier(text).strip("_") or "slice"
    return cleaned[:max_len].rstrip("_") or "slice"


def finite_min_max(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(finite.min())
    vmax = float(finite.max())
    if np.isclose(vmin, vmax):
        delta = 1.0 if vmin == 0 else abs(vmin) * 0.01
        return vmin - delta, vmax + delta
    return vmin, vmax


def centered_zero_limits(vmin: float, vmax: float) -> tuple[float, float]:
    max_abs = max(abs(float(vmin)), abs(float(vmax)))
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        max_abs = 1.0
    return -max_abs, max_abs


def append_unique_sorted_csv_number_list(existing_csv: str, value: float, decimals: int = 6) -> str:
    values: list[float] = []
    for token in existing_csv.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            continue
    if not any(np.isclose(value, item) for item in values):
        values.append(float(value))
    values = sorted(values)
    return ",".join(f"{item:.{decimals}g}" for item in values)


def combined_finite_min_max(arrays: Iterable[np.ndarray]) -> tuple[float, float]:
    mins: list[float] = []
    maxs: list[float] = []
    for arr in arrays:
        arr_np = np.asarray(arr, dtype=float).copy()
        arr_np[np.isclose(arr_np, -999.0)] = np.nan
        finite = arr_np[np.isfinite(arr_np)]
        if finite.size == 0:
            continue
        mins.append(float(finite.min()))
        maxs.append(float(finite.max()))
    if not mins or not maxs:
        return 0.0, 1.0
    return finite_min_max(np.array([min(mins), max(maxs)], dtype=float))


def strict_allclose(a: np.ndarray, b: np.ndarray, atol: float = 1e-10, rtol: float = 1e-10) -> bool:
    return a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol)


def sort_fields(field_names: Iterable[str]) -> list[str]:
    names = list(field_names)

    def key(name: str) -> tuple[int, int, str]:
        if name in KNOWN_FIELDS:
            return (0, KNOWN_FIELDS.index(name), name)
        return (1, 0, name)

    return sorted(names, key=key)


def sibling_file(path: Path, name: str) -> Path:
    return path.with_name(name)
