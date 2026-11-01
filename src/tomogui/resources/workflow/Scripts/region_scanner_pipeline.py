#!/usr/bin/env python3
"""Combine ISC CSV -> src_rec conversion with region scanning and ray coverage.

Pipeline (P-only by default):
1) Read ISC arrivals CSV and build a src_rec-ready dataframe.
2) Build a SrcRec object and apply global filters (phase/min stations/distance).
3) Scan candidate regions using station azimuth coverage (or use full bounds).
4) For each region:
   - subset arrivals to stations + events inside region
   - optional declustering by 3-D blocks
   - enforce minimum P picks per event
   - restrict events to a central-area fraction
   - write src_rec.dat + diagnostic plots
   - run ray coverage (optional)

Use --include-s to keep S picks alongside P picks. The region filters still
use P-pick thresholds to keep the workflow P-driven.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import matplotlib

_mplconfig_dir = Path(tempfile.gettempdir()) / "matplotlib"
_mplconfig_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfig_dir))
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

try:
    from pytomoatt.src_rec import SrcRec, setup_rec_points_dd
except ModuleNotFoundError:
    SrcRec = None  # type: ignore[assignment]
    setup_rec_points_dd = None  # type: ignore[assignment]

LOGGER = logging.getLogger("region_scanner")


def _require_pytomoatt() -> None:
    if SrcRec is None or setup_rec_points_dd is None:
        raise SystemExit(
            "ERROR: pytomoatt is required for region scanning. "
            "Install PyTomoATT into the active environment so `import pytomoatt` works."
        )


# --- Ray coverage utilities (inlined from Ray_coverage.py) ---


@dataclass
class Event:
    idx: int
    event_id: str
    lat: float
    lon: float
    depth_km: float


@dataclass
class Station:
    name: str
    lat: float
    lon: float
    elev_km: float
    station_id: int


@dataclass
class Ray:
    event_idx: int
    station_name: str
    travel_time: float
    weight: float
    phase: str


@dataclass
class CellMetrics:
    hit_count: int = 0
    path_length_km: float = 0.0
    azimuths: List[float] = field(default_factory=list)
    event_ids: set[str] = field(default_factory=set)
    station_ids: set[str] = field(default_factory=set)


@dataclass
class RegionalCoverageResult:
    region_id: int
    depth_threshold: float
    area_cells: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    n_events: int
    n_stations: int
    n_rays: int
    max_depth_km: float
    max_Q: float


@dataclass
class GridSpec:
    lat_edges: np.ndarray
    lon_edges: np.ndarray
    depth_edges: np.ndarray
    dlat: float
    dlon: float
    ddep: float

    @property
    def nlat(self) -> int:
        return max(len(self.lat_edges) - 1, 0)

    @property
    def nlon(self) -> int:
        return max(len(self.lon_edges) - 1, 0)

    @property
    def ndep(self) -> int:
        return max(len(self.depth_edges) - 1, 0)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.nlat, self.nlon, self.ndep


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometers."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def forward_azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute forward azimuth from (lat1, lon1) to (lat2, lon2) in degrees."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    azimuth = math.degrees(math.atan2(y, x))
    return (azimuth + 360.0) % 360.0


def parse_src_rec(path: Path) -> Tuple[Dict[int, Event], Dict[str, Station], List[Ray]]:
    """Parse a TomoATT src_rec.dat file."""
    events: Dict[int, Event] = {}
    stations: Dict[str, Station] = {}
    rays: List[Ray] = []

    station_counter = 0
    current_event_idx: Optional[int] = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            if not line:
                continue

            if raw_line[0].isspace():
                if current_event_idx is None:
                    raise ValueError("Receiver line encountered before any source header.")
                parts = line.split()
                if len(parts) < 9:
                    raise ValueError(f"Malformed receiver line: {line}")
                src_idx = int(parts[0])
                if src_idx != current_event_idx:
                    current_event_idx = src_idx
                station_name = parts[2]
                stla = float(parts[3])
                stlo = float(parts[4])
                stel = float(parts[5])
                phase = parts[6]
                tt = float(parts[7])
                weight = float(parts[8]) if len(parts) > 8 else 1.0

                if station_name not in stations:
                    stations[station_name] = Station(
                        name=station_name,
                        lat=stla,
                        lon=stlo,
                        elev_km=stel / 1000.0 if abs(stel) > 10.0 else stel,
                        station_id=station_counter,
                    )
                    station_counter += 1

                rays.append(
                    Ray(
                        event_idx=current_event_idx,
                        station_name=station_name,
                        travel_time=tt,
                        weight=weight,
                        phase=phase,
                    )
                )
                continue

            parts = line.split()
            if len(parts) < 14:
                raise ValueError(f"Malformed source line: {line}")

            event_idx = int(parts[0])
            lat = float(parts[7])
            lon = float(parts[8])
            depth = float(parts[9])
            event_id = parts[12]

            events[event_idx] = Event(
                idx=event_idx,
                event_id=event_id,
                lat=lat,
                lon=lon,
                depth_km=depth,
            )
            current_event_idx = event_idx

    if not events:
        raise ValueError(f"No sources found in {path}")
    if not rays:
        raise ValueError(f"No receivers found in {path}")

    LOGGER.info("Parsed %s events, %s stations, %s rays", len(events), len(stations), len(rays))
    return events, stations, rays


def _compute_default_bounds(
    events: Dict[int, Event], stations: Dict[str, Station]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    latitudes = [ev.lat for ev in events.values()] + [st.lat for st in stations.values()]
    longitudes = [ev.lon for ev in events.values()] + [st.lon for st in stations.values()]
    return (min(latitudes), max(latitudes)), (min(longitudes), max(longitudes))


def build_grid(
    events: Dict[int, Event],
    stations: Dict[str, Station],
    args: argparse.Namespace,
) -> GridSpec:
    """Construct the lat/lon/depth grid edges from inputs and configuration."""
    (data_lat_min, data_lat_max), (data_lon_min, data_lon_max) = _compute_default_bounds(events, stations)
    depth_min = 0.0
    depth_max = max(ev.depth_km for ev in events.values())

    lat_min, lat_max = (args.lat_range or [data_lat_min, data_lat_max])
    lon_min, lon_max = (args.lon_range or [data_lon_min, data_lon_max])
    dep_min, dep_max = (args.depth_range or [depth_min, depth_max])

    if lat_min >= lat_max or lon_min >= lon_max or dep_min >= dep_max:
        raise ValueError("Invalid region bounds (min must be < max for lat/lon/depth).")

    dlat = args.dlat
    dlon = args.dlon
    ddep = args.ddep

    lat_edges = _build_edges(lat_min, lat_max, dlat)
    lon_edges = _build_edges(lon_min, lon_max, dlon)
    depth_edges = _build_edges(dep_min, dep_max, ddep)

    LOGGER.info(
        "Grid spans lat %.2f-%.2f deg, lon %.2f-%.2f deg, depth %.1f-%.1f km (%s x %s x %s cells)",
        lat_edges[0],
        lat_edges[-1],
        lon_edges[0],
        lon_edges[-1],
        depth_edges[0],
        depth_edges[-1],
        len(lat_edges) - 1,
        len(lon_edges) - 1,
        len(depth_edges) - 1,
    )
    return GridSpec(lat_edges=lat_edges, lon_edges=lon_edges, depth_edges=depth_edges, dlat=dlat, dlon=dlon, ddep=ddep)


def _build_edges(min_value: float, max_value: float, spacing: float) -> np.ndarray:
    if spacing <= 0:
        raise ValueError("Grid spacing must be positive.")
    edges = np.arange(min_value, max_value + spacing * 0.5, spacing, dtype=float)
    if edges[-1] < max_value:
        edges = np.append(edges, max_value)
    if len(edges) < 2:
        edges = np.array([min_value, max_value], dtype=float)
    return edges


def sample_ray_points(
    event: Event,
    station: Station,
    step_km: float,
) -> Tuple[List[Tuple[float, float, float]], float, float]:
    """Generate sampled points along the ray along with per-segment length and azimuth."""
    station_depth = max(-station.elev_km, 0.0)
    horizontal_km = haversine_km(event.lat, event.lon, station.lat, station.lon)
    depth_diff = station_depth - event.depth_km
    total_length = math.hypot(horizontal_km, depth_diff)
    if total_length <= 0.0:
        return [], 0.0, 0.0

    steps = max(int(total_length / step_km), 1)
    segment_length = total_length / steps
    azimuth = forward_azimuth(event.lat, event.lon, station.lat, station.lon)

    samples: List[Tuple[float, float, float]] = []
    for idx in range(steps):
        frac = (idx + 0.5) / steps
        lat = (1.0 - frac) * event.lat + frac * station.lat
        lon = (1.0 - frac) * event.lon + frac * station.lon
        depth = (1.0 - frac) * event.depth_km + frac * station_depth
        samples.append((lat, lon, depth))

    return samples, segment_length, azimuth


def locate_index(value: float, edges: np.ndarray) -> Optional[int]:
    if value < edges[0] or value >= edges[-1]:
        return None
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    idx = min(max(idx, 0), len(edges) - 2)
    return idx


def accumulate_ray_coverage(
    events: Dict[int, Event],
    stations: Dict[str, Station],
    rays: Iterable[Ray],
    grid: GridSpec,
    step_km: float,
) -> Dict[Tuple[int, int, int], CellMetrics]:
    """Populate per-cell metrics by sampling rays across the grid."""
    cells: Dict[Tuple[int, int, int], CellMetrics] = {}
    for ray in rays:
        event = events.get(ray.event_idx)
        station = stations.get(ray.station_name)
        if event is None or station is None:
            continue
        samples, segment_len, azimuth = sample_ray_points(event, station, step_km)
        if not samples:
            continue

        for lat, lon, depth in samples:
            i = locate_index(lat, grid.lat_edges)
            j = locate_index(lon, grid.lon_edges)
            k = locate_index(depth, grid.depth_edges)
            if i is None or j is None or k is None:
                continue
            cell = cells.setdefault((i, j, k), CellMetrics())
            cell.hit_count += 1
            cell.path_length_km += segment_len
            cell.azimuths.append(azimuth)
            cell.event_ids.add(event.event_id)
            cell.station_ids.add(station.name)

    LOGGER.info("Accumulated coverage for %s populated cells.", len(cells))
    return cells


def compute_cell_arrays(
    grid: GridSpec,
    cells: Dict[Tuple[int, int, int], CellMetrics],
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    """Convert sparse cell metrics to dense arrays and compute Q components."""
    shape = grid.shape
    hit_counts = np.zeros(shape, dtype=np.int32)
    path_lengths = np.zeros(shape, dtype=np.float32)
    h_norm = np.zeros(shape, dtype=np.float32)
    a_score = np.zeros(shape, dtype=np.float32)
    d_score = np.zeros(shape, dtype=np.float32)
    q_score = np.zeros(shape, dtype=np.float32)

    if not cells:
        LOGGER.warning("No populated cells to compute metrics.")
        return {
            "hit_counts": hit_counts,
            "path_lengths": path_lengths,
            "H_norm": h_norm,
            "A_score": a_score,
            "D_score": d_score,
            "Q": q_score,
        }

    H_max = max((cell.hit_count for cell in cells.values()), default=0)
    Ne_max = max((len(cell.event_ids) for cell in cells.values()), default=0)
    Ns_max = max((len(cell.station_ids) for cell in cells.values()), default=0)

    H0 = args.hit_saturation if args.hit_saturation > 0 else H_max
    Ne0 = args.event_saturation if args.event_saturation > 0 else Ne_max
    Ns0 = args.station_saturation if args.station_saturation > 0 else Ns_max

    H0 = H0 or 1.0
    Ne0 = Ne0 or 1.0
    Ns0 = Ns0 or 1.0

    for (i, j, k), cell in cells.items():
        H = cell.hit_count
        hit_counts[i, j, k] = H
        path_lengths[i, j, k] = cell.path_length_km

        h_val = min(1.0, H / float(H0))
        h_norm[i, j, k] = h_val

        A_list = cell.azimuths
        if len(A_list) < 2:
            a_val = 0.0
        else:
            ordered = sorted(A_list)
            gaps = [ordered[idx + 1] - ordered[idx] for idx in range(len(ordered) - 1)]
            gaps.append((ordered[0] + 360.0) - ordered[-1])
            gap_max = max(gaps)
            a_val = max(0.0, min(1.0, 1.0 - gap_max / 180.0))
        a_score[i, j, k] = a_val

        e_norm = min(1.0, len(cell.event_ids) / float(Ne0))
        s_norm = min(1.0, len(cell.station_ids) / float(Ns0))
        d_val = 0.5 * (e_norm + s_norm)
        d_score[i, j, k] = d_val

        q_val = (
            args.weight_hits * h_val
            + args.weight_azimuth * a_val
            + args.weight_diversity * d_val
        )
        q_score[i, j, k] = max(0.0, min(1.0, q_val))

    return {
        "hit_counts": hit_counts,
        "path_lengths": path_lengths,
        "H_norm": h_norm,
        "A_score": a_score,
        "D_score": d_score,
        "Q": q_score,
    }


def compute_depth_max(grid: GridSpec, q_array: np.ndarray, q_min: float) -> np.ndarray:
    """Determine max depth per lat/lon cell where Q meets the threshold."""
    nlat, nlon, _ = q_array.shape
    depth_max = np.zeros((nlat, nlon), dtype=np.float32)
    depth_centers = 0.5 * (grid.depth_edges[:-1] + grid.depth_edges[1:])

    for i in range(nlat):
        for j in range(nlon):
            good = np.where(q_array[i, j, :] >= q_min)[0]
            if good.size:
                depth_max[i, j] = float(depth_centers[good.max()])

    return depth_max


def write_hdf5(
    path: Path,
    grid: GridSpec,
    arrays: Dict[str, np.ndarray],
    depth_max: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Persist metrics and metadata to an HDF5 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("lat_edges", data=grid.lat_edges)
        handle.create_dataset("lon_edges", data=grid.lon_edges)
        handle.create_dataset("depth_edges", data=grid.depth_edges)
        handle.create_dataset("depth_max", data=depth_max)

        metrics_group = handle.create_group("metrics")
        for name, array in arrays.items():
            metrics_group.create_dataset(name, data=array)

        meta = handle.create_group("metadata")
        meta.attrs["src_rec"] = str(Path(args.src_rec).resolve())
        meta.attrs["dlat"] = grid.dlat
        meta.attrs["dlon"] = grid.dlon
        meta.attrs["ddep"] = grid.ddep
        meta.attrs["weight_hits"] = args.weight_hits
        meta.attrs["weight_azimuth"] = args.weight_azimuth
        meta.attrs["weight_diversity"] = args.weight_diversity
        meta.attrs["q_min"] = args.q_min

    LOGGER.info("Wrote HDF5 metrics to %s", path)


def write_summary_csv(path: Path, grid: GridSpec, depth_max: np.ndarray, min_depth: float = 10.0) -> None:
    """Write per-square depth limits for downstream tomography configuration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lat_min", "lat_max", "lon_min", "lon_max", "depth_max_km"])
        for i in range(grid.nlat):
            lat_min = grid.lat_edges[i]
            lat_max = grid.lat_edges[i + 1]
            for j in range(grid.nlon):
                lon_min = grid.lon_edges[j]
                lon_max = grid.lon_edges[j + 1]
                max_depth = depth_max[i, j]
                if max_depth < min_depth:
                    continue
                writer.writerow([lat_min, lat_max, lon_min, lon_max, max_depth])
    LOGGER.info("Wrote coverage summary CSV to %s", path)


def write_regional_summary(path: Path, results: List[RegionalCoverageResult]) -> None:
    """Persist hierarchical regional coverage summaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "region_id",
                "depth_threshold_km",
                "area_cells",
                "lat_min",
                "lat_max",
                "lon_min",
                "lon_max",
                "n_events",
                "n_stations",
                "n_rays",
                "max_depth_km",
                "max_Q",
            ]
        )
        for res in results:
            writer.writerow(
                [
                    res.region_id,
                    res.depth_threshold,
                    res.area_cells,
                    res.lat_min,
                    res.lat_max,
                    res.lon_min,
                    res.lon_max,
                    res.n_events,
                    res.n_stations,
                    res.n_rays,
                    res.max_depth_km,
                    res.max_Q,
                ]
            )
    LOGGER.info("Wrote regional coverage summary to %s", path)


def plot_depth_map(path: Path, grid: GridSpec, depth_max: np.ndarray) -> None:
    """Create a plan-view map of max depth per cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("white")
    data = depth_max.copy()
    data_masked = np.ma.masked_where(data <= 0.0, data)
    positive = data[data > 0]
    vmin = float(np.nanmin(positive)) if positive.size else 0.0
    vmax = float(np.nanmax(positive)) if positive.size else 1.0
    mesh = ax.pcolormesh(
        grid.lon_edges,
        grid.lat_edges,
        data_masked,
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    cb = fig.colorbar(mesh, ax=ax, shrink=0.8)
    cb.set_label("Max depth [km]")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Ray coverage max-depth map")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved depth map PNG to %s", path)


def plot_regional_depth_map(path: Path, grid: GridSpec, depth_max: np.ndarray, region: RegionalCoverageResult) -> None:
    """Plot a depth map for a single regional rectangle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("white")
    data_masked = np.ma.masked_where(depth_max <= 0.0, depth_max)
    positive = depth_max[depth_max > 0]
    vmin = float(np.nanmin(positive)) if positive.size else 0.0
    vmax = float(np.nanmax(positive)) if positive.size else 1.0
    mesh = ax.pcolormesh(
        grid.lon_edges,
        grid.lat_edges,
        data_masked,
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    cb = fig.colorbar(mesh, ax=ax, shrink=0.8)
    cb.set_label("Max depth [km]")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(
        f"Regional coverage #{region.region_id} "
        f"({region.lat_min:.2f}-{region.lat_max:.2f} deg lat, {region.lon_min:.2f}-{region.lon_max:.2f} deg lon)"
    )
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved regional depth map to %s", path)


def resolve_output_path(target: str, base_dir: Path) -> Path:
    """Place outputs inside the designated directory unless an absolute path is provided."""
    candidate = Path(target)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def subset_region_data(
    events: Dict[int, Event],
    stations: Dict[str, Station],
    rays: Iterable[Ray],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Tuple[Dict[int, Event], Dict[str, Station], List[Ray]]:
    """Limit events, stations, and rays to those fully contained in the rectangle."""
    ev_subset = {
        idx: ev
        for idx, ev in events.items()
        if lat_min <= ev.lat <= lat_max and lon_min <= ev.lon <= lon_max
    }
    st_subset = {
        name: st
        for name, st in stations.items()
        if lat_min <= st.lat <= lat_max and lon_min <= st.lon <= lon_max
    }

    subset_rays: List[Ray] = []
    for ray in rays:
        ev = events.get(ray.event_idx)
        st = stations.get(ray.station_name)
        if ev is None or st is None:
            continue
        if (
            lat_min <= ev.lat <= lat_max
            and lon_min <= ev.lon <= lon_max
            and lat_min <= st.lat <= lat_max
            and lon_min <= st.lon <= lon_max
        ):
            subset_rays.append(ray)
    return ev_subset, st_subset, subset_rays


def compute_regional_coverage(
    events: Dict[int, Event],
    stations: Dict[str, Station],
    rays: Iterable[Ray],
    grid: GridSpec,
    args: argparse.Namespace,
    min_cell_size: float = 0.1,
    plot_dir: Optional[Path] = None,
) -> List[RegionalCoverageResult]:
    """Run the same coverage/rectangle logic as the main workflow, but with endpoint filtering."""
    lat_min = grid.lat_edges[0]
    lat_max = grid.lat_edges[-1]
    lon_min = grid.lon_edges[0]
    lon_max = grid.lon_edges[-1]

    ev_subset, st_subset, ray_subset = subset_region_data(events, stations, rays, lat_min, lat_max, lon_min, lon_max)
    if not ray_subset:
        LOGGER.warning("No rays remain after endpoint filtering for regional coverage.")
        return []

    cells = accumulate_ray_coverage(ev_subset, st_subset, ray_subset, grid, args.ray_step_km)
    arrays = compute_cell_arrays(grid, cells, args)
    depth_max = compute_depth_max(grid, arrays["Q"], args.q_min)

    if plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "regional_depth.png"
        plot_regional_depth_map(
            plot_path,
            grid,
            depth_max,
            RegionalCoverageResult(
                region_id=0,
                depth_threshold=float("nan"),
                area_cells=grid.nlat * grid.nlon,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                n_events=len(ev_subset),
                n_stations=len(st_subset),
                n_rays=len(ray_subset),
                max_depth_km=float(depth_max.max()) if depth_max.size else 0.0,
                max_Q=float(arrays["Q"].max()) if arrays["Q"].size else 0.0,
            ),
        )

    thresholds: List[float] = []
    if not args.skip_rectangles:
        thresholds = build_rectangle_thresholds(depth_max, args.rectangle_min_depth, args.rectangle_step)

    rectangles = find_max_rectangles(grid, depth_max, thresholds) if thresholds else []
    results: List[RegionalCoverageResult] = []
    region_counter = 1

    for rect in rectangles:
        if rect.get("area_cells", 0) <= 0:
            continue
        i_start = int(rect["i_start"])
        i_end = int(rect["i_end"])
        j_start = int(rect["j_start"])
        j_end = int(rect["j_end"])

        depth_sub = depth_max[i_start:i_end, j_start:j_end]
        q_sub = arrays["Q"][i_start:i_end, j_start:j_end, :]

        lat_min_r = rect["lat_min"]
        lat_max_r = rect["lat_max"]
        lon_min_r = rect["lon_min"]
        lon_max_r = rect["lon_max"]

        ev_r, st_r, rays_r = subset_region_data(ev_subset, st_subset, ray_subset, lat_min_r, lat_max_r, lon_min_r, lon_max_r)

        results.append(
            RegionalCoverageResult(
                region_id=region_counter,
                depth_threshold=rect["depth_threshold"],
                area_cells=int(rect["area_cells"]),
                lat_min=lat_min_r,
                lat_max=lat_max_r,
                lon_min=lon_min_r,
                lon_max=lon_max_r,
                n_events=len(ev_r),
                n_stations=len(st_r),
                n_rays=len(rays_r),
                max_depth_km=float(depth_sub.max()) if depth_sub.size else 0.0,
                max_Q=float(q_sub.max()) if q_sub.size else 0.0,
            )
        )

        if plot_dir:
            plot_dir.mkdir(parents=True, exist_ok=True)
            if rect.get("area_cells", 0) > 0:
                rect_path = plot_dir / f"regional_rectangle_depth_{rect['depth_threshold']:.1f}.png"
                plot_rectangle(rect_path, grid, depth_max, rect)

        region_counter += 1

    if not results:
        results.append(
            RegionalCoverageResult(
                region_id=region_counter,
                depth_threshold=float("nan"),
                area_cells=grid.nlat * grid.nlon,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                n_events=len(ev_subset),
                n_stations=len(st_subset),
                n_rays=len(ray_subset),
                max_depth_km=float(depth_max.max()) if depth_max.size else 0.0,
                max_Q=float(arrays["Q"].max()) if arrays["Q"].size else 0.0,
            )
        )

    LOGGER.info("Computed regional coverage for %s rectangles.", len(results))
    return results


def _compute_histogram(matrix: np.ndarray) -> np.ndarray:
    """Compute histogram heights for maximal rectangle calculation."""
    hist = matrix.astype(np.int32)
    n_rows, n_cols = hist.shape
    for i in range(1, n_rows):
        for j in range(n_cols):
            if hist[i, j]:
                hist[i, j] += hist[i - 1, j]
    return hist


def _largest_rectangle_area(hist_row: np.ndarray) -> Tuple[int, int, int]:
    """Return (area, width, height) for the largest rectangle in a histogram row."""
    stack: List[int] = []
    max_area = 0
    best_width = 0
    best_height = 0
    extended = np.append(hist_row, 0)
    for idx, height in enumerate(extended):
        while stack and extended[stack[-1]] > height:
            h = extended[stack.pop()]
            start = stack[-1] + 1 if stack else 0
            width = idx - start
            area = h * width
            if area > max_area:
                max_area = area
                best_width = width
                best_height = h
        stack.append(idx)
    return max_area, best_width, best_height


def find_max_rectangles(
    grid: GridSpec,
    depth_max: np.ndarray,
    thresholds: List[float],
) -> List[Dict[str, float]]:
    """Identify the largest rectangle per depth threshold."""
    rectangles: List[Dict[str, float]] = []
    mask = depth_max.copy()
    nlat, nlon = depth_max.shape

    for depth in sorted(thresholds, reverse=True):
        binary = (mask >= depth).astype(np.int32)
        if not binary.any():
            rectangles.append(
                {
                    "depth_threshold": depth,
                    "i_start": math.nan,
                    "i_end": math.nan,
                    "j_start": math.nan,
                    "j_end": math.nan,
                    "lat_min": math.nan,
                    "lat_max": math.nan,
                    "lon_min": math.nan,
                    "lon_max": math.nan,
                    "area_cells": 0,
                }
            )
            continue

        hist = _compute_histogram(binary)
        best = {"area": 0, "row": 0, "col": 0, "width": 0, "height": 0}
        for i in range(nlat):
            area, width, height = _largest_rectangle_area(hist[i])
            if area > best["area"]:
                best.update({"area": area, "row": i, "width": width, "height": height})
                if width > 0:
                    col = None
                    count = 0
                    for j in range(nlon):
                        if hist[i, j] >= height:
                            count += 1
                        else:
                            count = 0
                        if count == width:
                            col = j - width + 1
                            break
                    if col is None:
                        col = 0
                    best["col"] = col

        if best["area"] == 0 or best["width"] == 0 or best["height"] == 0:
            rectangles.append(
                {
                    "depth_threshold": depth,
                    "i_start": math.nan,
                    "i_end": math.nan,
                    "j_start": math.nan,
                    "j_end": math.nan,
                    "lat_min": math.nan,
                    "lat_max": math.nan,
                    "lon_min": math.nan,
                    "lon_max": math.nan,
                    "area_cells": 0,
                }
            )
            continue

        row_end = best["row"]
        row_start = row_end - best["height"] + 1
        col_start = best["col"]
        col_end = col_start + best["width"]

        lat_min = grid.lat_edges[row_start]
        lat_max = grid.lat_edges[row_end + 1]
        lon_min = grid.lon_edges[col_start]
        lon_max = grid.lon_edges[col_end]

        rectangles.append(
            {
                "depth_threshold": depth,
                "i_start": row_start,
                "i_end": row_end + 1,
                "j_start": col_start,
                "j_end": col_end,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "area_cells": best["area"],
            }
        )

    return rectangles


def plot_rectangle(
    path: Path,
    grid: GridSpec,
    depth_max: np.ndarray,
    rectangle: Dict[str, float],
) -> None:
    """Plot a depth map highlighting one rectangle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("white")
    data = depth_max.copy()
    data_masked = np.ma.masked_where(data <= 0.0, data)
    positive = data[data > 0]
    vmin = float(np.nanmin(positive)) if positive.size else 0.0
    vmax = float(np.nanmax(positive)) if positive.size else 1.0
    mesh = ax.pcolormesh(
        grid.lon_edges,
        grid.lat_edges,
        data_masked,
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    cb = fig.colorbar(mesh, ax=ax, shrink=0.8)
    cb.set_label("Max depth [km]")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"Largest rectangle for depth >= {rectangle['depth_threshold']} km")

    if math.isfinite(rectangle["lat_min"]):
        ax.add_patch(
            plt.Rectangle(
                (rectangle["lon_min"], rectangle["lat_min"]),
                rectangle["lon_max"] - rectangle["lon_min"],
                rectangle["lat_max"] - rectangle["lat_min"],
                fill=False,
                edgecolor="black",
                linewidth=2,
            )
        )
        label = (
            f"Depth >= {rectangle['depth_threshold']:.1f} km\n"
            f"Lat: {rectangle['lat_min']:.2f}-{rectangle['lat_max']:.2f} deg\n"
            f"Lon: {rectangle['lon_min']:.2f}-{rectangle['lon_max']:.2f} deg"
        )
        ax.text(
            rectangle["lon_min"],
            rectangle["lat_max"],
            label,
            color="black",
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    ax.set_aspect("equal", adjustable="box")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved rectangle plot to %s", path)


def build_rectangle_thresholds(depth_max: np.ndarray, min_depth: float, step: float) -> List[float]:
    """Create a descending list of depth thresholds starting at the max coverage depth."""
    if step <= 0:
        raise ValueError("Rectangle step must be positive.")
    positive = depth_max[np.isfinite(depth_max) & (depth_max > 0)]
    if positive.size == 0:
        return []
    max_depth = float(positive.max())
    min_depth = max(min_depth, 0.0)
    usable = positive[positive >= min_depth]
    if usable.size == 0:
        return []
    effective_min = float(usable.min())
    if max_depth < effective_min:
        return []
    thresholds: List[float] = []
    current = max_depth
    while current >= effective_min:
        thresholds.append(round(current, 3))
        current -= step
    if thresholds and thresholds[-1] != effective_min:
        thresholds.append(round(effective_min, 3))
    return thresholds


ray_coverage = SimpleNamespace(
    forward_azimuth=forward_azimuth,
    parse_src_rec=parse_src_rec,
    build_grid=build_grid,
    accumulate_ray_coverage=accumulate_ray_coverage,
    compute_cell_arrays=compute_cell_arrays,
    compute_depth_max=compute_depth_max,
    write_hdf5=write_hdf5,
    write_summary_csv=write_summary_csv,
    write_regional_summary=write_regional_summary,
    plot_depth_map=plot_depth_map,
    plot_regional_depth_map=plot_regional_depth_map,
    resolve_output_path=resolve_output_path,
    build_rectangle_thresholds=build_rectangle_thresholds,
    find_max_rectangles=find_max_rectangles,
    plot_rectangle=plot_rectangle,
    compute_regional_coverage=compute_regional_coverage,
)


# --- ISC conversion utilities (inlined from make_src_rec_from_isc.py) ---

MAX_SCATTER_POINTS = 80000
BORDER_RESOLUTION = "10m"
EARTH_RADIUS_KM = 6371.0


def read_isc_csv(path: str) -> pd.DataFrame:
    """Read an ISC arrivals CSV, trimming leading commentary rows."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    header_line = None
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            if line.lstrip().startswith("EVENTID"):
                header_line = idx
                break

    if header_line is None:
        raise ValueError("Could not locate header row starting with 'EVENTID'.")

    df = pd.read_csv(
        file_path,
        skiprows=header_line,
        dtype=str,
        skipinitialspace=True,
    )
    normalized_columns: List[str] = []
    for col in df.columns:
        name = re.sub(r"\s+", " ", str(col).strip().upper())
        name = name.replace(" .", ".")
        normalized_columns.append(name)
    df.columns = normalized_columns
    LOGGER.info("Read %d rows from %s", len(df), file_path)
    return df


def _get_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[pd.Series]:
    """Return the first matching column from a list of candidate names."""
    for name in candidates:
        if name in df.columns:
            return df[name]
    return None


def extract_columns(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Extract ISC fields, handling observed column naming quirks."""
    required_fields = {
        "event_id": ("EVENTID",),
        "phase": ("REPPHASE", "PHASE"),
        "arrival_date": ("DATE",),
        "arrival_time": ("TIME",),
        "station": ("STA", "STATION"),
        "station_lat": ("LAT", "STLA", "STLAT"),
        "station_lon": ("LON", "STLO", "STLON"),
        "origin_date": ("DATE.1", "ORIGIN DATE"),
        "origin_time": ("TIME.1", "ORIGIN TIME"),
        "origin_lat": ("LAT.1", "EVLA", "LAT.2"),
        "origin_lon": ("LON.1", "EVLO", "LON.2"),
        "depth": ("DEPTH", "EVDEP", "DEPTH.1"),
    }

    optional_fields = {
        "net": ("NET",),
        "elev": ("ELEV", "STEL"),
        "dist": ("DIST", "DELTA"),
        "mag": ("MAG", "MAGNITUDE"),
        "res": ("RES", "RESIDUAL"),
    }

    mapped: Dict[str, pd.Series] = {}

    for key, candidates in required_fields.items():
        series = _get_col(df, candidates)
        if series is None:
            raise ValueError(f"Missing required column(s) for '{key}': {candidates}")
        mapped[key] = series

    for key, candidates in optional_fields.items():
        series = _get_col(df, candidates)
        if series is not None:
            mapped[key] = series

    return mapped


def _parse_date_series(values: pd.Series) -> pd.Series:
    """Parse ISC date strings (MM/DD/YYYY) into pandas Timestamps."""
    series = values.astype(str).str.strip()
    parsed = pd.to_datetime(series, format="%m/%d/%Y", errors="coerce")
    if parsed.isna().any():
        parsed = parsed.fillna(pd.to_datetime(series, errors="coerce"))
    return parsed


def _time_str_to_seconds(value: object) -> Optional[float]:
    """Convert a mixed-format time string to seconds."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NAN", "NA"}:
        return None

    sign = 1.0
    if text.startswith("-"):
        sign = -1.0
        text = text[1:].strip()

    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours = float(parts[0]) if parts[0] else 0.0
            minutes = float(parts[1]) if parts[1] else 0.0
            seconds = float(parts[2]) if parts[2] else 0.0
        elif len(parts) == 2:
            hours = 0.0
            minutes = float(parts[0]) if parts[0] else 0.0
            seconds = float(parts[1]) if parts[1] else 0.0
        elif len(parts) == 1:
            hours = 0.0
            minutes = 0.0
            seconds = float(parts[0]) if parts[0] else 0.0
        else:
            return None
    except ValueError:
        return None

    return sign * (hours * 3600.0 + minutes * 60.0 + seconds)


def build_src_rec_df(mapped: Dict[str, pd.Series], args: argparse.Namespace) -> pd.DataFrame:
    """Transform raw ISC columns into the PyTomoATT-friendly schema."""
    df = pd.DataFrame(mapped).copy()
    n_before = len(df)

    station_raw = df["station"].fillna("").astype(str).str.strip()
    station_raw = station_raw.replace({"nan": "", "NONE": "", "None": ""})
    staname = station_raw.copy()
    if args.name_netsta and "net" in df:
        net = df["net"].fillna("").astype(str).str.strip()
        net = net.replace({"nan": "", "NONE": "", "None": ""})
        staname = net.where(net == "", net + ".") + staname
    df["staname"] = staname

    df["stla"] = pd.to_numeric(df["station_lat"], errors="coerce")
    df["stlo"] = pd.to_numeric(df["station_lon"], errors="coerce")
    df["stel"] = pd.to_numeric(df.get("elev", 0.0), errors="coerce").fillna(0.0)

    df["evla"] = pd.to_numeric(df["origin_lat"], errors="coerce")
    df["evlo"] = pd.to_numeric(df["origin_lon"], errors="coerce")
    df["evdp"] = pd.to_numeric(df["depth"], errors="coerce")

    phase = df["phase"].fillna("").astype(str).str.strip()
    if args.phases_anycase:
        phase = phase.str.upper()
    df["phase"] = phase

    df["weight"] = 1.0
    if "res" in df.columns:
        res = pd.to_numeric(df["res"], errors="coerce")
        weights = 1.0 / (1.0 + res.abs())
        df["weight"] = weights.clip(lower=0.1, upper=1.0).fillna(1.0)

    if "mag" in df.columns:
        df["mag"] = pd.to_numeric(df["mag"], errors="coerce")

    if args.use_dist_col and "dist" in df.columns:
        df["dist_deg"] = pd.to_numeric(df["dist"], errors="coerce")

    arrival_date = _parse_date_series(df["arrival_date"])
    origin_date = _parse_date_series(df["origin_date"])
    raw_arrival_time = df["arrival_time"].astype(str).str.strip()
    raw_origin_time = df["origin_time"].astype(str).str.strip()
    arrival_seconds = pd.to_numeric(raw_arrival_time.map(_time_str_to_seconds), errors="coerce")
    origin_seconds = pd.to_numeric(raw_origin_time.map(_time_str_to_seconds), errors="coerce")

    arrival_datetime = arrival_date + pd.to_timedelta(arrival_seconds, unit="s")
    origin_datetime = origin_date + pd.to_timedelta(origin_seconds, unit="s")
    origin_iso = origin_datetime.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").where(origin_datetime.notna())
    origin_iso = origin_iso.str.rstrip("0").str.rstrip(".")
    df["origin_time"] = origin_iso.where(origin_iso.isna(), origin_iso + "Z")

    tt_seconds = (arrival_datetime - origin_datetime).dt.total_seconds()

    arrival_colons = raw_arrival_time.str.count(":")
    origin_colons = raw_origin_time.str.count(":")
    minute_second_mask = (arrival_colons == 1) & (origin_colons == 1)

    adjust_mask = tt_seconds.notna() & (tt_seconds < 0) & minute_second_mask
    tt_seconds = tt_seconds.where(~adjust_mask, tt_seconds + 3600.0)

    adjust_mask = tt_seconds.notna() & (tt_seconds < 0)
    tt_seconds = tt_seconds.where(~adjust_mask, tt_seconds + 86400.0)

    df["tt"] = tt_seconds

    mask_valid = (
        tt_seconds.notna()
        & origin_datetime.notna()
        & arrival_datetime.notna()
        & df["stla"].notna()
        & df["stlo"].notna()
        & df["evla"].notna()
        & df["evlo"].notna()
        & df["evdp"].notna()
    )
    mask_valid &= tt_seconds.between(args.min_tt_sec, args.max_tt_sec)

    if args.lat_range:
        lat_min, lat_max = sorted(args.lat_range)
        mask_valid &= df["stla"].between(lat_min, lat_max)
    if args.lon_range:
        lon_min, lon_max = sorted(args.lon_range)
        mask_valid &= df["stlo"].between(lon_min, lon_max)
    if args.depth_range:
        dep_min, dep_max = sorted(args.depth_range)
        mask_valid &= df["evdp"].between(dep_min, dep_max)

    df = df.loc[mask_valid].copy()
    LOGGER.info("Preprocessing retained %d/%d arrivals after QC filters.", len(df), n_before)

    out_cols: List[str] = [
        "staname",
        "stla",
        "stlo",
        "stel",
        "phase",
        "tt",
        "weight",
        "event_id",
        "origin_time",
        "evla",
        "evlo",
        "evdp",
    ]

    if args.use_dist_col and "dist_deg" in df.columns:
        out_cols.append("dist_deg")
    if "mag" in df.columns:
        out_cols.append("mag")

    missing_cols = [col for col in out_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns after processing: {missing_cols}")

    return df[out_cols].reset_index(drop=True)


def _compute_default_region_bounds(arrivals: pd.DataFrame) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
    """Return min/max lat/lon derived from stations and events."""
    lat_series = pd.concat([arrivals["stla"], arrivals["evla"]], ignore_index=True)
    lon_series = pd.concat([arrivals["stlo"], arrivals["evlo"]], ignore_index=True)
    lat_series = pd.to_numeric(lat_series, errors="coerce")
    lon_series = pd.to_numeric(lon_series, errors="coerce")

    lat_bounds = None
    lon_bounds = None
    if lat_series.notna().any():
        lat_bounds = (float(lat_series.min()), float(lat_series.max()))
    if lon_series.notna().any():
        lon_bounds = (float(lon_series.min()), float(lon_series.max()))
    return lat_bounds, lon_bounds


def _prompt_for_csv_choice(candidates: List[Path], search_dir: Path) -> Path:
    """Interactively prompt to select a CSV file from available candidates."""
    print("Available CSV files:")
    for idx, path in enumerate(candidates, start=1):
        try:
            rel_path = path.relative_to(search_dir)
        except ValueError:
            rel_path = path
        print(f"  [{idx}] {rel_path}")

    default_choice = candidates[0]
    prompt = f"Select CSV by number or enter a path [{default_choice.name}]: "
    while True:
        response = input(prompt).strip()
        if not response:
            return default_choice
        if response.isdigit():
            choice = int(response)
            if 1 <= choice <= len(candidates):
                return candidates[choice - 1]
            print(f"Please choose a number between 1 and {len(candidates)}.")
            continue
        candidate = Path(response).expanduser()
        if not candidate.is_absolute():
            candidate = search_dir / candidate
        if candidate.exists() and candidate.is_file():
            return candidate
        print("Invalid selection; provide an existing file path or valid index.")


def resolve_csv_path(args: argparse.Namespace) -> Path:
    """Determine which CSV file to use, optionally prompting the user."""
    select_mode = args.choose_csv
    csv_candidate: Optional[Path] = None
    if args.csv:
        csv_candidate = Path(args.csv).expanduser()
        if csv_candidate.exists() and csv_candidate.is_file() and not select_mode:
            return csv_candidate
        if csv_candidate.exists() and csv_candidate.is_file():
            LOGGER.info("CSV %s found; entering interactive selection as requested.", csv_candidate)
        else:
            LOGGER.warning("CSV %s not found; entering interactive selection.", csv_candidate or args.csv)
        select_mode = True

    search_dir = Path(args.csv_dir).expanduser()
    if not search_dir.exists():
        raise FileNotFoundError(f"CSV search directory does not exist: {search_dir}")
    if search_dir.is_file():
        candidates = [search_dir]
    else:
        candidates = sorted(search_dir.glob("*.csv"))
    if csv_candidate and csv_candidate.exists() and csv_candidate.is_file():
        if csv_candidate not in candidates:
            candidates = [csv_candidate] + candidates
        if not select_mode:
            return csv_candidate

    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {search_dir} and --csv was not usable.")

    if not select_mode:
        LOGGER.info("Using CSV %s", candidates[0])
        return candidates[0]

    if not sys.stdin or not sys.stdin.isatty():
        LOGGER.info("Non-interactive environment; defaulting to %s", candidates[0])
        return candidates[0]

    return _prompt_for_csv_choice(candidates, search_dir)


def build_src_rec_object(arrivals: pd.DataFrame, args: argparse.Namespace, out_dat: str | Path) -> SrcRec:
    """Create a SrcRec instance from the arrival dataframe."""
    _require_pytomoatt()
    sr = SrcRec(fname=str(out_dat), src_only=False)
    cs_cols, _ = setup_rec_points_dd("cs")
    cr_cols, _ = setup_rec_points_dd("cr")
    sr.rec_points_cr = pd.DataFrame(columns=cr_cols)
    sr.rec_points_cs = pd.DataFrame(columns=cs_cols)

    src_rows: List[Dict[str, object]] = []
    rec_rows: List[Dict[str, object]] = []

    grouped = arrivals.groupby("event_id", sort=False)
    for idx, (event_id, group) in enumerate(grouped):
        first = group.iloc[0]
        origin_dt = pd.to_datetime(first["origin_time"], utc=True, errors="coerce")
        if pd.isna(origin_dt):
            continue

        mag_value = (
            float(group["mag"].dropna().iloc[0])
            if "mag" in group.columns and group["mag"].notna().any()
            else math.nan
        )
        event_weight = float(group["weight"].mean()) if group["weight"].notna().any() else 1.0

        src_rows.append(
            {
                "src_index": idx,
                "origin_time": origin_dt.to_pydatetime(),
                "evla": float(first["evla"]),
                "evlo": float(first["evlo"]),
                "evdp": float(first["evdp"]),
                "mag": mag_value,
                "num_rec": int(len(group)),
                "event_id": str(event_id),
                "weight": event_weight,
            }
        )

        for rec_idx, rec in enumerate(group.itertuples(index=False), start=0):
            rec_entry = {
                "src_index": idx,
                "rec_index": rec_idx,
                "staname": rec.staname,
                "stla": float(rec.stla),
                "stlo": float(rec.stlo),
                "stel": float(rec.stel),
                "phase": rec.phase,
            }
            if args.use_dist_col and "dist_deg" in arrivals.columns:
                rec_entry["dist_deg"] = float(getattr(rec, "dist_deg"))
            rec_entry["tt"] = float(rec.tt)
            rec_entry["weight"] = float(rec.weight)
            rec_rows.append(rec_entry)

    if not src_rows or not rec_rows:
        raise ValueError("Cannot build SrcRec object with empty source or receiver data.")

    src_points = pd.DataFrame(src_rows).set_index("src_index")
    rec_points = pd.DataFrame(rec_rows)

    numeric_rec_cols = ["stla", "stlo", "stel", "tt", "weight"]
    if args.use_dist_col and "dist_deg" in rec_points.columns:
        numeric_rec_cols.append("dist_deg")
    rec_points[numeric_rec_cols] = rec_points[numeric_rec_cols].apply(pd.to_numeric, errors="coerce")

    sr.src_points = src_points
    sr.rec_points = rec_points
    sr.update()
    return sr


def _downsample_dataframe(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Randomly down-sample a dataframe to a maximum number of rows."""
    if max_points <= 0 or len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=42).reset_index(drop=True)


def _compute_map_limits(stations: pd.DataFrame, events: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute plotting bounds covering both stations and events with padding."""
    lon_values: List[float] = []
    lat_values: List[float] = []
    if not stations.empty:
        lon_values.extend(stations["stlo"].astype(float).tolist())
        lat_values.extend(stations["stla"].astype(float).tolist())
    if not events.empty:
        lon_values.extend(events["evlo"].astype(float).tolist())
        lat_values.extend(events["evla"].astype(float).tolist())

    if not lon_values or not lat_values:
        return (-180.0, 180.0), (-90.0, 90.0)

    lon_min = min(lon_values)
    lon_max = max(lon_values)
    lat_min = min(lat_values)
    lat_max = max(lat_values)

    lon_span = max(lon_max - lon_min, 1e-6)
    lat_span = max(lat_max - lat_min, 1e-6)
    lon_margin = max(0.1 * lon_span, 0.5)
    lat_margin = max(0.1 * lat_span, 0.5)

    lon_limits = (max(-360.0, lon_min - lon_margin), min(360.0, lon_max + lon_margin))
    lat_limits = (max(-90.0, lat_min - lat_margin), min(90.0, lat_max + lat_margin))
    return lon_limits, lat_limits


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in kilometers."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    sin_dlat = np.sin(dlat / 2.0)
    sin_dlon = np.sin(dlon / 2.0)
    a = sin_dlat**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * sin_dlon**2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _average_source_receiver_distance(sr: SrcRec) -> float:
    """Average source-receiver distance in km."""
    rec = sr.rec_points
    if rec.empty or sr.src_points.empty:
        return float("nan")
    src_subset = sr.src_points[["evla", "evlo"]]
    src_lats = src_subset.loc[rec["src_index"], "evla"].to_numpy(dtype=float, copy=False)
    src_lons = src_subset.loc[rec["src_index"], "evlo"].to_numpy(dtype=float, copy=False)
    rec_lats = rec["stla"].to_numpy(dtype=float, copy=False)
    rec_lons = rec["stlo"].to_numpy(dtype=float, copy=False)
    distances = _haversine_km(src_lats, src_lons, rec_lats, rec_lons)
    if distances.size == 0:
        return float("nan")
    return float(np.nanmean(distances))


def _average_receiver_spacing(sr: SrcRec) -> float:
    """Average nearest-neighbor spacing between unique receivers."""
    rec = sr.rec_points
    if rec.empty:
        return float("nan")
    unique_rec = rec.drop_duplicates(subset=["staname"])
    if len(unique_rec) < 2:
        return float("nan")
    lat = unique_rec["stla"].to_numpy(dtype=float, copy=False)
    lon = unique_rec["stlo"].to_numpy(dtype=float, copy=False)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    dlat = lat_rad[:, None] - lat_rad
    dlon = lon_rad[:, None] - lon_rad
    sin_dlat = np.sin(dlat / 2.0)
    sin_dlon = np.sin(dlon / 2.0)
    cos_lat = np.cos(lat_rad)
    a = sin_dlat**2 + cos_lat[:, None] * cos_lat[None, :] * sin_dlon**2
    np.clip(a, 0.0, 1.0, out=a)
    distances = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    # Ignore self-distance and use only the closest receiver for each station.
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    finite = np.isfinite(nearest)
    if not finite.any():
        return float("nan")
    return float(np.nanmean(nearest[finite]))


def _plot_lateral_depth_connections(sr: SrcRec, output_path: Path, max_segments: int = 20000) -> None:
    """Plot latitude-depth connections similar to the event vertical section."""
    rec = sr.rec_points
    if rec.empty or sr.src_points.empty:
        return
    src_subset = (
        sr.src_points[["evla", "evlo", "evdp"]]
        .reset_index()
        .rename(columns={"evla": "src_lat", "evlo": "src_lon", "evdp": "src_dep"})
    )
    merged = rec.merge(src_subset, on="src_index", how="inner")
    required_cols = ["src_lat", "src_lon", "src_dep", "stla", "stlo"]
    merged = merged.dropna(subset=required_cols)
    if merged.empty:
        return
    sample_count = min(len(merged), max_segments)
    if len(merged) > sample_count:
        merged = merged.sample(n=sample_count, random_state=42)
    src_lat = merged["src_lat"].to_numpy(dtype=float, copy=False)
    src_lon = merged["src_lon"].to_numpy(dtype=float, copy=False)
    src_dep = merged["src_dep"].to_numpy(dtype=float, copy=False)
    rec_lat = merged["stla"].to_numpy(dtype=float, copy=False)

    starts = np.column_stack((rec_lat, np.zeros_like(rec_lat)))
    ends = np.column_stack((src_lat, src_dep))
    segments = np.stack([starts, ends], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    coll = LineCollection(segments, colors="0.5", linewidths=0.5, alpha=0.35)
    ax.add_collection(coll)
    norm = Normalize(vmin=np.nanmin(src_lon), vmax=np.nanmax(src_lon))
    ax.scatter(
        src_lat,
        src_dep,
        s=12,
        c=src_lon,
        cmap="plasma",
        norm=norm,
        marker="o",
        edgecolor="none",
        alpha=0.8,
        label="Events",
    )
    ax.scatter(
        rec_lat,
        np.zeros_like(rec_lat),
        s=16,
        c="tab:blue",
        marker="^",
        edgecolor="black",
        linewidth=0.2,
        alpha=0.7,
        label="Stations",
    )
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Depth [km]")
    ax.set_title("Source-receiver connections (lat-depth)")
    lat_values = np.concatenate([src_lat, rec_lat])
    lat_min = float(np.nanmin(lat_values))
    lat_max = float(np.nanmax(lat_values))
    lat_margin = max(0.02 * (lat_max - lat_min), 0.25)
    ax.set_xlim(lat_min - lat_margin, lat_max + lat_margin)
    depth_max = float(np.nanmax(src_dep)) if np.isfinite(np.nanmax(src_dep)) else 0.0
    depth_ylim = max(depth_max * 1.1, 1.0)
    ax.set_ylim(depth_ylim, -5.0)
    ax.invert_yaxis()
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="plasma"), ax=ax, shrink=0.8)
    cb.set_label("Event longitude [deg]")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved source-receiver connection plot: %s", output_path)


def _plot_longitudinal_depth_connections(sr: SrcRec, output_path: Path, max_segments: int = 20000) -> None:
    """Plot longitude-depth source-receiver connections."""
    rec = sr.rec_points
    if rec.empty or sr.src_points.empty:
        return
    src_subset = (
        sr.src_points[["evla", "evlo", "evdp"]]
        .reset_index()
        .rename(columns={"evla": "src_lat", "evlo": "src_lon", "evdp": "src_dep"})
    )
    merged = rec.merge(src_subset, on="src_index", how="inner")
    required_cols = ["src_lat", "src_lon", "src_dep", "stla", "stlo"]
    merged = merged.dropna(subset=required_cols)
    if merged.empty:
        return
    sample_count = min(len(merged), max_segments)
    if len(merged) > sample_count:
        merged = merged.sample(n=sample_count, random_state=42)

    src_lat = merged["src_lat"].to_numpy(dtype=float, copy=False)
    src_lon = merged["src_lon"].to_numpy(dtype=float, copy=False)
    src_dep = merged["src_dep"].to_numpy(dtype=float, copy=False)
    rec_lon = merged["stlo"].to_numpy(dtype=float, copy=False)

    starts = np.column_stack((rec_lon, np.zeros_like(rec_lon)))
    ends = np.column_stack((src_lon, src_dep))
    segments = np.stack([starts, ends], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    coll = LineCollection(segments, colors="0.5", linewidths=0.5, alpha=0.35)
    ax.add_collection(coll)
    norm = Normalize(vmin=np.nanmin(src_lat), vmax=np.nanmax(src_lat))
    ax.scatter(
        src_lon,
        src_dep,
        s=12,
        c=src_lat,
        cmap="viridis",
        norm=norm,
        marker="o",
        edgecolor="none",
        alpha=0.8,
        label="Events",
    )
    ax.scatter(
        rec_lon,
        np.zeros_like(rec_lon),
        s=16,
        c="tab:orange",
        marker="^",
        edgecolor="black",
        linewidth=0.2,
        alpha=0.7,
        label="Stations",
    )
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Depth [km]")
    ax.set_title("Source-receiver connections (lon-depth)")
    lon_values = np.concatenate([src_lon, rec_lon])
    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))
    lon_margin = max(0.02 * (lon_max - lon_min), 0.25)
    ax.set_xlim(lon_min - lon_margin, lon_max + lon_margin)
    depth_max = float(np.nanmax(src_dep)) if np.isfinite(np.nanmax(src_dep)) else 0.0
    depth_ylim = max(depth_max * 1.1, 1.0)
    ax.set_ylim(depth_ylim, -5.0)
    ax.invert_yaxis()
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, shrink=0.8)
    cb.set_label("Event latitude [deg]")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved source-receiver longitude-depth plot: %s", output_path)


def _iter_geometry_coords(geometry) -> Iterable[np.ndarray]:
    try:
        from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
    except Exception:
        return []

    if isinstance(geometry, Polygon):
        yield np.asarray(geometry.exterior.coords)
        for interior in geometry.interiors:
            interior_coords = np.asarray(interior.coords)
            if interior_coords.size:
                yield interior_coords
    elif isinstance(geometry, MultiPolygon):
        for part in geometry.geoms:
            yield from _iter_geometry_coords(part)
    elif isinstance(geometry, LineString):
        yield np.asarray(geometry.coords)
    elif isinstance(geometry, MultiLineString):
        for part in geometry.geoms:
            yield np.asarray(part.coords)
    else:
        try:
            geoms = geometry.geoms
        except AttributeError:
            return []
        for part in geoms:
            yield from _iter_geometry_coords(part)


def load_map_borders(lon_limits: tuple[float, float], lat_limits: tuple[float, float]) -> list[np.ndarray]:
    """Load Natural Earth country borders clipped to the plotting window."""
    try:
        from cartopy.io import shapereader
        from shapely.geometry import box
    except Exception as exc:
        warnings.warn(
            f"Skipping map border overlay (cartopy/shapely not available: {exc}).",
            RuntimeWarning,
            stacklevel=1,
        )
        return []

    region = box(
        lon_limits[0] - 1.0,
        lat_limits[0] - 1.0,
        lon_limits[1] + 1.0,
        lat_limits[1] + 1.0,
    )

    try:
        shp_path = shapereader.natural_earth(
            resolution=BORDER_RESOLUTION,
            category="cultural",
            name="admin_0_countries",
        )
        reader = shapereader.Reader(shp_path)
    except Exception as exc:
        warnings.warn(
            f"Could not read Natural Earth borders: {exc}",
            RuntimeWarning,
            stacklevel=1,
        )
        return []

    map_lines: list[np.ndarray] = []
    for record in reader.records():
        geometry = record.geometry
        try:
            clipped = geometry.intersection(region)
        except Exception:
            continue
        for coords in _iter_geometry_coords(clipped):
            if coords.size:
                map_lines.append(coords)
    return map_lines


def _plot_region_box(ax, region_box: tuple[float, float, float, float], **kwargs) -> None:
    """Draw a rectangular region specified by (lon_min, lon_max, lat_min, lat_max)."""
    lon_min, lon_max, lat_min, lat_max = region_box
    box = np.array(
        [
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ],
        dtype=float,
    )
    ax.plot(box[:, 0], box[:, 1], **kwargs)


def _plot_station_map_matplotlib(
    stations: pd.DataFrame,
    lon_limits: tuple[float, float],
    lat_limits: tuple[float, float],
    map_lines: Iterable[np.ndarray],
    output_path: Path,
    region_box: tuple[float, float, float, float] | None = None,
) -> None:
    if stations.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"Receiver distribution (N={len(stations)})")
    ax.set_xlim(lon_limits)
    ax.set_ylim(lat_limits)
    mean_lat = 0.5 * sum(lat_limits)
    ax.set_box_aspect(1.0 / max(math.cos(math.radians(mean_lat)), 1e-6))

    bbox_limits = (lon_limits[0], lon_limits[1], lat_limits[0], lat_limits[1])
    _plot_region_box(ax, bbox_limits, color="black", linewidth=1.0, linestyle="--")
    if region_box is not None:
        _plot_region_box(ax, region_box, color="red", linewidth=1.2, linestyle="-", label="Input region")
    for coords in map_lines:
        if coords.size:
            ax.plot(coords[:, 0], coords[:, 1], color="0.3", linewidth=0.5)

    ax.scatter(
        stations["stlo"],
        stations["stla"],
        marker="^",
        s=40,
        facecolor="#1f77b4",
        edgecolor="black",
        linewidth=0.4,
    )
    if region_box is not None:
        ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved Matplotlib station map: %s", output_path)


def _plot_event_plan_matplotlib(
    events: pd.DataFrame,
    lon_limits: tuple[float, float],
    lat_limits: tuple[float, float],
    map_lines: Iterable[np.ndarray],
    output_path: Path,
    region_box: tuple[float, float, float, float] | None = None,
) -> None:
    if events.empty:
        return

    events_plot = _downsample_dataframe(events, MAX_SCATTER_POINTS)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"Event lateral distribution (sampled N={len(events_plot)})")
    ax.set_xlim(lon_limits)
    ax.set_ylim(lat_limits)
    mean_lat = 0.5 * sum(lat_limits)
    ax.set_box_aspect(1.0 / max(math.cos(math.radians(mean_lat)), 1e-6))

    bbox_limits = (lon_limits[0], lon_limits[1], lat_limits[0], lat_limits[1])
    _plot_region_box(ax, bbox_limits, color="black", linewidth=1.0, linestyle="--")
    if region_box is not None:
        _plot_region_box(ax, region_box, color="red", linewidth=1.2, linestyle="-", label="Input region")
    for coords in map_lines:
        if coords.size:
            ax.plot(coords[:, 0], coords[:, 1], color="0.3", linewidth=0.5)

    depths = events_plot["evdp"].astype(float).to_numpy()
    norm = Normalize(vmin=np.nanmin(depths), vmax=np.nanmax(depths))
    scatter = ax.scatter(
        events_plot["evlo"],
        events_plot["evla"],
        c=depths,
        cmap="viridis",
        norm=norm,
        s=12,
        alpha=0.6,
        edgecolors="none",
    )
    cb = fig.colorbar(scatter, ax=ax, shrink=0.75)
    cb.set_label("Event depth [km]")
    if region_box is not None:
        ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved Matplotlib event map: %s", output_path)


def _plot_event_vertical_matplotlib(events: pd.DataFrame, output_path: Path) -> None:
    if events.empty:
        return

    events_plot = _downsample_dataframe(events, MAX_SCATTER_POINTS)
    lats = events_plot["evla"].astype(float).to_numpy()
    depths = events_plot["evdp"].astype(float).to_numpy()
    lons = events_plot["evlo"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Depth [km]")
    ax.set_title(f"Event vertical distribution (sampled N={len(events_plot)})")
    ax.invert_yaxis()

    norm = Normalize(vmin=np.nanmin(lons), vmax=np.nanmax(lons))
    scatter = ax.scatter(lats, depths, c=lons, cmap="plasma", norm=norm, s=10, alpha=0.6, edgecolors="none")
    cb = fig.colorbar(scatter, ax=ax, shrink=0.75)
    cb.set_label("Longitude [deg]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved Matplotlib event vertical map: %s", output_path)


def _generate_coverage_figures(
    sr: SrcRec,
    fig_dir: Path,
    region_box: tuple[float, float, float, float] | None = None,
) -> None:
    """Create coverage figures for stations and events."""
    station_cols = ["staname", "stla", "stlo", "stel"]
    event_cols = ["event_id", "evla", "evlo", "evdp"]

    stations = pd.DataFrame(columns=station_cols)
    events = pd.DataFrame(columns=event_cols)

    if hasattr(sr, "rec_points") and not sr.rec_points.empty:
        stations = (
            sr.rec_points.loc[:, station_cols]
            .dropna(subset=["stla", "stlo"])
            .drop_duplicates(subset=["staname"])
            .reset_index(drop=True)
        )
    if hasattr(sr, "src_points") and not sr.src_points.empty:
        events = (
            sr.src_points.loc[:, event_cols]
            .dropna(subset=["evla", "evlo", "evdp"])
            .reset_index(drop=True)
        )

    if stations.empty and events.empty:
        LOGGER.info("No stations or events available for coverage plots; skipping figure generation.")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    lon_limits, lat_limits = _compute_map_limits(stations, events)
    map_lines = load_map_borders(lon_limits, lat_limits)

    if not stations.empty:
        _plot_station_map_matplotlib(
            stations,
            lon_limits,
            lat_limits,
            map_lines,
            fig_dir / "stations_map.png",
            region_box=region_box,
        )
    if not events.empty:
        _plot_event_plan_matplotlib(
            events,
            lon_limits,
            lat_limits,
            map_lines,
            fig_dir / "events_plan.png",
            region_box=region_box,
        )
        _plot_event_vertical_matplotlib(events, fig_dir / "events_vertical.png")


def _print_sr_stats(sr: SrcRec, stage: str) -> None:
    """Print selection statistics."""
    n_arrivals = int(sr.rec_points.shape[0]) if hasattr(sr, "rec_points") else 0
    n_stations = (
        int(sr.rec_points["staname"].nunique())
        if hasattr(sr, "rec_points") and not sr.rec_points.empty
        else 0
    )
    n_events = int(sr.src_points.shape[0]) if hasattr(sr, "src_points") else 0
    LOGGER.info(
        "%s: %s arrivals, %s stations, %s events",
        stage,
        n_arrivals,
        n_stations,
        n_events,
    )


def _format_stage_label(label: str, message: str) -> str:
    """Create a consistent log prefix for a data subset."""
    return f"[{label}] {message}" if label else message


def apply_standard_filters(sr: SrcRec, args: argparse.Namespace, label: str = "") -> bool:
    """Apply phase/min-station/distance filters and log summary statistics."""
    stage = lambda msg: _format_stage_label(label, msg)
    _print_sr_stats(sr, stage("Initial load"))

    if not args.use_dist_col:
        sr.calc_distaz()
        LOGGER.debug("%s", stage("Computed distances and azimuths."))

    phases = [ph.strip().upper() if args.phases_anycase else ph.strip() for ph in args.phases.split(",")]
    phases = [ph for ph in phases if ph]
    if phases:
        sr.select_by_phase(phases)
        sr.update()
        _print_sr_stats(sr, stage(f"After phase filter {phases}"))

    if args.min_sta > 1:
        sr.select_by_num_rec(args.min_sta)
        sr.update()
        _print_sr_stats(sr, stage(f"After min stations filter ({args.min_sta})"))

    sr.select_by_distance([args.dist_min, args.dist_max])
    sr.update()
    _print_sr_stats(sr, stage(f"After distance filter [{args.dist_min}, {args.dist_max}]"))

    if sr.rec_points.empty:
        LOGGER.warning("%s No arrivals remain after filtering.", stage("Subset"))
        return False

    avg_sr_km = _average_source_receiver_distance(sr)
    avg_rr_km = _average_receiver_spacing(sr)
    if math.isfinite(avg_sr_km):
        LOGGER.info("%s %.1f km", stage("Average source-receiver distance"), avg_sr_km)
    else:
        LOGGER.info("%s N/A", stage("Average source-receiver distance"))
    if math.isfinite(avg_rr_km):
        LOGGER.info("%s %.1f km", stage("Average receiver spacing"), avg_rr_km)
    else:
        LOGGER.info("%s N/A", stage("Average receiver spacing"))

    return True


isc_tools = SimpleNamespace(
    read_isc_csv=read_isc_csv,
    extract_columns=extract_columns,
    build_src_rec_df=build_src_rec_df,
    build_src_rec_object=build_src_rec_object,
    apply_standard_filters=apply_standard_filters,
    resolve_csv_path=resolve_csv_path,
    _compute_default_region_bounds=_compute_default_region_bounds,
    _generate_coverage_figures=_generate_coverage_figures,
    _plot_lateral_depth_connections=_plot_lateral_depth_connections,
    _plot_longitudinal_depth_connections=_plot_longitudinal_depth_connections,
)


# --- Pipeline orchestration ---


@dataclass(frozen=True)
class RegionBounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    depth_min: float
    depth_max: float

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.lat_min + self.lat_max), 0.5 * (self.lon_min + self.lon_max))

    def to_dict(self) -> Dict[str, float]:
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "depth_min": self.depth_min,
            "depth_max": self.depth_max,
        }


def _normalize_phase_list(phases: str, include_s: bool) -> str:
    tokens = [tok.strip() for tok in phases.split(",") if tok.strip()]
    if include_s:
        normalized = {tok.upper() for tok in tokens}
        if "S" not in normalized:
            tokens.append("S")
    if not tokens:
        tokens = ["P"]
    return ",".join(tokens)


def build_arrivals_from_src_rec(sr) -> pd.DataFrame:
    """Merge SrcRec receiver and source tables into a single arrivals dataframe."""
    rec = sr.rec_points.copy()
    if "staname" not in rec.columns and "sta" in rec.columns:
        rec = rec.rename(columns={"sta": "staname"})
    if "tt" not in rec.columns and "travel_time" in rec.columns:
        rec = rec.rename(columns={"travel_time": "tt"})
    if "stel" not in rec.columns:
        rec["stel"] = 0.0
    if "weight" not in rec.columns:
        rec["weight"] = 1.0
    if "phase" not in rec.columns:
        raise ValueError("src_rec missing phase information for receiver rows.")

    src = sr.src_points.reset_index()
    if "event_id" not in src.columns:
        src["event_id"] = src["src_index"].astype(str)
    required_src = ["event_id", "origin_time", "evla", "evlo", "evdp"]
    missing_src = [col for col in required_src if col not in src.columns]
    if missing_src:
        raise ValueError(f"src_rec missing required source fields: {missing_src}")

    src_cols = ["event_id", "origin_time", "evla", "evlo", "evdp"]
    if "mag" in src.columns:
        src_cols.append("mag")
    src = src[["src_index"] + src_cols]
    arrivals = rec.merge(src, on="src_index", how="left", validate="many_to_one")
    return arrivals


def compute_full_bounds(arrivals: pd.DataFrame, args: argparse.Namespace) -> RegionBounds:
    """Derive full-region bounds from data unless explicitly supplied."""
    lat_bounds, lon_bounds = isc_tools._compute_default_region_bounds(arrivals)
    if lat_bounds is None or lon_bounds is None:
        raise ValueError("Could not compute lat/lon bounds from arrivals.")

    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    if args.lat_range:
        lat_min, lat_max = sorted(args.lat_range)
    if args.lon_range:
        lon_min, lon_max = sorted(args.lon_range)

    depth_series = pd.to_numeric(arrivals["evdp"], errors="coerce")
    depth_min = float(depth_series.min(skipna=True)) if depth_series.notna().any() else 0.0
    depth_max = float(depth_series.max(skipna=True)) if depth_series.notna().any() else 0.0
    if args.depth_range:
        depth_min, depth_max = sorted(args.depth_range)
    depth_min = max(0.0, depth_min)

    return RegionBounds(
        lat_min=float(lat_min),
        lat_max=float(lat_max),
        lon_min=float(lon_min),
        lon_max=float(lon_max),
        depth_min=float(depth_min),
        depth_max=float(depth_max),
    )


def _round_up_to_step(value: float, step: float) -> float:
    if not math.isfinite(value) or step <= 0:
        return value
    rounded = math.ceil(value / step) * step
    return round(rounded, 6)


def _compute_region_depth_max(arrivals: pd.DataFrame, fallback: float, step: float) -> float:
    depth_series = pd.to_numeric(arrivals["evdp"], errors="coerce")
    if not depth_series.notna().any():
        return fallback
    depth_max = float(depth_series.max(skipna=True))
    return _round_up_to_step(depth_max, step)


def compute_azimuth_bin_stats(
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    center_lat: float,
    center_lon: float,
    azimuth_bin_deg: float,
) -> Dict[str, object]:
    if azimuth_bin_deg <= 0:
        return {
            "azimuth_bins_total": 0,
            "azimuth_bins_populated": 0,
            "azimuth_bin_mean": 0.0,
            "azimuth_bin_std": 0.0,
            "azimuth_bin_min": 0,
            "azimuth_bin_max": 0,
        }

    num_bins = int(round(360.0 / azimuth_bin_deg))
    if num_bins <= 0:
        return {
            "azimuth_bins_total": 0,
            "azimuth_bins_populated": 0,
            "azimuth_bin_mean": 0.0,
            "azimuth_bin_std": 0.0,
            "azimuth_bin_min": 0,
            "azimuth_bin_max": 0,
        }

    counts = np.zeros(num_bins, dtype=int)
    for lat, lon in zip(station_lats, station_lons):
        az = ray_coverage.forward_azimuth(center_lat, center_lon, float(lat), float(lon))
        bin_idx = int(az / azimuth_bin_deg) % num_bins
        counts[bin_idx] += 1

    return {
        "azimuth_bins_total": int(num_bins),
        "azimuth_bins_populated": int((counts > 0).sum()),
        "azimuth_bin_mean": float(counts.mean()) if counts.size else 0.0,
        "azimuth_bin_std": float(counts.std()) if counts.size else 0.0,
        "azimuth_bin_min": int(counts.min()) if counts.size else 0,
        "azimuth_bin_max": int(counts.max()) if counts.size else 0,
    }


def compute_event_coverage_percent(
    region_arrivals: pd.DataFrame,
    region: "RegionBounds",
    bin_deg: float,
) -> float:
    if bin_deg <= 0:
        return 0.0
    if region_arrivals.empty:
        return 0.0
    events = region_arrivals.groupby("event_id", sort=False).agg(evla=("evla", "first"), evlo=("evlo", "first"))
    if events.empty:
        return 0.0
    lat_bins = max(1, int(math.ceil((region.lat_max - region.lat_min) / bin_deg)))
    lon_bins = max(1, int(math.ceil((region.lon_max - region.lon_min) / bin_deg)))
    lats = events["evla"].to_numpy(dtype=float, copy=False)
    lons = events["evlo"].to_numpy(dtype=float, copy=False)
    lat_idx = np.floor((lats - region.lat_min) / bin_deg).astype(int)
    lon_idx = np.floor((lons - region.lon_min) / bin_deg).astype(int)
    lat_idx = np.clip(lat_idx, 0, lat_bins - 1)
    lon_idx = np.clip(lon_idx, 0, lon_bins - 1)
    occupied = len(set(zip(lat_idx.tolist(), lon_idx.tolist())))
    total = lat_bins * lon_bins
    return 100.0 * occupied / total


RANKING_SORT_ORDER_DEFAULT: List[Tuple[str, bool]] = [
    ("azimuth_bins_populated", False),
    ("event_coverage_pct", False),
    ("azimuth_bin_std", True),
    ("region_id", True),
]


def _resolve_ranking_sort_order(summaries: List[Dict[str, object]]) -> List[Tuple[str, bool]]:
    require_all_bins = False
    for summary in summaries:
        if isinstance(summary.get("require_all_azimuth_bins"), bool):
            require_all_bins = bool(summary.get("require_all_azimuth_bins"))
            break
    if require_all_bins:
        return [
            ("event_coverage_pct", False),
            ("azimuth_bin_std", True),
            ("region_id", True),
        ]
    return RANKING_SORT_ORDER_DEFAULT


def _build_central_fraction_sequence(start: float, step: float = 0.05, max_fraction: float = 0.5) -> List[float]:
    if start <= 0.0 or start >= 1.0:
        return [start]
    start = max(start, step)
    if start > max_fraction:
        return [start]
    values: List[float] = []
    current = start
    while current <= max_fraction + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def has_good_azimuth_coverage(
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    center_lat: float,
    center_lon: float,
    azimuth_bin_deg: float,
    min_stations_per_bin: int,
    require_all_bins: bool,
) -> bool:
    """Check azimuth bin coverage for stations around a region center."""
    if station_lats.size == 0:
        return False
    if azimuth_bin_deg <= 0:
        raise ValueError("azimuth_bin_deg must be positive.")

    num_bins = int(round(360.0 / azimuth_bin_deg))
    if num_bins <= 0:
        return False
    counts = [0] * num_bins
    for lat, lon in zip(station_lats, station_lons):
        az = ray_coverage.forward_azimuth(center_lat, center_lon, float(lat), float(lon))
        bin_idx = int(az / azimuth_bin_deg) % num_bins
        counts[bin_idx] += 1

    if require_all_bins:
        return all(count >= min_stations_per_bin for count in counts)

    for count in counts:
        if count > 0 and count < min_stations_per_bin:
            return False
    return True


def find_candidate_regions(
    stations: pd.DataFrame,
    full_bounds: RegionBounds,
    region_width_deg: float,
    region_height_deg: float,
    region_step_deg: float,
    azimuth_bin_deg: float,
    min_stations_per_bin: int,
    require_all_bins: bool,
) -> List[RegionBounds]:
    """Slide a rectangular window across the full bounds and keep good-coverage regions."""
    if region_width_deg <= 0 or region_height_deg <= 0:
        raise ValueError("Region width/height must be positive.")
    if region_step_deg <= 0:
        raise ValueError("Region step must be positive.")

    lat_max = full_bounds.lat_max
    lon_max = full_bounds.lon_max
    lat_vals = np.arange(full_bounds.lat_min, lat_max - region_height_deg + 1e-9, region_step_deg)
    lon_vals = np.arange(full_bounds.lon_min, lon_max - region_width_deg + 1e-9, region_step_deg)

    st_lats = stations["stla"].to_numpy(dtype=float, copy=False)
    st_lons = stations["stlo"].to_numpy(dtype=float, copy=False)

    candidates: List[RegionBounds] = []
    for lat in lat_vals:
        for lon in lon_vals:
            region = RegionBounds(
                lat_min=float(lat),
                lat_max=float(lat + region_height_deg),
                lon_min=float(lon),
                lon_max=float(lon + region_width_deg),
                depth_min=full_bounds.depth_min,
                depth_max=full_bounds.depth_max,
            )
            mask = (
                (st_lats >= region.lat_min)
                & (st_lats <= region.lat_max)
                & (st_lons >= region.lon_min)
                & (st_lons <= region.lon_max)
            )
            if not mask.any():
                continue

            center_lat, center_lon = region.center
            if has_good_azimuth_coverage(
                station_lats=st_lats[mask],
                station_lons=st_lons[mask],
                center_lat=center_lat,
                center_lon=center_lon,
                azimuth_bin_deg=azimuth_bin_deg,
                min_stations_per_bin=min_stations_per_bin,
                require_all_bins=require_all_bins,
            ):
                candidates.append(region)

    return candidates


def subset_arrivals_to_region(
    arrivals: pd.DataFrame,
    region: RegionBounds,
    teleseismic: bool = False,
) -> pd.DataFrame:
    """Filter arrivals to region bounds.

    teleseismic=False:
      Keep arrivals only when both receiver and source are strictly inside the
      region (boundary points are excluded).

    teleseismic=True:
      Keep arrivals whose receiver is inside the region, regardless of source
      location/depth. This allows teleseismic sources while trimming out-of-
      bounds receivers. To avoid later swap-src-rec boundary failures, arrivals
      with sources exactly on region boundaries are also dropped.
    """
    tol = 1e-9

    def _strict_inside(series: pd.Series, lower: float, upper: float) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        return values.gt(lower + tol) & values.lt(upper - tol)

    def _on_boundary(series: pd.Series, lower: float, upper: float) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        return np.isclose(values, lower, atol=tol, rtol=0.0) | np.isclose(values, upper, atol=tol, rtol=0.0)

    receiver_inside = _strict_inside(arrivals["stla"], region.lat_min, region.lat_max) & _strict_inside(
        arrivals["stlo"], region.lon_min, region.lon_max
    )

    if teleseismic:
        source_on_boundary = (
            _on_boundary(arrivals["evla"], region.lat_min, region.lat_max)
            | _on_boundary(arrivals["evlo"], region.lon_min, region.lon_max)
            | _on_boundary(arrivals["evdp"], region.depth_min, region.depth_max)
        )
        mask = receiver_inside & ~source_on_boundary
    else:
        source_inside = (
            _strict_inside(arrivals["evla"], region.lat_min, region.lat_max)
            & _strict_inside(arrivals["evlo"], region.lon_min, region.lon_max)
            & _strict_inside(arrivals["evdp"], region.depth_min, region.depth_max)
        )
        mask = receiver_inside & source_inside
    return arrivals.loc[mask].copy()


def decluster_events_by_blocks(
    arrivals: pd.DataFrame,
    block_size_km: float,
    center_lat: float,
    center_lon: float,
) -> pd.DataFrame:
    """Keep only the highest-pick event in each 3-D block."""
    if block_size_km <= 0:
        return arrivals

    grouped = arrivals.groupby("event_id", sort=False)
    events = grouped.agg(evla=("evla", "first"), evlo=("evlo", "first"), evdp=("evdp", "first"), n_picks=("phase", "size"))

    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * max(math.cos(math.radians(center_lat)), 1e-6)
    x_km = (events["evlo"] - center_lon) * km_per_deg_lon
    y_km = (events["evla"] - center_lat) * km_per_deg_lat
    z_km = events["evdp"]

    block_ix = np.floor_divide(x_km, block_size_km).astype(int)
    block_iy = np.floor_divide(y_km, block_size_km).astype(int)
    block_iz = np.floor_divide(z_km, block_size_km).astype(int)

    best: Dict[Tuple[int, int, int], Tuple[str, int]] = {}
    for event_id, ix, iy, iz, n_picks in zip(events.index, block_ix, block_iy, block_iz, events["n_picks"]):
        key = (int(ix), int(iy), int(iz))
        if key not in best or int(n_picks) > best[key][1]:
            best[key] = (str(event_id), int(n_picks))

    keep_ids = {event_id for event_id, _ in best.values()}
    return arrivals[arrivals["event_id"].isin(keep_ids)].copy()


def filter_events_by_min_picks(arrivals: pd.DataFrame, phase: str, min_picks: int) -> pd.DataFrame:
    """Keep only events with at least min_picks for the specified phase."""
    if min_picks <= 1:
        return arrivals
    phase_mask = arrivals["phase"].astype(str).str.upper() == phase.upper()
    counts = arrivals.loc[phase_mask].groupby("event_id").size()
    keep_ids = counts[counts >= min_picks].index
    return arrivals[arrivals["event_id"].isin(keep_ids)].copy()


def compute_central_bounds(region: RegionBounds, area_fraction: float) -> RegionBounds:
    """Define the central subregion with a given area fraction."""
    if not (0.0 < area_fraction <= 1.0):
        raise ValueError("area_fraction must be in (0, 1].")
    dlat = region.lat_max - region.lat_min
    dlon = region.lon_max - region.lon_min
    scale = math.sqrt(area_fraction)
    dlat_c = dlat * scale
    dlon_c = dlon * scale
    lat_c, lon_c = region.center
    return RegionBounds(
        lat_min=lat_c - 0.5 * dlat_c,
        lat_max=lat_c + 0.5 * dlat_c,
        lon_min=lon_c - 0.5 * dlon_c,
        lon_max=lon_c + 0.5 * dlon_c,
        depth_min=region.depth_min,
        depth_max=region.depth_max,
    )


def filter_events_to_central_domain(
    arrivals: pd.DataFrame,
    region: RegionBounds,
    area_fraction: float,
) -> Tuple[pd.DataFrame, Optional[RegionBounds]]:
    """Restrict events to the central area fraction of the region."""
    if area_fraction >= 1.0:
        return arrivals, None
    central_bounds = compute_central_bounds(region, area_fraction)
    events = arrivals.groupby("event_id", sort=False).agg(evla=("evla", "first"), evlo=("evlo", "first"))
    keep = events["evla"].between(central_bounds.lat_min, central_bounds.lat_max) & events["evlo"].between(
        central_bounds.lon_min, central_bounds.lon_max
    )
    keep_ids = events.loc[keep].index
    return arrivals[arrivals["event_id"].isin(keep_ids)].copy(), central_bounds


def compute_teleseismic_stats(
    arrivals: pd.DataFrame,
    region: RegionBounds,
    tol: float = 1e-9,
) -> tuple[bool, int, int]:
    """Return whether teleseismic data exist, plus teleseismic event/arrival counts.

    A teleseismic arrival here means its source is not strictly inside region
    bounds in (lat, lon, depth).
    """
    if arrivals.empty:
        return False, 0, 0

    evla = pd.to_numeric(arrivals["evla"], errors="coerce")
    evlo = pd.to_numeric(arrivals["evlo"], errors="coerce")
    evdp = pd.to_numeric(arrivals["evdp"], errors="coerce")
    inside = (
        evla.gt(region.lat_min + tol)
        & evla.lt(region.lat_max - tol)
        & evlo.gt(region.lon_min + tol)
        & evlo.lt(region.lon_max - tol)
        & evdp.gt(region.depth_min + tol)
        & evdp.lt(region.depth_max - tol)
    )
    teleseismic_mask = ~inside.fillna(False)
    if not teleseismic_mask.any():
        return False, 0, 0

    tel_arrivals = int(teleseismic_mask.sum())
    tel_events = int(arrivals.loc[teleseismic_mask, "event_id"].astype(str).nunique())
    return True, tel_events, tel_arrivals


def build_ray_coverage_args(
    args: argparse.Namespace,
    src_rec_path: Path,
    output_dir: Path,
    region_bounds: Optional[RegionBounds],
) -> SimpleNamespace:
    lat_range = None
    lon_range = None
    if region_bounds is not None:
        lat_range = [region_bounds.lat_min, region_bounds.lat_max]
        lon_range = [region_bounds.lon_min, region_bounds.lon_max]

    return SimpleNamespace(
        src_rec=str(src_rec_path),
        dlat=args.rc_dlat,
        dlon=args.rc_dlon,
        ddep=args.rc_ddep,
        lat_range=lat_range,
        lon_range=lon_range,
        depth_range=args.rc_depth_range,
        ray_step_km=args.rc_ray_step_km,
        hit_saturation=args.rc_hit_saturation,
        event_saturation=args.rc_event_saturation,
        station_saturation=args.rc_station_saturation,
        weight_hits=args.rc_weight_hits,
        weight_azimuth=args.rc_weight_azimuth,
        weight_diversity=args.rc_weight_diversity,
        q_min=args.rc_q_min,
        output_dir=str(output_dir),
        hdf5_out=args.rc_hdf5_out,
        summary_csv=args.rc_summary_csv,
        depth_map=args.rc_depth_map,
        rectangle_step=args.rc_rectangle_step,
        rectangle_min_depth=args.rc_rectangle_min_depth,
        skip_rectangles=args.rc_skip_rectangles,
        regional_coverage=args.rc_regional_coverage,
        regional_min_size=args.rc_regional_min_size,
        regional_summary=args.rc_regional_summary,
        regional_map_dir=args.rc_regional_map_dir,
        log_level=args.log_level,
    )


def run_ray_coverage(src_rec_path: Path, output_dir: Path, args: argparse.Namespace, region: RegionBounds) -> None:
    """Execute the ray-coverage workflow on a single src_rec.dat."""
    rc_args = build_ray_coverage_args(args, src_rec_path, output_dir, region)
    events, stations, rays = ray_coverage.parse_src_rec(src_rec_path)
    grid = ray_coverage.build_grid(events, stations, rc_args)
    cells = ray_coverage.accumulate_ray_coverage(events, stations, rays, grid, rc_args.ray_step_km)
    arrays = ray_coverage.compute_cell_arrays(grid, cells, rc_args)
    depth_max = ray_coverage.compute_depth_max(grid, arrays["Q"], rc_args.q_min)

    output_dir = Path(rc_args.output_dir)
    hdf5_path = ray_coverage.resolve_output_path(rc_args.hdf5_out, output_dir)
    summary_path = ray_coverage.resolve_output_path(rc_args.summary_csv, output_dir)
    depth_map_path = ray_coverage.resolve_output_path(rc_args.depth_map, output_dir)

    ray_coverage.write_hdf5(hdf5_path, grid, arrays, depth_max, rc_args)
    ray_coverage.write_summary_csv(summary_path, grid, depth_max)
    ray_coverage.plot_depth_map(depth_map_path, grid, depth_max)

    thresholds: List[float] = []
    if not rc_args.skip_rectangles:
        thresholds = ray_coverage.build_rectangle_thresholds(
            depth_max, rc_args.rectangle_min_depth, rc_args.rectangle_step
        )

    if thresholds:
        rectangles = ray_coverage.find_max_rectangles(grid, depth_max, thresholds)
        rect_dir = output_dir / "rectangles"
        rect_dir.mkdir(parents=True, exist_ok=True)
        for rect in rectangles:
            depth = rect["depth_threshold"]
            rect_path = rect_dir / f"rectangle_depth_{depth:.1f}.png"
            ray_coverage.plot_rectangle(rect_path, grid, depth_max, rect)

    if rc_args.regional_coverage:
        map_dir = ray_coverage.resolve_output_path(rc_args.regional_map_dir, output_dir)
        regional_results = ray_coverage.compute_regional_coverage(
            events,
            stations,
            rays,
            grid,
            rc_args,
            rc_args.regional_min_size,
            map_dir,
        )
        regional_summary_path = ray_coverage.resolve_output_path(rc_args.regional_summary, output_dir)
        ray_coverage.write_regional_summary(regional_summary_path, regional_results)


def write_region_metadata(path: Path, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def write_region_summary_csv(path: Path, summaries: List[Dict[str, object]]) -> None:
    if not summaries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summaries)
    df.to_csv(path, index=False)


def write_region_rank_csv(path: Path, summaries: List[Dict[str, object]]) -> None:
    if not summaries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summaries)
    columns = [
        "region_id",
        "n_stations",
        "n_events",
        "n_arrivals",
        "avg_receiver_spacing_km",
        "n_p_picks",
        "n_s_picks",
        "event_coverage_pct",
        "event_coverage_bin_deg",
        "azimuth_bins_total",
        "azimuth_bins_populated",
        "azimuth_bin_mean",
        "azimuth_bin_std",
        "azimuth_bin_min",
        "azimuth_bin_max",
        "azimuth_bin_deg",
        "src_rec",
    ]
    columns = [col for col in columns if col in df.columns]
    if columns:
        df = df.loc[:, columns]

    sort_order = _resolve_ranking_sort_order(summaries)
    sort_cols = [col for col, _ in sort_order if col in df.columns]
    if sort_cols:
        ascending = [dict(sort_order).get(col, True) for col in sort_cols]
        df = df.sort_values(sort_cols, ascending=ascending, na_position="last", kind="mergesort")

    df.to_csv(path, index=False)


def process_region(
    region_id: int,
    region: RegionBounds,
    arrivals: pd.DataFrame,
    args: argparse.Namespace,
    output_root: Path,
) -> Optional[Dict[str, object]]:
    """Process a single region and write outputs."""
    teleseismic_mode = bool(getattr(args, "teleseismic", False))
    region_arrivals = subset_arrivals_to_region(arrivals, region, teleseismic=teleseismic_mode)
    if region_arrivals.empty:
        LOGGER.info("Region %03d skipped (no arrivals inside region bounds).", region_id)
        return None

    center_lat, center_lon = region.center
    region_arrivals = decluster_events_by_blocks(region_arrivals, args.decluster_block_km, center_lat, center_lon)
    if region_arrivals.empty:
        LOGGER.info("Region %03d skipped after declustering (no arrivals).", region_id)
        return None

    region_arrivals = filter_events_by_min_picks(region_arrivals, "P", args.min_p_picks)
    if region_arrivals.empty:
        LOGGER.info("Region %03d skipped after min-pick filter (no arrivals).", region_id)
        return None

    min_s_picks = getattr(args, "min_s_picks", 0)
    if min_s_picks > 1:
        region_arrivals = filter_events_by_min_picks(region_arrivals, "S", min_s_picks)
        if region_arrivals.empty:
            LOGGER.info("Region %03d skipped after S-pick filter (no arrivals).", region_id)
            return None

    central_bounds = None
    if teleseismic_mode and 0.0 < args.central_area_fraction < 1.0:
        LOGGER.info(
            "Region %03d teleseismic mode enabled; skipping central-area filter.",
            region_id,
        )
    elif 0.0 < args.central_area_fraction < 1.0:
        region_arrivals, central_bounds = filter_events_to_central_domain(
            region_arrivals, region, args.central_area_fraction
        )
        if region_arrivals.empty:
            LOGGER.info("Region %03d skipped after central-domain filter (no arrivals).", region_id)
            return None

    n_events = int(region_arrivals["event_id"].nunique())
    if args.min_events_region > 0 and n_events < args.min_events_region:
        LOGGER.info(
            "Region %03d skipped (events=%d < min_events_region=%d).",
            region_id,
            n_events,
            args.min_events_region,
        )
        return None

    stations_region = (
        region_arrivals.loc[:, ["staname", "stla", "stlo"]]
        .dropna(subset=["stla", "stlo"])
        .drop_duplicates(subset=["staname"])
    )
    if args.require_all_azimuth_bins and not has_good_azimuth_coverage(
        station_lats=stations_region["stla"].to_numpy(dtype=float, copy=False),
        station_lons=stations_region["stlo"].to_numpy(dtype=float, copy=False),
        center_lat=center_lat,
        center_lon=center_lon,
        azimuth_bin_deg=args.azimuth_bin_deg,
        min_stations_per_bin=args.min_stations_per_bin,
        require_all_bins=True,
    ):
        LOGGER.info("Region %03d skipped after final azimuth-bin filter.", region_id)
        return None

    if not args.depth_range:
        depth_max = _compute_region_depth_max(region_arrivals, region.depth_max, 0.1)
        depth_max = max(region.depth_min, depth_max)
        region = RegionBounds(
            lat_min=region.lat_min,
            lat_max=region.lat_max,
            lon_min=region.lon_min,
            lon_max=region.lon_max,
            depth_min=region.depth_min,
            depth_max=depth_max,
        )
        if central_bounds is not None:
            central_bounds = RegionBounds(
                lat_min=central_bounds.lat_min,
                lat_max=central_bounds.lat_max,
                lon_min=central_bounds.lon_min,
                lon_max=central_bounds.lon_max,
                depth_min=region.depth_min,
                depth_max=region.depth_max,
            )

    region_dir = output_root / f"region_{region_id:03d}"
    region_dir.mkdir(parents=True, exist_ok=True)

    src_rec_path = region_dir / "src_rec.dat"
    sr_region = isc_tools.build_src_rec_object(region_arrivals, args, src_rec_path)
    sr_region.write(str(src_rec_path))
    avg_rr_km = _average_receiver_spacing(sr_region)
    avg_rr_out = float(avg_rr_km) if math.isfinite(avg_rr_km) else None
    if avg_rr_out is not None:
        LOGGER.info("Region %03d average receiver spacing: %.1f km", region_id, avg_rr_out)
    else:
        LOGGER.info("Region %03d average receiver spacing: N/A", region_id)

    fig_dir = region_dir / "figures"
    region_box = (region.lon_min, region.lon_max, region.lat_min, region.lat_max)
    isc_tools._generate_coverage_figures(sr_region, fig_dir, region_box=region_box)
    isc_tools._plot_lateral_depth_connections(sr_region, fig_dir / "source_receiver_connections.png")
    isc_tools._plot_longitudinal_depth_connections(sr_region, fig_dir / "source_receiver_connections_lon.png")

    if args.ray_coverage:
        rc_dir = region_dir / args.rc_output_dir
        run_ray_coverage(src_rec_path, rc_dir, args, region)

    n_stations = int(stations_region["staname"].nunique())
    n_arrivals = int(region_arrivals.shape[0])
    n_p = int((region_arrivals["phase"].astype(str).str.upper() == "P").sum())
    n_s = int((region_arrivals["phase"].astype(str).str.upper() == "S").sum())
    event_bin_deg = args.event_coverage_bin_deg or args.region_step_deg
    event_coverage_pct = compute_event_coverage_percent(region_arrivals, region, event_bin_deg)
    az_stats = compute_azimuth_bin_stats(
        stations_region["stla"].to_numpy(dtype=float, copy=False),
        stations_region["stlo"].to_numpy(dtype=float, copy=False),
        center_lat,
        center_lon,
        args.azimuth_bin_deg,
    )
    has_teleseismic_data, n_teleseismic_events, n_teleseismic_arrivals = compute_teleseismic_stats(region_arrivals, region)

    metadata = {
        "region_id": region_id,
        "region_bounds": region.to_dict(),
        "central_bounds": central_bounds.to_dict() if central_bounds else None,
        "min_events_region": args.min_events_region,
        "central_area_fraction": args.central_area_fraction,
        "require_all_azimuth_bins": args.require_all_azimuth_bins,
        "n_events": n_events,
        "n_stations": n_stations,
        "n_arrivals": n_arrivals,
        "n_p_picks": n_p,
        "n_s_picks": n_s,
        "event_coverage_pct": event_coverage_pct,
        "event_coverage_bin_deg": event_bin_deg,
        "azimuth_bin_deg": args.azimuth_bin_deg,
        "avg_receiver_spacing_km": avg_rr_out,
        "teleseismic_mode": teleseismic_mode,
        "has_teleseismic_data": has_teleseismic_data,
        "n_teleseismic_events": n_teleseismic_events,
        "n_teleseismic_arrivals": n_teleseismic_arrivals,
        **az_stats,
        "src_rec": str(src_rec_path),
    }
    write_region_metadata(region_dir / "region_metadata.json", metadata)
    LOGGER.info(
        "Region %03d done: %d events, %d stations, %d arrivals",
        region_id,
        n_events,
        n_stations,
        n_arrivals,
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", default="isc_arrivals.csv", help="Input ISC arrivals CSV file.")
    parser.add_argument("--csv-dir", default=".", help="Directory to search for CSV files.")
    parser.add_argument("--choose-csv", action="store_true", help="Interactively choose the CSV file.")
    parser.add_argument("--src-rec", help="Read an existing src_rec.dat instead of an ISC CSV.")
    parser.add_argument("--phases", default="P", help="Comma-separated list of phases to keep.")
    parser.add_argument("--include-s", action="store_true", help="Include S picks alongside P picks.")
    parser.add_argument(
        "--phases-anycase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize phases to uppercase (default true).",
    )
    parser.add_argument("--min-sta", type=int, default=1, help="Minimum number of stations per event (global filter).")
    parser.add_argument("--dist-min", type=float, default=0.0, help="Minimum epicentral distance (degrees).")
    parser.add_argument("--dist-max", type=float, default=30.0, help="Maximum epicentral distance (degrees).")
    parser.add_argument("--use-dist-col", action="store_true", help="Use DIST column as authoritative distance.")
    parser.add_argument(
        "--name-netsta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NET.STA station naming (default true).",
    )
    parser.add_argument("--min-tt-sec", type=float, default=0.0, help="Minimum travel time (seconds).")
    parser.add_argument("--max-tt-sec", type=float, default=9000.0, help="Maximum travel time (seconds).")
    parser.add_argument(
        "--lat-range",
        type=float,
        nargs=2,
        metavar=("MIN_LAT", "MAX_LAT"),
        help="Lat bounds for filtering and region scanning (degrees).",
    )
    parser.add_argument(
        "--lon-range",
        type=float,
        nargs=2,
        metavar=("MIN_LON", "MAX_LON"),
        help="Lon bounds for filtering and region scanning (degrees).",
    )
    parser.add_argument(
        "--depth-range",
        type=float,
        nargs=2,
        metavar=("MIN_DEP", "MAX_DEP"),
        help="Event depth bounds for filtering (km).",
    )
    parser.add_argument(
        "--scan-regions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable sliding-window region scan (default true).",
    )
    parser.add_argument("--region-width-deg", type=float, default=1.0, help="Region width in degrees.")
    parser.add_argument("--region-height-deg", type=float, default=1.0, help="Region height in degrees.")
    parser.add_argument("--region-step-deg", type=float, default=0.5, help="Region scan step in degrees.")
    parser.add_argument("--azimuth-bin-deg", type=float, default=10.0, help="Azimuth bin size (degrees).")
    parser.add_argument(
        "--event-coverage-bin-deg",
        type=float,
        help="Bin size (degrees) for event coverage percentage (defaults to region-step-deg).",
    )
    parser.add_argument("--min-stations-per-bin", type=int, default=1, help="Min stations per azimuth bin.")
    parser.add_argument(
        "--require-all-azimuth-bins",
        action="store_true",
        help="Require each azimuth bin to meet the minimum station count.",
    )
    parser.add_argument("--max-regions", type=int, default=0, help="Maximum number of regions to process (0=all).")
    parser.add_argument(
        "--stop-after-first",
        action="store_true",
        help="Stop after the first successfully processed region.",
    )
    parser.add_argument(
        "--decluster-block-km",
        type=float,
        default=0.4,
        help="Block size for declustering (km); set <=0 to disable.",
    )
    parser.add_argument("--min-p-picks", type=int, default=6, help="Minimum number of P picks per event.")
    parser.add_argument("--min-s-picks", type=int, default=0, help="Minimum number of S picks per event.")
    parser.add_argument(
        "--min-events-region",
        type=int,
        default=0,
        help="Minimum number of events required per region (0 disables).",
    )
    parser.add_argument(
        "--central-area-fraction",
        type=float,
        default=0.05,
        help="Central area fraction for events (0<frac<=1).",
    )
    parser.add_argument(
        "--teleseismic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep out-of-region sources/events while removing out-of-region "
            "receivers. Useful for teleseismic-style inputs."
        ),
    )
    parser.add_argument("--output-root", default="tomo_regions", help="Root directory for region outputs.")
    parser.add_argument(
        "--write-global-src-rec",
        action="store_true",
        help="Write the globally filtered src_rec.dat to --out-dat.",
    )
    parser.add_argument("--out-dat", default="src_rec.dat", help="Output src_rec filename for global write.")
    parser.add_argument(
        "--ray-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ray coverage for each region (default true).",
    )
    parser.add_argument("--rc-output-dir", default="ray_coverage", help="Ray coverage output subdirectory.")
    parser.add_argument("--rc-dlat", type=float, default=0.1, help="Ray coverage latitude spacing (deg).")
    parser.add_argument("--rc-dlon", type=float, default=0.1, help="Ray coverage longitude spacing (deg).")
    parser.add_argument("--rc-ddep", type=float, default=10.0, help="Ray coverage depth spacing (km).")
    parser.add_argument(
        "--rc-depth-range",
        type=float,
        nargs=2,
        metavar=("MIN_DEPTH", "MAX_DEPTH"),
        help="Optional depth range override for ray coverage (km).",
    )
    parser.add_argument("--rc-ray-step-km", type=float, default=10.0, help="Ray coverage sampling step (km).")
    parser.add_argument("--rc-hit-saturation", type=float, default=100.0, help="Hit-count saturation threshold.")
    parser.add_argument("--rc-event-saturation", type=float, default=10.0, help="Event-count saturation threshold.")
    parser.add_argument("--rc-station-saturation", type=float, default=10.0, help="Station-count saturation threshold.")
    parser.add_argument("--rc-weight-hits", type=float, default=0.4, help="Weight for hit density.")
    parser.add_argument("--rc-weight-azimuth", type=float, default=0.4, help="Weight for azimuth coverage.")
    parser.add_argument("--rc-weight-diversity", type=float, default=0.2, help="Weight for diversity metric.")
    parser.add_argument("--rc-q-min", type=float, default=0.5, help="Q threshold for coverage depth map.")
    parser.add_argument("--rc-hdf5-out", default="ray_coverage_metrics.h5", help="Ray coverage HDF5 output name.")
    parser.add_argument("--rc-summary-csv", default="ray_coverage_summary.csv", help="Ray coverage CSV output name.")
    parser.add_argument("--rc-depth-map", default="ray_coverage_depth.png", help="Ray coverage depth map output name.")
    parser.add_argument(
        "--rc-rectangle-step",
        type=float,
        default=5.0,
        help="Depth decrement (km) for rectangle search.",
    )
    parser.add_argument(
        "--rc-rectangle-min-depth",
        type=float,
        default=10.0,
        help="Minimum depth (km) for rectangle search.",
    )
    parser.add_argument("--rc-skip-rectangles", action="store_true", help="Disable rectangle search.")
    parser.add_argument(
        "--rc-regional-coverage",
        action="store_true",
        help="Enable regional-coverage analysis (endpoint filtered).",
    )
    parser.add_argument(
        "--rc-regional-min-size",
        type=float,
        default=0.1,
        help="Minimum rectangle size for regional coverage (degrees).",
    )
    parser.add_argument(
        "--rc-regional-summary",
        default="regional_coverage_regions.csv",
        help="Regional coverage CSV output name.",
    )
    parser.add_argument(
        "--rc-regional-map-dir",
        default="regional_maps",
        help="Regional coverage map output directory (inside ray coverage dir).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _require_pytomoatt()

    args.phases = _normalize_phase_list(args.phases, args.include_s)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.src_rec:
        src_rec_path = Path(args.src_rec).expanduser()
        if not src_rec_path.exists():
            raise FileNotFoundError(f"src_rec file not found: {src_rec_path}")
        sr_all = SrcRec.read(str(src_rec_path))
        if args.phases_anycase and hasattr(sr_all, "rec_points") and "phase" in sr_all.rec_points.columns:
            sr_all.rec_points["phase"] = sr_all.rec_points["phase"].astype(str).str.upper()
        if not isc_tools.apply_standard_filters(sr_all, args, "Global"):
            raise SystemExit("No arrivals remain after global filters.")
        if args.write_global_src_rec:
            sr_all.write(str(Path(args.out_dat)))
            LOGGER.info("Wrote global src_rec: %s", args.out_dat)
        arrivals = build_arrivals_from_src_rec(sr_all)
    else:
        csv_path = isc_tools.resolve_csv_path(args)
        args.csv = str(csv_path)
        raw_df = isc_tools.read_isc_csv(args.csv)
        mapped = isc_tools.extract_columns(raw_df)
        src_rec_df = isc_tools.build_src_rec_df(mapped, args)
        if src_rec_df.empty:
            raise SystemExit("No valid arrivals after preprocessing.")

        sr_all = isc_tools.build_src_rec_object(src_rec_df, args, args.out_dat)
        if not isc_tools.apply_standard_filters(sr_all, args, "Global"):
            raise SystemExit("No arrivals remain after global filters.")

        if args.write_global_src_rec:
            sr_all.write(str(Path(args.out_dat)))
            LOGGER.info("Wrote global src_rec: %s", args.out_dat)

        arrivals = build_arrivals_from_src_rec(sr_all)
    if not args.include_s:
        arrivals = arrivals[arrivals["phase"].astype(str).str.upper() == "P"].copy()

    full_bounds = compute_full_bounds(arrivals, args)
    stations = (
        arrivals.loc[:, ["staname", "stla", "stlo", "stel"]]
        .dropna(subset=["stla", "stlo"])
        .drop_duplicates(subset=["staname"])
        .reset_index(drop=True)
    )

    candidates: List[RegionBounds]
    if args.scan_regions:
        candidates = find_candidate_regions(
            stations=stations,
            full_bounds=full_bounds,
            region_width_deg=args.region_width_deg,
            region_height_deg=args.region_height_deg,
            region_step_deg=args.region_step_deg,
            azimuth_bin_deg=args.azimuth_bin_deg,
            min_stations_per_bin=args.min_stations_per_bin,
            require_all_bins=args.require_all_azimuth_bins,
        )
        if not candidates:
            raise SystemExit("No candidate regions found with the requested azimuth coverage.")
    else:
        candidates = [full_bounds]

    if args.max_regions > 0:
        candidates = candidates[: args.max_regions]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    central_fractions = _build_central_fraction_sequence(args.central_area_fraction)
    for attempt_idx, fraction in enumerate(central_fractions, start=1):
        if attempt_idx > 1:
            LOGGER.info(
                "No regions found; retrying with central-area fraction %.0f%%.",
                fraction * 100.0,
            )
        args.central_area_fraction = fraction
        summaries = []
        for idx, region in enumerate(candidates, start=1):
            summary = process_region(idx, region, arrivals, args, output_root)
            if summary:
                summaries.append(summary)
                if args.stop_after_first:
                    break
        if summaries:
            break

    write_region_summary_csv(output_root / "region_summary.csv", summaries)
    write_region_rank_csv(output_root / "region_rankings.csv", summaries)
    if not summaries:
        LOGGER.warning(
            "No regions found for central-area fractions up to %.0f%%.",
            central_fractions[-1] * 100.0,
        )
    LOGGER.info("Done. Processed %d region(s).", len(summaries))


if __name__ == "__main__":
    main()
