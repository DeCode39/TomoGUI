#!/usr/bin/env python3
"""Build raw 3D ray archives and node coverage summaries for TomoATT outputs."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


EARTH_RADIUS_KM = 6371.0
DEFAULT_STEP_KM = 0.75
DEFAULT_MAX_STEPS = 2000
DEFAULT_PHASE = "P"
DEFAULT_ITERATION_TAG = "0039"
NAZ_BINS = 12
NDIP_BINS = 6
AZ_BIN_WIDTH = 360.0 / NAZ_BINS
DIP_BIN_WIDTH = 90.0 / NDIP_BINS


@dataclass(frozen=True)
class Station:
    station_id: int
    name: str
    lon: float
    lat: float
    dep: float


@dataclass(frozen=True)
class Event:
    event_id: int
    lon: float
    lat: float
    dep: float
    npicks: int


@dataclass(frozen=True)
class RayPick:
    ray_id: int
    event_id: int
    station_id: int
    station_name: str
    phase: str


@dataclass(frozen=True)
class GridSpec:
    lon: np.ndarray
    lat: np.ndarray
    dep: np.ndarray
    node_id_grid: np.ndarray
    lon_edges: np.ndarray
    lat_edges: np.ndarray
    dep_edges: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to inversion_P or inversion_S folder.")
    parser.add_argument("--step", default="joint_step2", help="OUTPUT_FILES step name, e.g. joint_step2.")
    parser.add_argument("--phase", default=DEFAULT_PHASE, help="Phase to trace from src_rec file.")
    parser.add_argument("--iteration-tag", default=DEFAULT_ITERATION_TAG, help="Iteration tag for time fields.")
    parser.add_argument("--step-km", type=float, default=DEFAULT_STEP_KM, help="Tracing step in km.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum tracing steps per ray.")
    parser.add_argument("--limit-rays", type=int, default=0, help="Optional cap on number of rays for testing.")
    return parser.parse_args()


def _parse_src_rec(path: Path, phase: str) -> tuple[list[Station], list[Event], list[RayPick]]:
    stations_by_name: dict[str, Station] = {}
    events_by_id: dict[int, Event] = {}
    rays: list[RayPick] = []
    current_event: Event | None = None
    seen = set()

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.split()
            if len(parts) >= 13:
                try:
                    event_id = int(parts[0])
                    current_event = Event(
                        event_id=event_id,
                        lon=float(parts[8]),
                        lat=float(parts[7]),
                        dep=float(parts[9]),
                        npicks=int(float(parts[11])),
                    )
                except ValueError:
                    current_event = None
                    continue
                events_by_id[event_id] = current_event
                continue
            if len(parts) < 8 or current_event is None:
                continue
            if parts[6] != phase:
                continue
            station_name = parts[2]
            if station_name not in stations_by_name:
                try:
                    station = Station(
                        station_id=len(stations_by_name),
                        name=station_name,
                        lon=float(parts[4]),
                        lat=float(parts[3]),
                        dep=-float(parts[5]) / 1000.0,
                    )
                except ValueError:
                    continue
                stations_by_name[station_name] = station
            station = stations_by_name[station_name]
            key = (current_event.event_id, station.station_id, phase)
            if key in seen:
                continue
            seen.add(key)
            rays.append(
                RayPick(
                    ray_id=len(rays),
                    event_id=current_event.event_id,
                    station_id=station.station_id,
                    station_name=station.name,
                    phase=phase,
                )
            )

    stations = sorted(stations_by_name.values(), key=lambda item: item.station_id)
    events = sorted(events_by_id.values(), key=lambda item: item.event_id)
    return stations, events, rays


def _axis_edges(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.size == 1:
        return np.array([coords[0] - 0.5, coords[0] + 0.5], dtype=float)
    mids = 0.5 * (coords[:-1] + coords[1:])
    start = coords[0] - (mids[0] - coords[0])
    end = coords[-1] + (coords[-1] - mids[-1])
    return np.concatenate([[start], mids, [end]])


def _load_grid(step_dir: Path) -> GridSpec:
    grid_path = step_dir / "out_data_grid.h5"
    with h5py.File(grid_path, "r") as handle:
        lon_nodes = np.asarray(handle["Mesh/node_coords_p"][...], dtype=float)
        lat_nodes = np.asarray(handle["Mesh/node_coords_t"][...], dtype=float)
        dep_nodes = EARTH_RADIUS_KM - np.asarray(handle["Mesh/node_coords_r"][...], dtype=float)

    lon = np.unique(lon_nodes)
    lat = np.unique(lat_nodes)
    dep = np.unique(dep_nodes)
    lon_idx = np.searchsorted(lon, lon_nodes)
    lat_idx = np.searchsorted(lat, lat_nodes)
    dep_idx = np.searchsorted(dep, dep_nodes)

    node_id_grid = np.empty((dep.size, lat.size, lon.size), dtype=np.int32)
    node_id_grid[dep_idx, lat_idx, lon_idx] = np.arange(lon_nodes.size, dtype=np.int32)

    return GridSpec(
        lon=lon,
        lat=lat,
        dep=dep,
        node_id_grid=node_id_grid,
        lon_edges=_axis_edges(lon),
        lat_edges=_axis_edges(lat),
        dep_edges=_axis_edges(dep),
    )


def _find_station_source_files(step_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for sim_path in sorted(step_dir.glob("out_data_sim_group_*.h5")):
        with h5py.File(sim_path, "r") as handle:
            for key in handle.keys():
                if key == "model" or not key.startswith("src_"):
                    continue
                mapping[key.removeprefix("src_")] = sim_path
    return mapping


def _grid_from_flat(values: np.ndarray, node_id_grid: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[node_id_grid]


def _interp_regular_3d(
    dep_coords: np.ndarray,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    values: np.ndarray,
    dep: float,
    lat: float,
    lon: float,
) -> float:
    if dep < dep_coords[0] or dep > dep_coords[-1]:
        return float("nan")
    if lat < lat_coords[0] or lat > lat_coords[-1]:
        return float("nan")
    if lon < lon_coords[0] or lon > lon_coords[-1]:
        return float("nan")

    def _bounds(coords: np.ndarray, value: float) -> tuple[int, int, float]:
        hi = int(np.searchsorted(coords, value, side="right"))
        if hi <= 0:
            return 0, 0, 0.0
        if hi >= coords.size:
            idx = coords.size - 1
            return idx, idx, 0.0
        lo = hi - 1
        frac = (value - coords[lo]) / (coords[hi] - coords[lo])
        return lo, hi, float(frac)

    d0, d1, fd = _bounds(dep_coords, dep)
    a0, a1, fa = _bounds(lat_coords, lat)
    o0, o1, fo = _bounds(lon_coords, lon)

    c000 = values[d0, a0, o0]
    c001 = values[d0, a0, o1]
    c010 = values[d0, a1, o0]
    c011 = values[d0, a1, o1]
    c100 = values[d1, a0, o0]
    c101 = values[d1, a0, o1]
    c110 = values[d1, a1, o0]
    c111 = values[d1, a1, o1]

    if not np.all(np.isfinite([c000, c001, c010, c011, c100, c101, c110, c111])):
        return float("nan")

    c00 = c000 * (1.0 - fo) + c001 * fo
    c01 = c010 * (1.0 - fo) + c011 * fo
    c10 = c100 * (1.0 - fo) + c101 * fo
    c11 = c110 * (1.0 - fo) + c111 * fo
    c0 = c00 * (1.0 - fa) + c01 * fa
    c1 = c10 * (1.0 - fa) + c11 * fa
    return float(c0 * (1.0 - fd) + c1 * fd)


def _interp_regular_3d_multi(
    dep_coords: np.ndarray,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    values: np.ndarray,
    dep: float,
    lat: float,
    lon: float,
) -> np.ndarray:
    if dep < dep_coords[0] or dep > dep_coords[-1]:
        return np.full(values.shape[0], np.nan, dtype=np.float64)
    if lat < lat_coords[0] or lat > lat_coords[-1]:
        return np.full(values.shape[0], np.nan, dtype=np.float64)
    if lon < lon_coords[0] or lon > lon_coords[-1]:
        return np.full(values.shape[0], np.nan, dtype=np.float64)

    def _bounds(coords: np.ndarray, value: float) -> tuple[int, int, float]:
        hi = int(np.searchsorted(coords, value, side="right"))
        if hi <= 0:
            return 0, 0, 0.0
        if hi >= coords.size:
            idx = coords.size - 1
            return idx, idx, 0.0
        lo = hi - 1
        frac = (value - coords[lo]) / (coords[hi] - coords[lo])
        return lo, hi, float(frac)

    d0, d1, fd = _bounds(dep_coords, dep)
    a0, a1, fa = _bounds(lat_coords, lat)
    o0, o1, fo = _bounds(lon_coords, lon)

    w000 = (1.0 - fd) * (1.0 - fa) * (1.0 - fo)
    w001 = (1.0 - fd) * (1.0 - fa) * fo
    w010 = (1.0 - fd) * fa * (1.0 - fo)
    w011 = (1.0 - fd) * fa * fo
    w100 = fd * (1.0 - fa) * (1.0 - fo)
    w101 = fd * (1.0 - fa) * fo
    w110 = fd * fa * (1.0 - fo)
    w111 = fd * fa * fo

    result = (
        values[:, d0, a0, o0] * w000
        + values[:, d0, a0, o1] * w001
        + values[:, d0, a1, o0] * w010
        + values[:, d0, a1, o1] * w011
        + values[:, d1, a0, o0] * w100
        + values[:, d1, a0, o1] * w101
        + values[:, d1, a1, o0] * w110
        + values[:, d1, a1, o1] * w111
    )
    if not np.all(np.isfinite(result)):
        return np.full(values.shape[0], np.nan, dtype=np.float64)
    return result


def _approx_distance_km(
    lon1: float,
    lat1: float,
    dep1: float,
    lon2: float,
    lat2: float,
    dep2: float,
) -> float:
    mean_lat = math.radians((lat1 + lat2) * 0.5)
    dx = (lon2 - lon1) * 111.32 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 111.32
    dz = dep2 - dep1
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def _direction_from_segment(
    lon0: float,
    lat0: float,
    dep0: float,
    lon1: float,
    lat1: float,
    dep1: float,
) -> tuple[float, float] | None:
    mean_lat = math.radians((lat0 + lat1) * 0.5)
    east = (lon1 - lon0) * 111.32 * math.cos(mean_lat)
    north = (lat1 - lat0) * 111.32
    vertical = dep1 - dep0
    horiz = math.hypot(east, north)
    norm = math.sqrt(horiz * horiz + vertical * vertical)
    if norm <= 1.0e-12:
        return None
    azimuth = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
    dip = math.degrees(math.atan2(abs(vertical), max(horiz, 1.0e-12)))
    return azimuth, dip


def _coord_to_index(edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, values, side="right") - 1
    return idx.astype(np.int32, copy=False)


def _trace_ray(
    event: Event,
    station: Station,
    dep_coords: np.ndarray,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    interp_bundle: np.ndarray,
    step_km: float,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    dep = float(event.dep)
    lat = float(event.lat)
    lon = float(event.lon)

    lons = [lon]
    lats = [lat]
    deps = [dep]
    ok = False

    for _ in range(max_steps):
        if _approx_distance_km(lon, lat, dep, station.lon, station.lat, station.dep) < max(step_km, 1.0):
            ok = True
            break

        current = _interp_regular_3d_multi(dep_coords, lat_coords, lon_coords, interp_bundle, dep, lat, lon)
        g_dep, g_lat, g_lon = current[:3]
        if not np.all(np.isfinite([g_dep, g_lat, g_lon])):
            break

        km_per_deg_lat = 111.32
        km_per_deg_lon = max(111.32 * math.cos(math.radians(lat)), 1.0e-6)
        vx = g_lon / km_per_deg_lon
        vy = g_lat / km_per_deg_lat
        vz = g_dep
        norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        if norm <= 1.0e-10:
            break

        dep -= step_km * vz / norm
        lat -= (step_km * vy / norm) / km_per_deg_lat
        lon -= (step_km * vx / norm) / km_per_deg_lon

        lon = float(np.clip(lon, lon_coords[0], lon_coords[-1]))
        lat = float(np.clip(lat, lat_coords[0], lat_coords[-1]))
        dep = float(np.clip(dep, dep_coords[0], dep_coords[-1]))

        lons.append(lon)
        lats.append(lat)
        deps.append(dep)

        t_here = _interp_regular_3d_multi(dep_coords, lat_coords, lon_coords, interp_bundle, dep, lat, lon)[3]
        if np.isfinite(t_here) and t_here < 0.25:
            ok = True
            break

    if lons[-1] != station.lon or lats[-1] != station.lat or deps[-1] != station.dep:
        lons.append(station.lon)
        lats.append(station.lat)
        deps.append(station.dep)
    return np.asarray(lons, dtype=np.float32), np.asarray(lats, dtype=np.float32), np.asarray(deps, dtype=np.float32), ok


def _path_length_km(lon: np.ndarray, lat: np.ndarray, dep: np.ndarray) -> float:
    if lon.size < 2:
        return 0.0
    lon0 = lon[:-1].astype(np.float64, copy=False)
    lon1 = lon[1:].astype(np.float64, copy=False)
    lat0 = lat[:-1].astype(np.float64, copy=False)
    lat1 = lat[1:].astype(np.float64, copy=False)
    dep0 = dep[:-1].astype(np.float64, copy=False)
    dep1 = dep[1:].astype(np.float64, copy=False)
    mean_lat = np.radians((lat0 + lat1) * 0.5)
    dx = (lon1 - lon0) * 111.32 * np.cos(mean_lat)
    dy = (lat1 - lat0) * 111.32
    dz = dep1 - dep0
    return float(np.sqrt(dx * dx + dy * dy + dz * dz).sum())


def _path_node_hits(
    lon: np.ndarray,
    lat: np.ndarray,
    dep: np.ndarray,
    grid: GridSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_i = _coord_to_index(grid.lon_edges, lon)
    lat_i = _coord_to_index(grid.lat_edges, lat)
    dep_i = _coord_to_index(grid.dep_edges, dep)
    mask = (
        (lon_i >= 0)
        & (lon_i < grid.lon.size)
        & (lat_i >= 0)
        & (lat_i < grid.lat.size)
        & (dep_i >= 0)
        & (dep_i < grid.dep.size)
    )
    if not np.any(mask):
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty

    lon_i = lon_i[mask]
    lat_i = lat_i[mask]
    dep_i = dep_i[mask]
    point_idx = np.flatnonzero(mask)
    node_ids = grid.node_id_grid[dep_i, lat_i, lon_i]

    keep = np.ones(node_ids.size, dtype=bool)
    keep[1:] = node_ids[1:] != node_ids[:-1]
    node_ids = node_ids[keep]
    point_idx = point_idx[keep]
    if node_ids.size == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty

    _, first_idx = np.unique(node_ids, return_index=True)
    first_idx.sort()
    return (
        node_ids[first_idx].astype(np.int32, copy=False),
        point_idx[first_idx].astype(np.int32, copy=False),
        np.asarray(point_idx, dtype=np.int32),
    )


def _append_1d_dataset(handle: h5py.File, path: str, values: np.ndarray) -> None:
    dataset = handle[path]
    start = dataset.shape[0]
    stop = start + values.shape[0]
    dataset.resize((stop,))
    dataset[start:stop] = values


def _create_archive_file(
    path: Path,
    stations: list[Station],
    events: list[Event],
    phase: str,
    step: str,
    iteration_tag: str,
    step_dir: Path,
    src_rec_path: Path,
    trace_method: str,
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "w")
    handle.attrs["schema_version"] = "1.0"
    handle.attrs["phase"] = phase
    handle.attrs["step"] = step
    handle.attrs["iteration_tag"] = iteration_tag
    handle.attrs["swap_src_rec"] = True
    handle.attrs["trace_method"] = trace_method
    handle.attrs["coord_system"] = "lon_lat_dep_km"
    handle.attrs["source_step_dir"] = str(step_dir)
    handle.attrs["src_rec_file"] = str(src_rec_path)

    station_group = handle.create_group("stations")
    station_group.create_dataset("station_id", data=np.asarray([s.station_id for s in stations], dtype=np.int32))
    station_group.create_dataset("name", data=np.asarray([s.name.encode("utf-8") for s in stations], dtype=h5py.string_dtype("utf-8")))
    station_group.create_dataset("lon", data=np.asarray([s.lon for s in stations], dtype=np.float32))
    station_group.create_dataset("lat", data=np.asarray([s.lat for s in stations], dtype=np.float32))
    station_group.create_dataset("dep", data=np.asarray([s.dep for s in stations], dtype=np.float32))

    event_group = handle.create_group("events")
    event_group.create_dataset("event_id", data=np.asarray([e.event_id for e in events], dtype=np.int32))
    event_group.create_dataset("lon", data=np.asarray([e.lon for e in events], dtype=np.float32))
    event_group.create_dataset("lat", data=np.asarray([e.lat for e in events], dtype=np.float32))
    event_group.create_dataset("dep", data=np.asarray([e.dep for e in events], dtype=np.float32))
    event_group.create_dataset("npicks", data=np.asarray([e.npicks for e in events], dtype=np.int32))

    rays = handle.create_group("rays")
    for name, dtype in (
        ("ray_id", np.int64),
        ("event_id", np.int32),
        ("station_id", np.int32),
        ("src_lon", np.float32),
        ("src_lat", np.float32),
        ("src_dep", np.float32),
        ("rec_lon", np.float32),
        ("rec_lat", np.float32),
        ("rec_dep", np.float32),
        ("traced_ok", np.uint8),
        ("n_points", np.int32),
        ("path_length_km", np.float32),
        ("point_offset", np.int64),
        ("node_hit_offset", np.int64),
        ("lon", np.float32),
        ("lat", np.float32),
        ("dep", np.float32),
        ("node_hit_ids", np.int32),
    ):
        shape = (1,) if name in {"point_offset", "node_hit_offset"} else (0,)
        maxshape = (None,)
        chunks = True
        rays.create_dataset(name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=dtype, compression="gzip")
    rays["point_offset"][0] = 0
    rays["node_hit_offset"][0] = 0
    return handle


def _write_node_summary(
    path: Path,
    grid: GridSpec,
    hit_count: np.ndarray,
    unique_station_count: np.ndarray,
    unique_event_count: np.ndarray,
    az_bins_2d: np.ndarray,
    az_dip_bins_3d: np.ndarray,
    archive_path: Path,
    phase: str,
    step: str,
    iteration_tag: str,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "1.0"
        handle.attrs["phase"] = phase
        handle.attrs["step"] = step
        handle.attrs["iteration_tag"] = iteration_tag
        handle.attrs["source_ray_archive"] = str(archive_path)
        handle.attrs["naz_bins"] = NAZ_BINS
        handle.attrs["ndip_bins"] = NDIP_BINS
        handle.attrs["az_bin_width_deg"] = AZ_BIN_WIDTH
        handle.attrs["dip_bin_width_deg"] = DIP_BIN_WIDTH

        node_group = handle.create_group("nodes")
        dep_grid, lat_grid, lon_grid = np.meshgrid(grid.dep, grid.lat, grid.lon, indexing="ij")
        node_ids = grid.node_id_grid.ravel().astype(np.int32)

        lon_by_id = np.empty(node_ids.size, dtype=np.float32)
        lat_by_id = np.empty(node_ids.size, dtype=np.float32)
        dep_by_id = np.empty(node_ids.size, dtype=np.float32)
        lon_by_id[node_ids] = lon_grid.ravel().astype(np.float32)
        lat_by_id[node_ids] = lat_grid.ravel().astype(np.float32)
        dep_by_id[node_ids] = dep_grid.ravel().astype(np.float32)

        lon_lo = np.repeat(grid.lon_edges[:-1][None, None, :], grid.dep.size, axis=0)
        lon_lo = np.repeat(lon_lo, grid.lat.size, axis=1)
        lon_hi = np.repeat(grid.lon_edges[1:][None, None, :], grid.dep.size, axis=0)
        lon_hi = np.repeat(lon_hi, grid.lat.size, axis=1)
        lat_lo = np.repeat(grid.lat_edges[:-1][None, :, None], grid.dep.size, axis=0)
        lat_lo = np.repeat(lat_lo, grid.lon.size, axis=2)
        lat_hi = np.repeat(grid.lat_edges[1:][None, :, None], grid.dep.size, axis=0)
        lat_hi = np.repeat(lat_hi, grid.lon.size, axis=2)
        dep_lo = np.repeat(grid.dep_edges[:-1][:, None, None], grid.lat.size, axis=1)
        dep_lo = np.repeat(dep_lo, grid.lon.size, axis=2)
        dep_hi = np.repeat(grid.dep_edges[1:][:, None, None], grid.lat.size, axis=1)
        dep_hi = np.repeat(dep_hi, grid.lon.size, axis=2)
        cell_lon_min = np.empty(node_ids.size, dtype=np.float32)
        cell_lon_max = np.empty(node_ids.size, dtype=np.float32)
        cell_lat_min = np.empty(node_ids.size, dtype=np.float32)
        cell_lat_max = np.empty(node_ids.size, dtype=np.float32)
        cell_dep_min = np.empty(node_ids.size, dtype=np.float32)
        cell_dep_max = np.empty(node_ids.size, dtype=np.float32)
        cell_lon_min[node_ids] = lon_lo.ravel().astype(np.float32)
        cell_lon_max[node_ids] = lon_hi.ravel().astype(np.float32)
        cell_lat_min[node_ids] = lat_lo.ravel().astype(np.float32)
        cell_lat_max[node_ids] = lat_hi.ravel().astype(np.float32)
        cell_dep_min[node_ids] = dep_lo.ravel().astype(np.float32)
        cell_dep_max[node_ids] = dep_hi.ravel().astype(np.float32)

        node_group.create_dataset("node_id", data=np.arange(node_ids.size, dtype=np.int32), compression="gzip")
        node_group.create_dataset("lon", data=lon_by_id, compression="gzip")
        node_group.create_dataset("lat", data=lat_by_id, compression="gzip")
        node_group.create_dataset("dep", data=dep_by_id, compression="gzip")
        node_group.create_dataset("cell_lon_min", data=cell_lon_min, compression="gzip")
        node_group.create_dataset("cell_lon_max", data=cell_lon_max, compression="gzip")
        node_group.create_dataset("cell_lat_min", data=cell_lat_min, compression="gzip")
        node_group.create_dataset("cell_lat_max", data=cell_lat_max, compression="gzip")
        node_group.create_dataset("cell_dep_min", data=cell_dep_min, compression="gzip")
        node_group.create_dataset("cell_dep_max", data=cell_dep_max, compression="gzip")

        node_group.create_dataset("hit_count", data=hit_count.astype(np.int32), compression="gzip")
        node_group.create_dataset("unique_station_count", data=unique_station_count.astype(np.int16), compression="gzip")
        node_group.create_dataset("unique_event_count", data=unique_event_count.astype(np.int32), compression="gzip")
        node_group.create_dataset("azimuth_bin_count_2d", data=az_bins_2d.sum(axis=1).astype(np.int16), compression="gzip")
        node_group.create_dataset(
            "azimuth_dip_bin_count_3d",
            data=az_dip_bins_3d.reshape(az_dip_bins_3d.shape[0], -1).sum(axis=1).astype(np.int16),
            compression="gzip",
        )
        node_group.create_dataset("azimuth_bins_2d", data=az_bins_2d.astype(np.uint8), compression="gzip")
        node_group.create_dataset("azimuth_dip_bins_3d", data=az_dip_bins_3d.astype(np.uint8), compression="gzip")


def _station_time_bundle(sim_path: Path, station_name: str, iteration_tag: str, grid: GridSpec) -> np.ndarray:
    with h5py.File(sim_path, "r") as handle:
        group = handle[f"src_{station_name}"]
        field = group[f"time_field_inv_{iteration_tag}"][...]
    time_grid = _grid_from_flat(field, grid.node_id_grid)
    grad_dep, grad_lat, grad_lon = np.gradient(time_grid, grid.dep, grid.lat, grid.lon, edge_order=1)
    return np.stack((grad_dep, grad_lat, grad_lon, time_grid), axis=0)


def _timing_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    step_dir = run_dir / "OUTPUT_FILES" / f"OUTPUT_FILES_{args.step}"
    src_rec_path = run_dir / "1_src_rec_files" / "src_rec_config.dat"
    if not step_dir.exists():
        print(f"Skipping azimuth archive build: step directory not found: {step_dir}")
        return
    if not src_rec_path.exists():
        print(f"Skipping azimuth archive build: src_rec file not found: {src_rec_path}")
        return
    if not (step_dir / "out_data_grid.h5").exists():
        print(f"Skipping azimuth archive build: out_data_grid.h5 not found in {step_dir}")
        return

    output_dir = run_dir / "azcov_archive"
    archive_path = output_dir / f"azcov_{args.phase}_{args.step}_ray_archive.h5"
    summary_path = output_dir / f"azcov_{args.phase}_{args.step}_node_coverage.h5"
    timing_path = output_dir / f"azcov_{args.phase}_{args.step}_timing.json"

    t0 = time.perf_counter()
    stations, events, rays_all = _parse_src_rec(src_rec_path, args.phase)
    if not rays_all:
        print(f"Skipping azimuth archive build: no {args.phase}-phase picks found in {src_rec_path}")
        return
    if args.limit_rays > 0:
        rays_all = rays_all[: args.limit_rays]
    event_map = {event.event_id: event for event in events}
    station_map = {station.station_id: station for station in stations}
    station_to_rays: dict[int, list[RayPick]] = {}
    for ray in rays_all:
        station_to_rays.setdefault(ray.station_id, []).append(ray)

    grid = _load_grid(step_dir)
    source_files = _find_station_source_files(step_dir)
    if not source_files:
        print(f"Skipping azimuth archive build: no out_data_sim_group source files found in {step_dir}")
        return
    parse_elapsed = time.perf_counter() - t0

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = _create_archive_file(
        archive_path,
        stations,
        events,
        args.phase,
        args.step,
        args.iteration_tag,
        step_dir,
        src_rec_path,
        trace_method="continuous_gradient_descent_on_time_field",
    )

    n_nodes = grid.node_id_grid.size
    station_seen = np.zeros((n_nodes, len(stations)), dtype=bool)
    hit_count = np.zeros(n_nodes, dtype=np.int32)
    az_bins_2d = np.zeros((n_nodes, NAZ_BINS), dtype=bool)
    az_dip_bins_3d = np.zeros((n_nodes, NDIP_BINS, NAZ_BINS), dtype=bool)

    trace_start = time.perf_counter()
    traced_rays = 0
    traced_ok = 0
    for station_id, station_rays in sorted(station_to_rays.items()):
        station = station_map[station_id]
        sim_path = source_files.get(station.name)
        if sim_path is None:
            print(f"Skipping station without source field: {station.name}")
            continue
        station_t0 = time.perf_counter()
        interp_bundle = _station_time_bundle(sim_path, station.name, args.iteration_tag, grid)
        load_elapsed = time.perf_counter() - station_t0

        ray_meta = {
            "ray_id": [],
            "event_id": [],
            "station_id": [],
            "src_lon": [],
            "src_lat": [],
            "src_dep": [],
            "rec_lon": [],
            "rec_lat": [],
            "rec_dep": [],
            "traced_ok": [],
            "n_points": [],
            "path_length_km": [],
        }
        lon_parts: list[np.ndarray] = []
        lat_parts: list[np.ndarray] = []
        dep_parts: list[np.ndarray] = []
        node_hit_parts: list[np.ndarray] = []
        point_offset = int(archive["rays/point_offset"][-1])
        node_hit_offset = int(archive["rays/node_hit_offset"][-1])
        point_offsets = [point_offset]
        node_hit_offsets = [node_hit_offset]

        for ray in station_rays:
            event = event_map.get(ray.event_id)
            if event is None:
                continue
            lon_path, lat_path, dep_path, ok = _trace_ray(
                event,
                station,
                grid.dep,
                grid.lat,
                grid.lon,
                interp_bundle,
                args.step_km,
                args.max_steps,
            )
            path_len = _path_length_km(lon_path, lat_path, dep_path)
            node_hit_ids, node_first_points, _all_points = _path_node_hits(lon_path, lat_path, dep_path, grid)

            if node_hit_ids.size:
                hit_count[node_hit_ids] += 1
                station_seen[node_hit_ids, station.station_id] = True
                for node_id, point_index in zip(node_hit_ids.tolist(), node_first_points.tolist()):
                    if lon_path.size >= 2:
                        if point_index <= 0:
                            i0, i1 = 0, 1
                        elif point_index >= lon_path.size - 1:
                            i0, i1 = lon_path.size - 2, lon_path.size - 1
                        else:
                            i0, i1 = point_index - 1, point_index + 1
                        direction = _direction_from_segment(
                            float(lon_path[i0]),
                            float(lat_path[i0]),
                            float(dep_path[i0]),
                            float(lon_path[i1]),
                            float(lat_path[i1]),
                            float(dep_path[i1]),
                        )
                        if direction is not None:
                            azimuth, dip = direction
                            az_bin = int(math.floor(azimuth / AZ_BIN_WIDTH)) % NAZ_BINS
                            dip_bin = min(int(math.floor(dip / DIP_BIN_WIDTH)), NDIP_BINS - 1)
                            az_bins_2d[node_id, az_bin] = True
                            az_dip_bins_3d[node_id, dip_bin, az_bin] = True

            ray_meta["ray_id"].append(ray.ray_id)
            ray_meta["event_id"].append(ray.event_id)
            ray_meta["station_id"].append(ray.station_id)
            ray_meta["src_lon"].append(event.lon)
            ray_meta["src_lat"].append(event.lat)
            ray_meta["src_dep"].append(event.dep)
            ray_meta["rec_lon"].append(station.lon)
            ray_meta["rec_lat"].append(station.lat)
            ray_meta["rec_dep"].append(station.dep)
            ray_meta["traced_ok"].append(1 if ok else 0)
            ray_meta["n_points"].append(int(lon_path.size))
            ray_meta["path_length_km"].append(path_len)

            lon_parts.append(lon_path)
            lat_parts.append(lat_path)
            dep_parts.append(dep_path)
            node_hit_parts.append(node_hit_ids.astype(np.int32, copy=False))

            point_offset += int(lon_path.size)
            node_hit_offset += int(node_hit_ids.size)
            point_offsets.append(point_offset)
            node_hit_offsets.append(node_hit_offset)

            traced_rays += 1
            traced_ok += int(ok)
            if traced_rays % 1000 == 0:
                elapsed = time.perf_counter() - trace_start
                print(f"Traced {traced_rays} rays in {elapsed:.1f}s (ok={traced_ok})")

        if ray_meta["ray_id"]:
            rays_group = archive["rays"]
            for name, dtype in (
                ("ray_id", np.int64),
                ("event_id", np.int32),
                ("station_id", np.int32),
                ("src_lon", np.float32),
                ("src_lat", np.float32),
                ("src_dep", np.float32),
                ("rec_lon", np.float32),
                ("rec_lat", np.float32),
                ("rec_dep", np.float32),
                ("traced_ok", np.uint8),
                ("n_points", np.int32),
                ("path_length_km", np.float32),
            ):
                _append_1d_dataset(archive, f"rays/{name}", np.asarray(ray_meta[name], dtype=dtype))
            _append_1d_dataset(archive, "rays/lon", np.concatenate(lon_parts).astype(np.float32, copy=False))
            _append_1d_dataset(archive, "rays/lat", np.concatenate(lat_parts).astype(np.float32, copy=False))
            _append_1d_dataset(archive, "rays/dep", np.concatenate(dep_parts).astype(np.float32, copy=False))
            node_concat = np.concatenate(node_hit_parts) if node_hit_parts else np.empty(0, dtype=np.int32)
            _append_1d_dataset(archive, "rays/node_hit_ids", node_concat.astype(np.int32, copy=False))
            _append_1d_dataset(archive, "rays/point_offset", np.asarray(point_offsets[1:], dtype=np.int64))
            _append_1d_dataset(archive, "rays/node_hit_offset", np.asarray(node_hit_offsets[1:], dtype=np.int64))

        print(
            f"Station {station.name}: {len(station_rays)} rays, "
            f"time-field load+grad {load_elapsed:.2f}s, total elapsed {time.perf_counter() - station_t0:.2f}s"
        )

    archive.flush()
    archive.close()
    trace_elapsed = time.perf_counter() - trace_start

    post_t0 = time.perf_counter()
    with h5py.File(archive_path, "r") as handle:
        event_ids = np.asarray(handle["rays/event_id"][...], dtype=np.int32)
        node_hit_offsets = np.asarray(handle["rays/node_hit_offset"][...], dtype=np.int64)
        node_hit_ids = np.asarray(handle["rays/node_hit_ids"][...], dtype=np.int32)

    unique_event_count = np.zeros(n_nodes, dtype=np.int32)
    order = np.argsort(event_ids, kind="mergesort")
    sorted_event_ids = event_ids[order]
    start = 0
    while start < order.size:
        event_id = sorted_event_ids[start]
        stop = start + 1
        while stop < order.size and sorted_event_ids[stop] == event_id:
            stop += 1
        ray_indices = order[start:stop]
        hits: list[np.ndarray] = []
        for ray_index in ray_indices:
            lo = int(node_hit_offsets[ray_index])
            hi = int(node_hit_offsets[ray_index + 1])
            if hi > lo:
                hits.append(node_hit_ids[lo:hi])
        if hits:
            unique_nodes = np.unique(np.concatenate(hits))
            unique_event_count[unique_nodes] += 1
        start = stop
    unique_station_count = station_seen.sum(axis=1).astype(np.int16)
    _write_node_summary(
        summary_path,
        grid,
        hit_count,
        unique_station_count,
        unique_event_count,
        az_bins_2d,
        az_dip_bins_3d,
        archive_path,
        args.phase,
        args.step,
        args.iteration_tag,
    )
    post_elapsed = time.perf_counter() - post_t0

    timing_payload = {
        "run_dir": str(run_dir),
        "step_dir": str(step_dir),
        "src_rec": str(src_rec_path),
        "phase": args.phase,
        "iteration_tag": args.iteration_tag,
        "step_km": args.step_km,
        "max_steps": args.max_steps,
        "rays_requested": len(rays_all),
        "rays_traced": traced_rays,
        "rays_traced_ok": traced_ok,
        "events": len(events),
        "stations": len(stations),
        "grid_shape": [int(grid.dep.size), int(grid.lat.size), int(grid.lon.size)],
        "parse_seconds": parse_elapsed,
        "trace_seconds": trace_elapsed,
        "postprocess_seconds": post_elapsed,
        "total_seconds": time.perf_counter() - t0,
        "archive_path": str(archive_path),
        "summary_path": str(summary_path),
    }
    _timing_report(timing_path, timing_payload)
    print(json.dumps(timing_payload, indent=2))


if __name__ == "__main__":
    main()
