#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path

MPLCONFIGDIR = Path("/tmp/matplotlib-codex")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OBJ_RE = re.compile(
    r"iteration:\s*(\d+),\s*objective function:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


@dataclass
class EventRecord:
    event_id: int
    lat: float
    lon: float
    dep: float
    mag: float
    nrec: int


@dataclass
class PickRecord:
    event_id: int
    station: str
    rec_lat: float
    rec_lon: float
    phase: str
    travel_time: float
    weight: float


def numeric_key(path: Path) -> tuple[int, ...]:
    nums = [int(token) for token in re.findall(r"\d+", path.stem)]
    return tuple(nums) if nums else (-1,)


def km_per_deg_lon(lat_deg: float) -> float:
    return 111.32 * max(math.cos(math.radians(lat_deg)), 1e-6)


def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    r = 6371.0
    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def parse_src_rec(path: Path) -> tuple[dict[int, EventRecord], list[PickRecord]]:
    events: dict[int, EventRecord] = {}
    picks: list[PickRecord] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 12:
            continue
        try:
            event_id = int(tokens[0])
            lat = float(tokens[7])
            lon = float(tokens[8])
            dep = float(tokens[9])
            mag = float(tokens[10])
            nrec = int(float(tokens[11]))
        except ValueError:
            continue
        events[event_id] = EventRecord(event_id=event_id, lat=lat, lon=lon, dep=dep, mag=mag, nrec=nrec)

        read_rec = 0
        while read_rec < nrec and i < len(lines):
            rec_line = lines[i].strip()
            i += 1
            if not rec_line:
                continue
            rt = rec_line.split()
            if len(rt) < 8:
                continue
            if rt[6].upper() not in {"P", "S"}:
                continue
            try:
                rec_event_id = int(rt[0])
                station = rt[2]
                rec_lat = float(rt[3])
                rec_lon = float(rt[4])
                phase = rt[6].upper()
                travel_time = float(rt[7])
                weight = float(rt[8]) if len(rt) >= 9 else 1.0
            except ValueError:
                continue
            picks.append(
                PickRecord(
                    event_id=rec_event_id,
                    station=station,
                    rec_lat=rec_lat,
                    rec_lon=rec_lon,
                    phase=phase,
                    travel_time=travel_time,
                    weight=weight,
                )
            )
            read_rec += 1
    return events, picks


def parse_objective_from_out(path: Path) -> tuple[np.ndarray, np.ndarray]:
    idx: list[int] = []
    obj: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = OBJ_RE.search(line)
        if not match:
            continue
        idx.append(int(match.group(1)))
        obj.append(float(match.group(2)))
    return np.asarray(idx, dtype=int), np.asarray(obj, dtype=float)


def parse_objective_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    iterations: list[int] = []
    obj: list[float] = []
    obj_abs: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 4:
            continue
        try:
            iterations.append(int(parts[0]))
            obj.append(float(parts[2]))
            obj_abs.append(float(parts[3]))
        except ValueError:
            continue

    if not iterations:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    latest: dict[int, tuple[float, float]] = {}
    for iteration, obj_val, obj_abs_val in zip(iterations, obj, obj_abs):
        latest[iteration] = (obj_val, obj_abs_val)
    uniq_iters = np.asarray(sorted(latest.keys()), dtype=int)
    uniq_obj = np.asarray([latest[i][0] for i in uniq_iters], dtype=float)
    uniq_obj_abs = np.asarray([latest[i][1] for i in uniq_iters], dtype=float)
    return uniq_iters, uniq_obj, uniq_obj_abs


def step_dirs(output_root: Path) -> list[Path]:
    return sorted([p for p in output_root.glob("OUTPUT_FILES_*") if p.is_dir()], key=numeric_key)


def objective_series_from_step_files(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[int], list[str], str]:
    output_root = run_dir / "OUTPUT_FILES"
    if not output_root.exists():
        return np.array([], dtype=float), np.array([], dtype=float), [], [], ""

    x_all: list[float] = []
    y_all: list[float] = []
    breaks: list[int] = []
    labels: list[str] = []
    offset = 0
    used = 0

    for step in step_dirs(output_root):
        obj_file = step / "objective_function.txt"
        if not obj_file.exists():
            continue
        _iters, _obj, obj_abs = parse_objective_file(obj_file)
        if obj_abs.size == 0:
            continue
        x = np.arange(offset, offset + obj_abs.size, dtype=float)
        x_all.extend(x.tolist())
        y_all.extend(obj_abs.tolist())
        breaks.append(offset)
        labels.append(step.name)
        offset += int(obj_abs.size)
        used += 1

    if used == 0:
        return np.array([], dtype=float), np.array([], dtype=float), [], [], ""
    return np.asarray(x_all, dtype=float), np.asarray(y_all, dtype=float), breaks, labels, "obj_abs from objective_function.txt"


def objective_series_from_log(latest_out: Path) -> tuple[np.ndarray, np.ndarray, list[int], list[str], str]:
    iterations, values = parse_objective_from_out(latest_out)
    if values.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float), [], [], ""

    segments: list[np.ndarray] = []
    start = 0
    for i in range(1, values.size):
        if iterations[i] <= iterations[i - 1]:
            segments.append(values[start:i])
            start = i
    segments.append(values[start:])

    kept: list[np.ndarray] = []
    nonzero_seen = False
    for seg in segments:
        if seg.size == 0:
            continue
        if np.any(seg != 0.0):
            kept.append(seg)
            nonzero_seen = True
        elif not nonzero_seen:
            kept.append(seg)
    if not kept:
        kept = segments

    x_all: list[float] = []
    y_all: list[float] = []
    breaks: list[int] = []
    labels: list[str] = []
    offset = 0
    for idx, seg in enumerate(kept):
        x = np.arange(offset, offset + seg.size, dtype=float)
        x_all.extend(x.tolist())
        y_all.extend(seg.tolist())
        breaks.append(offset)
        labels.append(f"log_stage_{idx + 1}")
        offset += int(seg.size)
    return np.asarray(x_all, dtype=float), np.asarray(y_all, dtype=float), breaks, labels, f"objective from {latest_out.name}"


def nonobs_src_files(step_dir: Path) -> list[Path]:
    files = [p for p in step_dir.glob("src_rec_file*.dat") if "_obs" not in p.name]
    return sorted(files, key=numeric_key)


def closest_obs_for(nonobs_file: Path, obs_files: list[Path]) -> Path | None:
    direct = nonobs_file.with_name(nonobs_file.stem + "_obs.dat")
    if direct.exists():
        return direct
    if not obs_files:
        return None
    return sorted(obs_files, key=numeric_key)[-1]


def original_src_file(run_dir: Path) -> Path | None:
    candidate = run_dir / "1_src_rec_files" / "src_rec_config.dat"
    if candidate.exists():
        return candidate
    return None


def plot_objective(run_dir: Path, latest_out: Path, out_dir: Path) -> Path | None:
    x, vals, breaks, labels, source = objective_series_from_step_files(run_dir)
    if vals.size == 0:
        x, vals, breaks, labels, source = objective_series_from_log(latest_out)
    if vals.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(x, vals, "-o", ms=2.8, lw=1.2, color="tab:blue")
    ax.set_xlabel("Optimization progress index")
    ax.set_ylabel("Objective function")
    ax.set_title("Objective function history")
    ax.grid(alpha=0.25, linestyle="--")

    for idx, b in enumerate(breaks):
        if idx == 0:
            continue
        ax.axvline(float(b) - 0.5, color="0.5", lw=0.9, ls=":")
    if breaks:
        for b, lab in zip(breaks, labels):
            ax.text(float(b), 0.02, lab, rotation=90, va="bottom", ha="left", transform=ax.get_xaxis_transform(), fontsize=8)

    tail_n = min(10, max(1, vals.size - 1))
    local_drop = float(np.median(np.abs(np.diff(vals[-(tail_n + 1) :]))) if vals.size > 1 else 0.0)
    text = (
        f"source={source}\n"
        f"first={vals[0]:.3f}\n"
        f"last={vals[-1]:.3f}\n"
        f"ratio={vals[-1] / vals[0]:.3f}\n"
        f"median |d| over last {tail_n} steps={local_drop:.4f}"
    )
    ax.text(0.98, 0.97, text, transform=ax.transAxes, ha="right", va="top", fontsize=9)
    out = out_dir / "qc_objective_history.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_relocation(step_name: str, init_file: Path, final_file: Path, out_dir: Path) -> Path | None:
    init_events, _ = parse_src_rec(init_file)
    final_events, _ = parse_src_rec(final_file)
    common = sorted(set(init_events).intersection(final_events))
    if not common:
        return None

    lon0 = np.asarray([init_events[e].lon for e in common], dtype=float)
    lat0 = np.asarray([init_events[e].lat for e in common], dtype=float)
    dep0 = np.asarray([init_events[e].dep for e in common], dtype=float)
    lon1 = np.asarray([final_events[e].lon for e in common], dtype=float)
    lat1 = np.asarray([final_events[e].lat for e in common], dtype=float)
    dep1 = np.asarray([final_events[e].dep for e in common], dtype=float)

    hshift = haversine_km(lon0, lat0, lon1, lat1)
    dshift = dep1 - dep0
    med_lat = float(np.nanmedian(np.r_[lat0, lat1]))
    u = (lon1 - lon0) * km_per_deg_lon(med_lat)
    v = (lat1 - lat0) * 110.574

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    axm, axh, axd = axes

    axm.scatter(lon0, lat0, s=8, color="0.7", alpha=0.55, label="initial")
    q = axm.quiver(lon0, lat0, u, v, hshift, cmap="viridis", angles="xy", scale_units="xy", scale=1.0, width=0.0022)
    axm.scatter(lon1, lat1, s=8, color="k", alpha=0.55, label="final")
    cb = fig.colorbar(q, ax=axm, shrink=0.86)
    cb.set_label("Horizontal shift (km)")
    axm.set_xlabel("Longitude")
    axm.set_ylabel("Latitude")
    axm.set_title(f"{step_name}: relocation vectors from original input")
    axm.grid(alpha=0.2, linestyle=":")
    axm.legend(loc="best", fontsize=8)

    axh.hist(hshift, bins=45, color="tab:blue", alpha=0.82)
    axh.set_xlabel("Horizontal shift (km)")
    axh.set_ylabel("Event count")
    axh.set_title("Horizontal displacement")
    axh.grid(alpha=0.2, linestyle=":")

    axd.hist(dshift, bins=45, color="tab:orange", alpha=0.82)
    axd.set_xlabel("Depth shift (km, final - initial)")
    axd.set_ylabel("Event count")
    axd.set_title("Depth displacement")
    axd.grid(alpha=0.2, linestyle=":")

    stats = (
        f"n={len(common)}\n"
        f"hshift median={np.median(hshift):.3f} km\n"
        f"hshift p90={np.quantile(hshift, 0.9):.3f} km\n"
        f"|dshift| median={np.median(np.abs(dshift)):.3f} km"
    )
    axd.text(0.98, 0.96, stats, transform=axd.transAxes, ha="right", va="top", fontsize=9)

    out = out_dir / f"qc_relocation_{step_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def residual_arrays(pred_file: Path, obs_file: Path) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], list[str]]:
    pred_events, pred_picks = parse_src_rec(pred_file)
    _, obs_picks = parse_src_rec(obs_file)
    obs_map: dict[tuple[int, str, str], float] = {}
    for p in obs_picks:
        obs_map[(p.event_id, p.station, p.phase)] = p.travel_time

    residuals: list[float] = []
    distances: list[float] = []
    sta_lon: list[float] = []
    sta_lat: list[float] = []
    evt_key: list[int] = []
    station_key: list[str] = []

    for p in pred_picks:
        key = (p.event_id, p.station, p.phase)
        if key not in obs_map or p.event_id not in pred_events:
            continue
        event = pred_events[p.event_id]
        residual = p.travel_time - obs_map[key]
        dist = haversine_km(
            np.asarray([event.lon]),
            np.asarray([event.lat]),
            np.asarray([p.rec_lon]),
            np.asarray([p.rec_lat]),
        )[0]
        residuals.append(float(residual))
        distances.append(float(dist))
        sta_lon.append(float(p.rec_lon))
        sta_lat.append(float(p.rec_lat))
        evt_key.append(int(p.event_id))
        station_key.append(p.station)

    return (
        np.asarray(residuals, dtype=float),
        np.asarray(distances, dtype=float),
        np.asarray(sta_lon, dtype=float),
        np.asarray(sta_lat, dtype=float),
        np.asarray(evt_key, dtype=int),
    ), station_key


def plot_residual_qc(step_name: str, pred_file: Path, obs_file: Path, out_dir: Path) -> Path | None:
    (residuals, distances, sta_lon, sta_lat, event_ids), station_names = residual_arrays(pred_file, obs_file)
    if residuals.size == 0:
        return None

    station_stats: dict[str, list[float]] = defaultdict(list)
    station_coords: dict[str, tuple[float, float]] = {}
    for r, lon, lat, sta in zip(residuals, sta_lon, sta_lat, station_names):
        station_stats[sta].append(float(r))
        station_coords[sta] = (float(lon), float(lat))

    event_stats: dict[int, list[float]] = defaultdict(list)
    for r, eid in zip(residuals, event_ids):
        event_stats[int(eid)].append(float(r))

    sta_mean = np.asarray([np.mean(vals) for vals in station_stats.values()], dtype=float)
    sta_count = np.asarray([len(vals) for vals in station_stats.values()], dtype=int)
    sta_lons = np.asarray([station_coords[s][0] for s in station_stats.keys()], dtype=float)
    sta_lats = np.asarray([station_coords[s][1] for s in station_stats.keys()], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    axh, axd, axs, axe = axes.ravel()

    axh.hist(residuals, bins=65, color="tab:purple", alpha=0.83)
    axh.set_xlabel("Residual (pred - obs) [s]")
    axh.set_ylabel("Pick count")
    axh.set_title(f"{step_name}: residual histogram")
    axh.grid(alpha=0.2, linestyle=":")

    axd.scatter(distances, residuals, s=7, alpha=0.3, color="tab:blue")
    axd.axhline(0.0, color="k", lw=0.9)
    axd.set_xlabel("Event-station distance (km)")
    axd.set_ylabel("Residual (s)")
    axd.set_title("Residual vs distance")
    axd.grid(alpha=0.2, linestyle=":")

    vmax = float(np.nanpercentile(np.abs(sta_mean), 95))
    vmax = max(vmax, 1e-4)
    scs = axs.scatter(sta_lons, sta_lats, c=sta_mean, s=np.clip(sta_count * 2.0, 8, 90), cmap="seismic", vmin=-vmax, vmax=vmax)
    cbs = fig.colorbar(scs, ax=axs, shrink=0.86)
    cbs.set_label("Mean residual (s)")
    axs.set_xlabel("Longitude")
    axs.set_ylabel("Latitude")
    axs.set_title("Station mean residual")
    axs.grid(alpha=0.2, linestyle=":")

    unique_evt, inv = np.unique(event_ids, return_inverse=True)
    counts = np.bincount(inv)
    means = np.zeros_like(unique_evt, dtype=float)
    for i, eid in enumerate(unique_evt):
        means[i] = np.mean(residuals[event_ids == eid])
    sce = axe.scatter(np.arange(unique_evt.size), means, c=counts, cmap="viridis", s=10)
    cbe = fig.colorbar(sce, ax=axe, shrink=0.86)
    cbe.set_label("Picks per event")
    axe.axhline(0.0, color="k", lw=0.9)
    axe.set_xlabel("Event index")
    axe.set_ylabel("Mean residual (s)")
    axe.set_title("Event mean residual")
    axe.grid(alpha=0.2, linestyle=":")

    summary = (
        f"n picks = {residuals.size}\n"
        f"mean = {np.mean(residuals):.4f} s\n"
        f"std = {np.std(residuals):.4f} s\n"
        f"median = {np.median(residuals):.4f} s\n"
        f"p95(|res|) = {np.quantile(np.abs(residuals), 0.95):.4f} s"
    )
    axh.text(0.98, 0.98, summary, transform=axh.transAxes, ha="right", va="top", fontsize=9)

    out = out_dir / f"qc_residuals_{step_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_data_coverage(step_name: str, src_file: Path, out_dir: Path) -> Path | None:
    events, picks = parse_src_rec(src_file)
    if not events:
        return None
    ev_lon = np.asarray([e.lon for e in events.values()], dtype=float)
    ev_lat = np.asarray([e.lat for e in events.values()], dtype=float)
    ev_dep = np.asarray([e.dep for e in events.values()], dtype=float)

    sta_by_name: dict[str, tuple[float, float]] = {}
    for p in picks:
        sta_by_name[p.station] = (p.rec_lon, p.rec_lat)
    st_lon = np.asarray([v[0] for v in sta_by_name.values()], dtype=float) if sta_by_name else np.array([], dtype=float)
    st_lat = np.asarray([v[1] for v in sta_by_name.values()], dtype=float) if sta_by_name else np.array([], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    axm, axd = axes
    if st_lon.size:
        axm.scatter(st_lon, st_lat, s=20, marker="^", color="tab:blue", alpha=0.8, label=f"stations ({st_lon.size})")
    axm.scatter(ev_lon, ev_lat, s=8, marker="o", color="tab:red", alpha=0.35, label=f"events ({ev_lon.size})")
    axm.set_xlabel("Longitude")
    axm.set_ylabel("Latitude")
    axm.set_title(f"{step_name}: event/station coverage")
    axm.grid(alpha=0.2, linestyle=":")
    axm.legend(loc="best", fontsize=8)

    axd.hist(ev_dep, bins=45, color="tab:orange", alpha=0.85)
    axd.invert_xaxis()
    axd.set_xlabel("Depth (km)")
    axd.set_ylabel("Event count")
    axd.set_title("Event depth distribution")
    axd.grid(alpha=0.2, linestyle=":")

    out = out_dir / f"qc_coverage_{step_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def run_qc(run_dir: Path, output_subdir: str) -> list[Path]:
    out_dir = run_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    out_logs = sorted(run_dir.glob("tomoatt_*.out"), key=lambda p: p.stat().st_mtime)
    if out_logs:
        obj_png = plot_objective(run_dir, out_logs[-1], out_dir)
        if obj_png:
            generated.append(obj_png)

    of_root = run_dir / "OUTPUT_FILES"
    if not of_root.exists():
        return generated

    baseline_src = original_src_file(run_dir)
    steps = step_dirs(of_root)
    for step in steps:
        step_name = step.name
        nonobs = nonobs_src_files(step)
        obs = sorted(step.glob("src_rec_file*_obs.dat"), key=numeric_key)
        reloc_init = baseline_src if baseline_src is not None else (nonobs[0] if len(nonobs) >= 2 else None)
        if reloc_init is not None and nonobs:
            reloc_png = plot_relocation(step_name, reloc_init, nonobs[-1], out_dir)
            if reloc_png:
                generated.append(reloc_png)
        if nonobs and obs:
            obs_match = closest_obs_for(nonobs[-1], obs)
            if obs_match is not None:
                res_png = plot_residual_qc(step_name, nonobs[-1], obs_match, out_dir)
                if res_png:
                    generated.append(res_png)

    if steps:
        latest_step = steps[-1]
        latest_nonobs = nonobs_src_files(latest_step)
        if latest_nonobs:
            cov_png = plot_data_coverage(latest_step.name, latest_nonobs[-1], out_dir)
            if cov_png:
                generated.append(cov_png)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QC plots from TomoATT inversion outputs.")
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent, help="Inversion run directory")
    parser.add_argument("--output-subdir", type=str, default="img/qc", help="Output subdirectory under run-dir")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        print(f"Skipping QC plots: run directory does not exist: {run_dir}")
        return

    generated = run_qc(run_dir, args.output_subdir)
    if not generated:
        print(f"Skipping QC plots: no plottable QC inputs found in {run_dir}")
        return

    print("Generated QC plots:")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()
