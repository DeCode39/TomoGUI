#!/usr/bin/env python3
"""Render step-by-step frames and an MP4 for one traced ray from an azcov archive."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

MPLCONFIGDIR = Path("/tmp/matplotlib-codex")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EARTH_RADIUS_KM = 6371.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render frames and a video for one traced event-station ray."
    )
    parser.add_argument("--archive", type=Path, default=None, help="Path to the ray archive HDF5.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root output directory. A per-ray subdirectory will be created inside it.",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Path to inversion_P or inversion_S. Used when --archive is omitted.")
    parser.add_argument("--phase", default=None, help="Phase tag used in the archive filename when --archive is omitted.")
    parser.add_argument(
        "--ray-id",
        type=int,
        default=None,
        help="Trace this ray_id directly. If omitted, the longest straight-line event-station pair above min depth is used.",
    )
    parser.add_argument(
        "--min-depth-km",
        type=float,
        default=40.0,
        help="Minimum source depth used when auto-selecting the longest pair.",
    )
    parser.add_argument("--fps", type=int, default=12, help="Output video frame rate.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG and MP4 render DPI.")
    parser.add_argument(
        "--wavefield",
        action="store_true",
        help="Overlay the reciprocal station-source travel-time field on the panels.",
    )
    parser.add_argument(
        "--step-dir",
        type=Path,
        default=None,
        help="Path to OUTPUT_FILES/OUTPUT_FILES_joint_step2. Defaults to the sibling of the archive's inversion folder.",
    )
    parser.add_argument(
        "--iteration-tag",
        default="0039",
        help="Iteration tag for the saved time field, e.g. 0039.",
    )
    parser.add_argument(
        "--keep-existing-frames",
        action="store_true",
        help="Do not delete existing frame PNGs before rendering.",
    )
    return parser.parse_args()


def _decode_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _straight_line_distance_km(
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    src_dep: np.ndarray,
    rec_lon: np.ndarray,
    rec_lat: np.ndarray,
    rec_dep: np.ndarray,
) -> np.ndarray:
    mean_lat = np.deg2rad((src_lat + rec_lat) * 0.5)
    dx = (rec_lon - src_lon) * 111.32 * np.cos(mean_lat)
    dy = (rec_lat - src_lat) * 111.32
    dz = rec_dep - src_dep
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def _select_row(h5f: h5py.File, ray_id: int | None, min_depth_km: float) -> tuple[int, dict[str, object]]:
    rays = h5f["rays"]
    stations = h5f["stations"]
    events = h5f["events"]

    ray_ids = np.asarray(rays["ray_id"][...], dtype=np.int64)
    if ray_id is not None:
        matches = np.nonzero(ray_ids == ray_id)[0]
        if matches.size == 0:
            raise ValueError(f"ray_id {ray_id} not found in archive.")
        row = int(matches[0])
    else:
        ok = np.asarray(rays["traced_ok"][...], dtype=np.uint8).astype(bool)
        src_dep = np.asarray(rays["src_dep"][...], dtype=np.float32)
        mask = ok & np.isfinite(src_dep) & (src_dep > min_depth_km)
        rows = np.nonzero(mask)[0]
        if rows.size == 0:
            raise ValueError(f"No traced rays found with source depth > {min_depth_km:.1f} km.")
        dist = _straight_line_distance_km(
            np.asarray(rays["src_lon"][rows], dtype=np.float32),
            np.asarray(rays["src_lat"][rows], dtype=np.float32),
            np.asarray(rays["src_dep"][rows], dtype=np.float32),
            np.asarray(rays["rec_lon"][rows], dtype=np.float32),
            np.asarray(rays["rec_lat"][rows], dtype=np.float32),
            np.asarray(rays["rec_dep"][rows], dtype=np.float32),
        )
        row = int(rows[int(np.argmax(dist))])

    station_id = int(rays["station_id"][row])
    event_id = int(rays["event_id"][row])
    event_rows = np.asarray(events["event_id"][...], dtype=np.int32)
    event_match = np.nonzero(event_rows == event_id)[0]
    if event_match.size == 0:
        raise ValueError(f"Event {event_id} was not found in /events.")
    event_row = int(event_match[0])

    info = {
        "row": row,
        "ray_id": int(ray_ids[row]),
        "event_id": event_id,
        "event_row": event_row,
        "station_id": station_id,
        "station_name": _decode_string(stations["name"][station_id]),
        "source_lon": float(rays["src_lon"][row]),
        "source_lat": float(rays["src_lat"][row]),
        "source_dep": float(rays["src_dep"][row]),
        "receiver_lon": float(rays["rec_lon"][row]),
        "receiver_lat": float(rays["rec_lat"][row]),
        "receiver_dep": float(rays["rec_dep"][row]),
        "n_points": int(rays["n_points"][row]),
        "path_length_km": float(rays["path_length_km"][row]),
        "event_npicks": int(events["npicks"][event_row]),
    }
    info["straight_line_distance_km"] = float(
        _straight_line_distance_km(
            np.array([info["source_lon"]], dtype=np.float32),
            np.array([info["source_lat"]], dtype=np.float32),
            np.array([info["source_dep"]], dtype=np.float32),
            np.array([info["receiver_lon"]], dtype=np.float32),
            np.array([info["receiver_lat"]], dtype=np.float32),
            np.array([info["receiver_dep"]], dtype=np.float32),
        )[0]
    )
    return row, info


def _load_ray_path(h5f: h5py.File, row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rays = h5f["rays"]
    offsets = np.asarray(rays["point_offset"][row : row + 2], dtype=np.int64)
    lo = int(offsets[0])
    hi = int(offsets[1])
    lon = np.asarray(rays["lon"][lo:hi], dtype=np.float32)
    lat = np.asarray(rays["lat"][lo:hi], dtype=np.float32)
    dep = np.asarray(rays["dep"][lo:hi], dtype=np.float32)
    return lon, lat, dep


def _axis_edges(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.size == 1:
        return np.array([coords[0] - 0.5, coords[0] + 0.5], dtype=float)
    mids = 0.5 * (coords[:-1] + coords[1:])
    start = coords[0] - (mids[0] - coords[0])
    end = coords[-1] + (coords[-1] - mids[-1])
    return np.concatenate([[start], mids, [end]])


def _infer_step_dir(archive_path: Path) -> Path:
    inversion_dir = archive_path.parent.parent
    candidate = inversion_dir / "OUTPUT_FILES" / "OUTPUT_FILES_joint_step2"
    return candidate


def _infer_phase(run_dir: Path) -> str:
    name = run_dir.name.upper()
    if name.endswith("_S"):
        return "S"
    return "P"


def _load_grid(step_dir: Path) -> dict[str, np.ndarray]:
    with h5py.File(step_dir / "out_data_grid.h5", "r") as handle:
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
    return {
        "lon": lon,
        "lat": lat,
        "dep": dep,
        "lon_edges": _axis_edges(lon),
        "lat_edges": _axis_edges(lat),
        "dep_edges": _axis_edges(dep),
        "node_id_grid": node_id_grid,
    }


def _find_station_group_file(step_dir: Path, station_name: str) -> tuple[Path, str]:
    group_name = f"src_{station_name}"
    for sim_path in sorted(step_dir.glob("out_data_sim_group_*.h5")):
        with h5py.File(sim_path, "r") as handle:
            if group_name in handle:
                return sim_path, group_name
    raise FileNotFoundError(f"{group_name} was not found under {step_dir}")


def _load_time_grid(step_dir: Path, station_name: str, iteration_tag: str, grid: dict[str, np.ndarray]) -> np.ndarray:
    sim_path, group_name = _find_station_group_file(step_dir, station_name)
    dataset_name = f"time_field_inv_{iteration_tag}"
    with h5py.File(sim_path, "r") as handle:
        flat = np.asarray(handle[group_name][dataset_name][...], dtype=np.float64)
    return flat[grid["node_id_grid"]]


def _compute_limits(lon: np.ndarray, lat: np.ndarray, dep: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    lon_min = float(np.min(lon))
    lon_max = float(np.max(lon))
    lat_min = float(np.min(lat))
    lat_max = float(np.max(lat))
    dep_min = float(np.min(dep))
    dep_max = float(np.max(dep))

    lon_pad = max(0.08, (lon_max - lon_min) * 0.12)
    lat_pad = max(0.08, (lat_max - lat_min) * 0.12)
    dep_pad = max(2.0, (dep_max - dep_min) * 0.08)
    return (
        (lon_min - lon_pad, lon_max + lon_pad),
        (lat_min - lat_pad, lat_max + lat_pad),
        (max(-1.0, dep_min - dep_pad), dep_max + dep_pad),
    )


def _render_frame(
    frame_index: int,
    lon: np.ndarray,
    lat: np.ndarray,
    dep: np.ndarray,
    info: dict[str, object],
    x_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    frame_path: Path,
    dpi: int,
    wavefield: dict[str, np.ndarray] | None,
) -> None:
    current_lon = float(lon[frame_index])
    current_lat = float(lat[frame_index])
    current_dep = float(dep[frame_index])
    frac = (frame_index + 1) / lon.size

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.6))
    fig.suptitle(
        (
            f"Single-ray trace | event {info['event_id']} to station {info['station_name']} "
            f"(ray_id={info['ray_id']})\n"
            f"step {frame_index + 1}/{lon.size} | event depth={info['source_dep']:.2f} km | "
            f"straight={info['straight_line_distance_km']:.1f} km | traced={info['path_length_km']:.1f} km"
        ),
        fontsize=13,
    )

    future_style = {"color": "white" if wavefield is not None else "0.78", "linewidth": 1.0, "alpha": 0.55 if wavefield is not None else 0.7, "zorder": 2}
    past_style = {"color": "#ffb703" if wavefield is not None else "#0b5fa5", "linewidth": 2.0, "alpha": 0.98, "zorder": 4}

    if wavefield is not None:
        lon_i = int(np.argmin(np.abs(wavefield["lon"] - current_lon)))
        lat_i = int(np.argmin(np.abs(wavefield["lat"] - current_lat)))
        dep_i = int(np.argmin(np.abs(wavefield["dep"] - current_dep)))

        m0 = axes[0].pcolormesh(
            wavefield["lon_edges"],
            wavefield["lat_edges"],
            np.ma.masked_invalid(wavefield["time_grid"][dep_i, :, :]),
            shading="auto",
            cmap="viridis",
            zorder=0,
        )
        axes[1].pcolormesh(
            wavefield["lon_edges"],
            wavefield["dep_edges"],
            np.ma.masked_invalid(wavefield["time_grid"][:, lat_i, :]),
            shading="auto",
            cmap="viridis",
            zorder=0,
        )
        axes[2].pcolormesh(
            wavefield["lat_edges"],
            wavefield["dep_edges"],
            np.ma.masked_invalid(wavefield["time_grid"][:, :, lon_i]),
            shading="auto",
            cmap="viridis",
            zorder=0,
        )
        cbar = fig.colorbar(m0, ax=axes, fraction=0.03, pad=0.025)
        cbar.set_label("Travel time from station source (s)")

    axes[0].plot(lon, lat, **future_style)
    axes[0].plot(lon[: frame_index + 1], lat[: frame_index + 1], **past_style)
    axes[0].scatter([info["source_lon"]], [info["source_lat"]], s=90, marker="*", c="#e76f51", edgecolors="black", linewidths=0.5, zorder=4)
    axes[0].scatter([info["receiver_lon"]], [info["receiver_lat"]], s=85, marker="^", c="#264653", edgecolors="black", linewidths=0.5, zorder=4)
    axes[0].scatter([current_lon], [current_lat], s=60, marker="o", c="#ffb703", edgecolors="black", linewidths=0.6, zorder=5)
    axes[0].set_title("Longitude-Latitude")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_xlim(*x_limits[0])
    axes[0].set_ylim(*x_limits[1])
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(lon, dep, **future_style)
    axes[1].plot(lon[: frame_index + 1], dep[: frame_index + 1], **past_style)
    axes[1].scatter([info["source_lon"]], [info["source_dep"]], s=90, marker="*", c="#e76f51", edgecolors="black", linewidths=0.5, zorder=4)
    axes[1].scatter([info["receiver_lon"]], [info["receiver_dep"]], s=85, marker="^", c="#264653", edgecolors="black", linewidths=0.5, zorder=4)
    axes[1].scatter([current_lon], [current_dep], s=60, marker="o", c="#ffb703", edgecolors="black", linewidths=0.6, zorder=5)
    axes[1].set_title("Longitude-Depth")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Depth (km)")
    axes[1].set_xlim(*x_limits[0])
    axes[1].set_ylim(*x_limits[2])
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(lat, dep, **future_style)
    axes[2].plot(lat[: frame_index + 1], dep[: frame_index + 1], **past_style)
    axes[2].scatter([info["source_lat"]], [info["source_dep"]], s=90, marker="*", c="#e76f51", edgecolors="black", linewidths=0.5, zorder=4)
    axes[2].scatter([info["receiver_lat"]], [info["receiver_dep"]], s=85, marker="^", c="#264653", edgecolors="black", linewidths=0.5, zorder=4)
    axes[2].scatter([current_lat], [current_dep], s=60, marker="o", c="#ffb703", edgecolors="black", linewidths=0.6, zorder=5)
    axes[2].set_title("Latitude-Depth")
    axes[2].set_xlabel("Latitude")
    axes[2].set_ylabel("Depth (km)")
    axes[2].set_xlim(*x_limits[1])
    axes[2].set_ylim(*x_limits[2])
    axes[2].invert_yaxis()
    axes[2].grid(True, alpha=0.25)

    note_lines = [
        f"progress: {frac:5.1%}",
        f"current lon: {current_lon:.4f}",
        f"current lat: {current_lat:.4f}",
        f"current dep: {current_dep:.2f} km",
        f"event picks: {info['event_npicks']}",
    ]
    if wavefield is not None:
        note_lines.extend(
            [
                "",
                "wavefield slices:",
                f"map depth ~ {float(wavefield['dep'][dep_i]):.1f} km",
                f"lon-depth lat ~ {float(wavefield['lat'][lat_i]):.3f}",
                f"lat-depth lon ~ {float(wavefield['lon'][lon_i]):.3f}",
            ]
        )

    axes[2].text(
        0.02,
        0.02,
        "\n".join(note_lines),
        transform=axes[2].transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.6"},
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(frame_path, dpi=dpi)
    plt.close(fig)


def _write_video(frames_dir: Path, video_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH.")
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _parse_args()
    if args.run_dir is None:
        args.run_dir = Path(__file__).resolve().parent
    args.run_dir = args.run_dir.resolve()
    if args.phase is None:
        args.phase = _infer_phase(args.run_dir)
    if args.archive is None:
        args.archive = args.run_dir / "azcov_archive" / f"azcov_{args.phase}_joint_step2_ray_archive.h5"
    args.archive = args.archive.resolve()
    if args.output_root is None:
        args.output_root = args.run_dir / "img" / "single_ray_trace"
    args.output_root = args.output_root.resolve()
    if not args.archive.exists():
        print(f"Skipping single-ray trace: archive not found: {args.archive}")
        return
    if args.step_dir is None:
        args.step_dir = _infer_step_dir(args.archive)
    else:
        args.step_dir = args.step_dir.resolve()

    with h5py.File(args.archive, "r") as h5f:
        row, info = _select_row(h5f, args.ray_id, args.min_depth_km)
        lon, lat, dep = _load_ray_path(h5f, row)

    wavefield = None
    if args.wavefield:
        if not args.step_dir.exists():
            print(f"Skipping wavefield overlay: step dir not found: {args.step_dir}")
            args.wavefield = False
        else:
            try:
                grid = _load_grid(args.step_dir)
                wavefield = {
                    **grid,
                    "time_grid": _load_time_grid(args.step_dir, str(info["station_name"]), args.iteration_tag, grid),
                }
            except Exception as exc:
                print(f"Skipping wavefield overlay: {exc}")
                args.wavefield = False

    out_dir = args.output_root / (
        f"ray_{info['ray_id']}_event_{info['event_id']}_station_{info['station_name']}"
    )
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing_frames:
        for frame in frames_dir.glob("frame_*.png"):
            frame.unlink()

    limits = _compute_limits(
        np.concatenate([lon, np.array([info["source_lon"], info["receiver_lon"]], dtype=np.float32)]),
        np.concatenate([lat, np.array([info["source_lat"], info["receiver_lat"]], dtype=np.float32)]),
        np.concatenate([dep, np.array([info["source_dep"], info["receiver_dep"]], dtype=np.float32)]),
    )

    for idx in range(lon.size):
        _render_frame(
            frame_index=idx,
            lon=lon,
            lat=lat,
            dep=dep,
            info=info,
            x_limits=limits,
            frame_path=frames_dir / f"frame_{idx:04d}.png",
            dpi=args.dpi,
            wavefield=wavefield,
        )

    video_path = out_dir / f"ray_trace_event_{info['event_id']}_station_{info['station_name']}.mp4"
    try:
        _write_video(frames_dir, video_path, args.fps)
    except RuntimeError as exc:
        print(f"Skipping MP4 creation: {exc}")
        video_path = None

    summary = {
        "archive": str(args.archive),
        "output_dir": str(out_dir),
        "video_path": str(video_path) if video_path is not None else "",
        "frames_dir": str(frames_dir),
        "fps": int(args.fps),
        "dpi": int(args.dpi),
        "wavefield": bool(args.wavefield),
        "step_dir": str(args.step_dir),
        "iteration_tag": args.iteration_tag,
        **info,
    }
    (out_dir / "trace_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "trace_summary.txt").write_text(
        "\n".join(
            [
                "Single-ray trace summary",
                f"archive: {args.archive}",
                f"output_dir: {out_dir}",
                f"video_path: {video_path or '(not written)'}",
                "",
                f"ray_id: {info['ray_id']}",
                f"event_id: {info['event_id']}",
                f"station_id: {info['station_id']}",
                f"station_name: {info['station_name']}",
                f"source_lon_lat_dep: {info['source_lon']:.6f}, {info['source_lat']:.6f}, {info['source_dep']:.3f}",
                f"receiver_lon_lat_dep: {info['receiver_lon']:.6f}, {info['receiver_lat']:.6f}, {info['receiver_dep']:.3f}",
                f"event_npicks: {info['event_npicks']}",
                f"n_points: {info['n_points']}",
                f"path_length_km: {info['path_length_km']:.3f}",
                f"straight_line_distance_km: {info['straight_line_distance_km']:.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Rendered {lon.size} frames into {frames_dir}")
    if video_path is not None:
        print(f"Wrote video: {video_path}")
    else:
        print("MP4 not written.")


if __name__ == "__main__":
    main()
