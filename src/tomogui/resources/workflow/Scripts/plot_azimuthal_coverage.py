#!/usr/bin/env python3
"""Render compact azimuthal-coverage summary plots from an azcov node archive."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
MPLCONFIGDIR = Path("/tmp/matplotlib-codex")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DEPTHS = (10.0, 25.0, 40.0)
CMAP = "Greys"
DEFAULT_TOP_NODES = 5


def _infer_phase(run_dir: Path) -> str:
    name = run_dir.name.upper()
    if name.endswith("_S"):
        return "S"
    return "P"


def _parse_depths(text: str) -> tuple[float, ...]:
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    return tuple(values) if values else DEFAULT_DEPTHS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent, help="Path to inversion_P or inversion_S folder.")
    parser.add_argument("--phase", default=None, help="Phase tag used in archive filename. Defaults to phase inferred from folder name.")
    parser.add_argument("--step", default="joint_step2", help="Step tag used in archive filename.")
    parser.add_argument("--archive-subdir", default="azcov_archive", help="Archive subdirectory under run-dir.")
    parser.add_argument("--output-subdir", default="img/azimuth", help="Output subdirectory under run-dir.")
    parser.add_argument("--depths", default="10,25,40", help="Comma-separated requested depths in km.")
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI.")
    parser.add_argument("--top-nodes", type=int, default=DEFAULT_TOP_NODES, help="Number of top-covered nodes to render as rose diagrams.")
    return parser.parse_args()


def _closest_depths(available: np.ndarray, requested: tuple[float, ...]) -> list[float]:
    chosen: list[float] = []
    for value in requested:
        depth = float(available[int(np.argmin(np.abs(available - value)))])
        if not any(abs(depth - existing) < 1.0e-6 for existing in chosen):
            chosen.append(depth)
    return chosen


def _top_node_indices(
    bins_3d_count: np.ndarray,
    bins_2d_count: np.ndarray,
    hit_count: np.ndarray,
    top_n: int,
) -> list[int]:
    valid = np.isfinite(bins_3d_count) & np.isfinite(bins_2d_count) & np.isfinite(hit_count)
    candidates = np.where(valid)[0]
    if candidates.size == 0:
        return []
    ranked = sorted(
        candidates.tolist(),
        key=lambda idx: (
            float(bins_3d_count[idx]),
            float(bins_2d_count[idx]),
            float(hit_count[idx]),
        ),
        reverse=True,
    )
    return ranked[: max(0, int(top_n))]


def _plot_top_node_roses(
    out_dir: Path,
    phase: str,
    step: str,
    lon: np.ndarray,
    lat: np.ndarray,
    dep: np.ndarray,
    hit_count: np.ndarray,
    bins_2d_count: np.ndarray,
    bins_3d_count: np.ndarray,
    bins_2d_occ: np.ndarray,
    bins_3d_occ: np.ndarray,
    dpi: int,
    top_n: int,
) -> tuple[Path | None, list[str]]:
    top_indices = _top_node_indices(bins_3d_count, bins_2d_count, hit_count, top_n)
    if not top_indices:
        return None, []

    ncols = min(3, len(top_indices))
    nrows = int(np.ceil(len(top_indices) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 4.7 * nrows),
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )
    axes_flat = axes.ravel()
    azimuth_bins = bins_2d_occ.shape[1]
    width = 2.0 * np.pi / float(azimuth_bins)
    theta = np.arange(azimuth_bins, dtype=float) * width
    summary_lines: list[str] = []

    for rank, (ax, node_idx) in enumerate(zip(axes_flat, top_indices), start=1):
        counts_3d = np.asarray(bins_3d_occ[node_idx], dtype=float).sum(axis=0)
        mask_2d = np.asarray(bins_2d_occ[node_idx], dtype=bool)
        max_count = max(float(np.nanmax(counts_3d)), 1.0)

        ax.bar(
            theta,
            counts_3d,
            width=width,
            align="edge",
            color="0.70",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.95,
        )
        centers = theta + 0.5 * width
        if mask_2d.any():
            ax.scatter(
                centers[mask_2d],
                np.full(mask_2d.sum(), max_count + 0.35, dtype=float),
                s=16,
                c="black",
                marker="o",
                zorder=3,
            )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(
            (
                f"Top node #{rank}\n"
                f"{lat[node_idx]:.2f}N, {lon[node_idx]:.2f}E, {dep[node_idx]:.0f} km\n"
                f"hits={int(hit_count[node_idx])}, az2={int(bins_2d_count[node_idx])}, az3={int(bins_3d_count[node_idx])}"
            ),
            fontsize=9,
        )
        ax.set_ylim(0.0, max_count + 0.7)
        yticks = np.arange(0, int(np.ceil(max_count)) + 1, max(1, int(np.ceil(max_count / 3.0))))
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(v)) for v in yticks], fontsize=7)
        ax.grid(alpha=0.25)

        summary_lines.append(
            "Top node #{rank}: lon={lon:.4f}, lat={lat:.4f}, dep={dep:.1f}, hits={hits}, az2={az2}, az3={az3}".format(
                rank=rank,
                lon=float(lon[node_idx]),
                lat=float(lat[node_idx]),
                dep=float(dep[node_idx]),
                hits=int(hit_count[node_idx]),
                az2=int(bins_2d_count[node_idx]),
                az3=int(bins_3d_count[node_idx]),
            )
        )

    for ax in axes_flat[len(top_indices):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Top-node azimuth roses | phase {phase} | {step}\nBars: occupied dip bins per azimuth | Dots: occupied 2D azimuth bins",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = out_dir / f"azimuth_top_nodes_rose_{phase}_{step}.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path, summary_lines


def _grid_slice(
    lon: np.ndarray,
    lat: np.ndarray,
    dep: np.ndarray,
    values: np.ndarray,
    target_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(dep) & (np.abs(dep - target_depth) < 1.0e-6)
    if not np.any(mask):
        return np.array([]), np.array([]), np.array([[]])

    lon_u = np.unique(lon[mask])
    lat_u = np.unique(lat[mask])
    grid = np.full((lat_u.size, lon_u.size), np.nan, dtype=float)
    lon_i = np.searchsorted(lon_u, lon[mask])
    lat_i = np.searchsorted(lat_u, lat[mask])
    grid[lat_i, lon_i] = values[mask]
    return lon_u, lat_u, grid


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if args.phase is None:
        args.phase = _infer_phase(run_dir)

    archive_dir = run_dir / args.archive_subdir
    summary_path = archive_dir / f"azcov_{args.phase}_{args.step}_node_coverage.h5"
    if not summary_path.exists():
        print(f"Skipping azimuth plots: node coverage archive not found: {summary_path}")
        return

    with h5py.File(summary_path, "r") as handle:
        nodes = handle["nodes"]
        lon = np.asarray(nodes["lon"][...], dtype=float)
        lat = np.asarray(nodes["lat"][...], dtype=float)
        dep = np.asarray(nodes["dep"][...], dtype=float)
        hit_count = np.asarray(nodes["hit_count"][...], dtype=float)
        bins_2d = np.asarray(nodes["azimuth_bin_count_2d"][...], dtype=float)
        bins_3d = np.asarray(nodes["azimuth_dip_bin_count_3d"][...], dtype=float)
        bins_2d_occ = np.asarray(nodes["azimuth_bins_2d"][...], dtype=np.uint8)
        bins_3d_occ = np.asarray(nodes["azimuth_dip_bins_3d"][...], dtype=np.uint8)

    if lon.size == 0:
        print(f"Skipping azimuth plots: node coverage archive is empty: {summary_path}")
        return

    requested = _parse_depths(args.depths)
    chosen_depths = _closest_depths(np.unique(dep[np.isfinite(dep)]), requested)
    if not chosen_depths:
        print(f"Skipping azimuth plots: no valid depth levels found in {summary_path}")
        return

    out_dir = run_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(chosen_depths), figsize=(5.3 * len(chosen_depths), 8.6), squeeze=False)

    vmax_2d = max(float(np.nanmax(bins_2d)), 1.0)
    vmax_3d = max(float(np.nanmax(bins_3d)), 1.0)
    plotted = 0
    for col, depth_value in enumerate(chosen_depths):
        lon_u, lat_u, grid_2d = _grid_slice(lon, lat, dep, bins_2d, depth_value)
        _, _, grid_3d = _grid_slice(lon, lat, dep, bins_3d, depth_value)
        if grid_2d.size == 0 or grid_3d.size == 0:
            continue
        plotted += 1

        im0 = axes[0, col].imshow(
            grid_2d,
            origin="lower",
            extent=[float(lon_u[0]), float(lon_u[-1]), float(lat_u[0]), float(lat_u[-1])],
            cmap=CMAP,
            vmin=0.0,
            vmax=vmax_2d,
            aspect="auto",
        )
        axes[0, col].set_title(f"2D azimuth bins | depth {depth_value:g} km")
        axes[0, col].set_xlabel("Longitude")
        axes[0, col].set_ylabel("Latitude")

        im1 = axes[1, col].imshow(
            grid_3d,
            origin="lower",
            extent=[float(lon_u[0]), float(lon_u[-1]), float(lat_u[0]), float(lat_u[-1])],
            cmap=CMAP,
            vmin=0.0,
            vmax=vmax_3d,
            aspect="auto",
        )
        axes[1, col].set_title(f"3D azimuth-dip bins | depth {depth_value:g} km")
        axes[1, col].set_xlabel("Longitude")
        axes[1, col].set_ylabel("Latitude")

    if plotted == 0:
        plt.close(fig)
        print(f"Skipping azimuth plots: no slice grids could be reconstructed from {summary_path}")
        return

    cbar0 = fig.colorbar(im0, ax=axes[0, :], fraction=0.025, pad=0.02)
    cbar0.set_label("Occupied 2D azimuth bins")
    cbar1 = fig.colorbar(im1, ax=axes[1, :], fraction=0.025, pad=0.02)
    cbar1.set_label("Occupied 3D azimuth-dip bins")

    fig.suptitle(f"Azimuthal coverage summary | phase {args.phase} | {args.step}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / f"azimuth_coverage_summary_{args.phase}_{args.step}.png"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    rose_path, rose_lines = _plot_top_node_roses(
        out_dir,
        args.phase,
        args.step,
        lon,
        lat,
        dep,
        hit_count,
        bins_2d,
        bins_3d,
        bins_2d_occ,
        bins_3d_occ,
        args.dpi,
        args.top_nodes,
    )

    summary_txt = out_dir / f"azimuth_coverage_summary_{args.phase}_{args.step}.txt"
    summary_txt.write_text(
        "\n".join(
            [
                "Azimuth coverage summary",
                f"run_dir: {run_dir}",
                f"phase: {args.phase}",
                f"step: {args.step}",
                f"node_summary: {summary_path}",
                f"depths_plotted_km: {', '.join(f'{v:g}' for v in chosen_depths)}",
                f"max_2d_bins: {float(np.nanmax(bins_2d)):.0f}",
                f"max_3d_bins: {float(np.nanmax(bins_3d)):.0f}",
                f"mean_2d_bins: {float(np.nanmean(bins_2d)):.3f}",
                f"mean_3d_bins: {float(np.nanmean(bins_3d)):.3f}",
                f"top_node_rose_figure: {rose_path if rose_path is not None else 'not written'}",
                *rose_lines,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote azimuth summary figure: {out_path}")


if __name__ == "__main__":
    main()
