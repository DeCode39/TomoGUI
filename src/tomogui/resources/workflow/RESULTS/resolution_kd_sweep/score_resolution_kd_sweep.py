#!/usr/bin/env python3
"""Score the resolution/Kdensity_coe sweep after inversion outputs exist."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DEPTHS = (5, 10, 15, 20, 25, 30, 35, 40, 45)

for _parent in Path(__file__).resolve().parents:
    if (_parent / "resolution_window.py").exists():
        sys.path.insert(0, str(_parent))
        break
else:  # pragma: no cover - runtime configuration failure
    raise RuntimeError("Could not find resolution_window.py in this sweep tree.")

from resolution_window import windowed_resolution, windowed_sum


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_model(path: Path, *, coords_from: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            arrays[key] = handle[key][()]
    if coords_from is not None:
        for key in ("dep", "lat", "lon"):
            arrays.setdefault(key, coords_from[key])
    return arrays


def weighted_percent(mask: np.ndarray, weight: np.ndarray, valid: np.ndarray) -> float:
    use = valid & np.isfinite(weight)
    denom = float(np.nansum(weight[use]))
    if denom <= 0.0:
        return float("nan")
    return 100.0 * float(np.nansum(weight[use] * mask[use])) / denom


def artifact_ratio(rec_energy: np.ndarray, true_energy: np.ndarray, valid: np.ndarray) -> float:
    outside = (~valid) & np.isfinite(rec_energy)
    inside = valid & np.isfinite(true_energy)
    ref = float(np.nanmean(true_energy[inside])) if np.any(inside) else 0.0
    if ref <= 0.0 or not np.any(outside):
        return 0.0
    return float(np.nanmean(rec_energy[outside])) / ref


def compute_scores(row: dict[str, str], args: argparse.Namespace) -> dict[str, object] | None:
    initial_path = ROOT / row["initial_model"]
    true_path = ROOT / row["true_model"]
    recovered_path = ROOT / row["recovered_model"]
    if not recovered_path.exists():
        print(f"Skipping missing final model: {recovered_path}")
        return None

    initial = read_model(initial_path)
    true_model = read_model(true_path, coords_from=initial)
    recovered = read_model(recovered_path, coords_from=initial)
    dep = initial["dep"]
    lat = initial["lat"]
    lon = initial["lon"]

    dv_true = (true_model["vel"] - initial["vel"]) / initial["vel"]
    dv_rec = (recovered["vel"] - initial["vel"]) / initial["vel"]
    xi_true = true_model["xi"]
    eta_true = true_model["eta"]
    xi_rec = recovered["xi"]
    eta_rec = recovered["eta"]

    rv = windowed_resolution(
        (dv_rec - dv_true) ** 2,
        dv_rec**2 + dv_true**2,
        dep,
        lat,
        lon,
        window_depth_km=args.window_depth_km,
        window_lat_deg=args.window_lat_deg,
        window_lon_deg=args.window_lon_deg,
    )
    rani = windowed_resolution(
        (xi_rec - xi_true) ** 2 + (eta_rec - eta_true) ** 2,
        xi_rec**2 + eta_rec**2 + xi_true**2 + eta_true**2,
        dep,
        lat,
        lon,
        window_depth_km=args.window_depth_km,
        window_lat_deg=args.window_lat_deg,
        window_lon_deg=args.window_lon_deg,
    )

    ev = windowed_sum(dv_true**2, dep, lat, lon, window_depth_km=args.window_depth_km, window_lat_deg=args.window_lat_deg, window_lon_deg=args.window_lon_deg)
    eani = windowed_sum(xi_true**2 + eta_true**2, dep, lat, lon, window_depth_km=args.window_depth_km, window_lat_deg=args.window_lat_deg, window_lon_deg=args.window_lon_deg)
    valid_v = ev > (np.nanmax(ev) * args.energy_floor)
    valid_ani = eani > (np.nanmax(eani) * args.energy_floor)
    valid_both = valid_v & valid_ani

    vpass = rv >= args.threshold
    anipass = rani >= args.threshold
    bothpass = vpass & anipass
    eboth = np.sqrt(np.maximum(ev * eani, 0.0))

    pv = weighted_percent(vpass, ev, valid_v)
    pani = weighted_percent(anipass, eani, valid_ani)
    pboth = weighted_percent(bothpass, eboth, valid_both)
    art_v = artifact_ratio(dv_rec**2, dv_true**2, valid_v)
    art_ani = artifact_ratio(xi_rec**2 + eta_rec**2, xi_true**2 + eta_true**2, valid_ani)
    art = 0.5 * (art_v + art_ani)
    artifact_penalty = args.artifact_weight * art
    score = 0.50 * pboth + 0.25 * pv + 0.25 * pani - artifact_penalty

    out_dir = ROOT / row["run_dir"] / "resolution_mask"
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_dir / "scored_resolution_masks.h5", "w") as handle:
        for key, value in {"dep": dep, "lat": lat, "lon": lon, "R_v": rv, "R_ani": rani}.items():
            handle.create_dataset(key, data=value, compression="gzip")
        handle.create_dataset("velocity_pass", data=vpass.astype(np.uint8), compression="gzip")
        handle.create_dataset("anisotropy_pass", data=anipass.astype(np.uint8), compression="gzip")
        handle.create_dataset("combined_pass", data=bothpass.astype(np.uint8), compression="gzip")
        handle.create_dataset("valid_velocity", data=valid_v.astype(np.uint8), compression="gzip")
        handle.create_dataset("valid_anisotropy", data=valid_ani.astype(np.uint8), compression="gzip")
        handle.attrs["threshold"] = args.threshold
        handle.attrs["energy_floor"] = args.energy_floor
        handle.attrs["score"] = score

    return {
        "run_tag": row["run_tag"],
        "wave_tag": row["wave_tag"],
        "kdensity_coe": float(row["kdensity_coe"]),
        "wave_weight": float(row["wave_weight"]),
        "velocity_period_deg": float(row["velocity_period_deg"]),
        "anisotropy_period_deg": float(row["anisotropy_period_deg"]),
        "depth_period_km": float(row["depth_period_km"]),
        "P_v": pv,
        "P_ani": pani,
        "P_both": pboth,
        "artifact_ratio": art,
        "artifact_penalty": artifact_penalty,
        "score": score,
        "valid_velocity_pct": 100.0 * float(np.mean(valid_v)),
        "valid_anisotropy_pct": 100.0 * float(np.mean(valid_ani)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["kdensity_coe"])].append(row)

    out: list[dict[str, object]] = []
    for kdensity, vals in sorted(grouped.items()):
        total_weight = sum(float(v["wave_weight"]) for v in vals)
        if total_weight <= 0.0:
            continue
        out.append(
            {
                "kdensity_coe": kdensity,
                "n_wave_cases": len(vals),
                "weighted_score": sum(float(v["score"]) * float(v["wave_weight"]) for v in vals) / total_weight,
                "weighted_P_v": sum(float(v["P_v"]) * float(v["wave_weight"]) for v in vals) / total_weight,
                "weighted_P_ani": sum(float(v["P_ani"]) * float(v["wave_weight"]) for v in vals) / total_weight,
                "weighted_P_both": sum(float(v["P_both"]) * float(v["wave_weight"]) for v in vals) / total_weight,
                "weighted_artifact_ratio": sum(float(v["artifact_ratio"]) * float(v["wave_weight"]) for v in vals) / total_weight,
            }
        )
    return out


def nearest_depth_indices(dep: np.ndarray, depths: tuple[float, ...]) -> list[int]:
    out: list[int] = []
    for depth in depths:
        idx = int(np.nanargmin(np.abs(dep - depth)))
        if idx not in out:
            out.append(idx)
    return out


def plot_heatmaps(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    waves = sorted({str(r["wave_tag"]) for r in rows})
    kds = sorted({float(r["kdensity_coe"]) for r in rows})
    metrics = [("P_v", "Velocity pass"), ("P_ani", "Anisotropy pass"), ("P_both", "Combined pass"), ("score", "Score")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.6), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    lookup = {(str(r["wave_tag"]), float(r["kdensity_coe"])): r for r in rows}
    for ax, (key, title) in zip(axes, metrics):
        matrix = np.full((len(waves), len(kds)), np.nan)
        for iy, wave in enumerate(waves):
            for ix, kdensity in enumerate(kds):
                row = lookup.get((wave, kdensity))
                if row is not None:
                    matrix[iy, ix] = float(row[key])
        image = ax.imshow(matrix, vmin=0.0 if key != "score" else None, vmax=100.0 if key != "score" else None, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(kds)), [f"{k:g}" for k in kds])
        ax.set_yticks(range(len(waves)), waves)
        ax.set_xlabel("Kdensity_coe")
        for iy in range(len(waves)):
            for ix in range(len(kds)):
                value = matrix[iy, ix]
                if np.isfinite(value):
                    ax.text(ix, iy, f"{value:.1f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.78)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def suggest_next_values(kds: list[float], best: float) -> list[float]:
    if not kds:
        return []
    diffs = [abs(best - kdensity) for kdensity in kds if abs(best - kdensity) > 1.0e-8]
    nearest = min(diffs) if diffs else 0.3
    if nearest >= 0.25:
        step = 0.1
    elif nearest >= 0.09:
        step = 0.05
    else:
        step = max(0.025, nearest * 0.5)
    candidates = [round(best - step, 6), round(best + step, 6)]
    out = []
    for value in candidates:
        if value < 0.0 or value > 0.95:
            continue
        if not any(abs(value - existing) < 1.0e-6 for existing in kds):
            out.append(value)
    return out


def write_best_report(path: Path, agg_rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not agg_rows:
        path.write_text("No completed runs were available to score.\n", encoding="utf-8")
        return
    sorted_rows = sorted(agg_rows, key=lambda r: (-float(r["weighted_score"]), float(r["kdensity_coe"])))
    best = sorted_rows[0]
    kds = sorted(float(r["kdensity_coe"]) for r in agg_rows)
    next_values = suggest_next_values(kds, float(best["kdensity_coe"]))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Resolution / Kdensity_coe sweep best-coefficient report\n")
        handle.write("========================================================\n\n")
        handle.write(f"threshold: {args.threshold}\n")
        handle.write(f"energy_floor: {args.energy_floor}\n")
        handle.write(f"artifact_weight: {args.artifact_weight}\n\n")
        handle.write("Ranked coefficients:\n")
        for row in sorted_rows:
            handle.write(
                f"Kdensity_coe={float(row['kdensity_coe']):g} "
                f"score={float(row['weighted_score']):.3f} "
                f"P_both={float(row['weighted_P_both']):.3f} "
                f"P_v={float(row['weighted_P_v']):.3f} "
                f"P_ani={float(row['weighted_P_ani']):.3f} "
                f"artifact={float(row['weighted_artifact_ratio']):.5f} "
                f"n_wave_cases={int(row['n_wave_cases'])}\n"
            )
        handle.write("\nRecommended current best:\n")
        handle.write(f"Kdensity_coe = {float(best['kdensity_coe']):g}\n")
        if next_values:
            handle.write("\nSuggested refinement values:\n")
            handle.write(",".join(f"{v:g}" for v in next_values) + "\n")
            handle.write("\nPrepare them with:\n")
            handle.write(f"python prepare_resolution_kd_sweep.py --kd-values {','.join(f'{v:g}' for v in next_values)}\n")
        else:
            handle.write("\nNo new refinement values suggested inside [0, 0.95].\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest_inversion.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "summaries")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--energy-floor", type=float, default=0.02, help="Fraction of max local true energy required for scored cells.")
    parser.add_argument("--artifact-weight", type=float, default=10.0, help="Penalty multiplier applied to outside-signal recovered energy ratio.")
    parser.add_argument("--window-lon-deg", type=float, default=0.5)
    parser.add_argument("--window-lat-deg", type=float, default=0.5)
    parser.add_argument("--window-depth-km", type=float, default=10.0)
    args = parser.parse_args()

    rows = read_csv(args.manifest)
    scored = []
    for row in rows:
        result = compute_scores(row, args)
        if result is not None:
            scored.append(result)

    agg_rows = aggregate(scored)
    write_csv(args.output_dir / "kd_sweep_scores.csv", scored)
    write_csv(args.output_dir / "kd_sweep_aggregate.csv", agg_rows)
    plot_heatmaps(args.output_dir / "kd_sweep_heatmaps.png", scored)
    write_best_report(args.output_dir / "kd_sweep_best.txt", agg_rows, args)

    print(f"Scored completed runs: {len(scored)}")
    print(f"Wrote: {args.output_dir / 'kd_sweep_scores.csv'}")
    print(f"Wrote: {args.output_dir / 'kd_sweep_aggregate.csv'}")
    print(f"Wrote: {args.output_dir / 'kd_sweep_best.txt'}")


if __name__ == "__main__":
    main()
