#!/usr/bin/env python3
"""Build a single region from CWB src_rec input (event/station-based bounds).

- Uses event+station footprint to define bounds (after phase filtering).
- Allows manual lat/lon overrides for the base bounds.
- Applies a configurable border around data-driven bounds.
- Snaps min/max bounds to 0.1 degrees (min=floor, max=ceil).
- Applies a 99% central-area filter to events.
- Includes S picks by default, requires all azimuth bins by default.
- Disables ray coverage by default.
"""

from __future__ import annotations

import argparse
import logging
import json
import shutil
import math
from pathlib import Path
from typing import Tuple

import pandas as pd

try:
    from pytomoatt.src_rec import SrcRec
except ModuleNotFoundError:
    SrcRec = None  # type: ignore[assignment]

import region_scanner_pipeline as rscan

LOGGER = logging.getLogger("region_scanner_cwb_square")


def _require_pytomoatt() -> None:
    if SrcRec is None:
        raise SystemExit(
            "ERROR: pytomoatt is required for region scanning. "
            "Install PyTomoATT into the active environment so `import pytomoatt` works."
        )


def _remap_cwb_weights(weights: pd.Series) -> pd.Series:
    """Map CWB weight codes so 0->1.0 and 4->0.2; accept legacy 0.25 steps."""
    series = pd.to_numeric(weights, errors="coerce")
    if series.notna().empty:
        return series
    values = series.dropna()
    if values.empty:
        return series
    rounded = values.round(6)
    if ((rounded >= 0) & (rounded <= 4) & (rounded == rounded.round())).all():
        mapped = 1.0 - 0.2 * rounded
        series.loc[values.index] = mapped
        return series

    legacy = {0.0, 0.25, 0.5, 0.75, 1.0}
    if rounded.isin(legacy).all():
        mapped = 0.2 + 0.8 * rounded
        series.loc[values.index] = mapped
        return series

    return series


def _write_phase_src_rec(src_rec_path: Path, phase: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sr = SrcRec.read(str(src_rec_path))
    if not hasattr(sr, "rec_points") or sr.rec_points.empty:
        return False
    rec = sr.rec_points.copy()
    if "phase" not in rec.columns:
        return False
    phase_mask = rec["phase"].astype(str).str.upper() == phase.upper()
    rec = rec.loc[phase_mask].copy()
    if rec.empty:
        return False
    src_idx = rec["src_index"].unique()
    sr.src_points = sr.src_points.loc[src_idx].copy()
    sr.rec_points = rec
    sr.update()
    sr.write(str(out_path))
    return True


def _compute_phase_metadata(
    sr_phase: SrcRec,
    region: rscan.RegionBounds,
    base_metadata: dict,
    args: argparse.Namespace,
    src_rec_path: Path,
) -> dict | None:
    arrivals = rscan.build_arrivals_from_src_rec(sr_phase)
    if arrivals.empty:
        return None
    arrivals["phase"] = arrivals["phase"].astype(str).str.upper()

    stations_region = (
        arrivals.loc[:, ["staname", "stla", "stlo"]]
        .dropna(subset=["stla", "stlo"])
        .drop_duplicates(subset=["staname"])
    )

    n_events = int(arrivals["event_id"].nunique())
    n_stations = int(stations_region["staname"].nunique())
    n_arrivals = int(arrivals.shape[0])
    n_p = int((arrivals["phase"] == "P").sum())
    n_s = int((arrivals["phase"] == "S").sum())

    event_bin_deg = args.event_coverage_bin_deg or args.region_step_deg
    event_coverage_pct = rscan.compute_event_coverage_percent(arrivals, region, event_bin_deg)
    center_lat, center_lon = region.center
    az_stats = rscan.compute_azimuth_bin_stats(
        stations_region["stla"].to_numpy(dtype=float, copy=False),
        stations_region["stlo"].to_numpy(dtype=float, copy=False),
        center_lat,
        center_lon,
        args.azimuth_bin_deg,
    )
    avg_rr_km = rscan._average_receiver_spacing(sr_phase)
    avg_rr_out = float(avg_rr_km) if math.isfinite(avg_rr_km) else None
    has_teleseismic_data, n_teleseismic_events, n_teleseismic_arrivals = rscan.compute_teleseismic_stats(arrivals, region)

    return {
        "region_id": base_metadata.get("region_id", 1),
        "phase": str(base_metadata.get("phase", "")).upper() or None,
        "region_bounds": base_metadata.get("region_bounds", region.to_dict()),
        "central_bounds": base_metadata.get("central_bounds"),
        "min_events_region": base_metadata.get("min_events_region", args.min_events_region),
        "central_area_fraction": base_metadata.get("central_area_fraction", args.central_area_fraction),
        "require_all_azimuth_bins": base_metadata.get("require_all_azimuth_bins", args.require_all_azimuth_bins),
        "n_events": n_events,
        "n_stations": n_stations,
        "n_arrivals": n_arrivals,
        "n_p_picks": n_p,
        "n_s_picks": n_s,
        "event_coverage_pct": event_coverage_pct,
        "event_coverage_bin_deg": event_bin_deg,
        "azimuth_bin_deg": args.azimuth_bin_deg,
        "avg_receiver_spacing_km": avg_rr_out,
        "teleseismic_mode": bool(getattr(args, "teleseismic", False)),
        "has_teleseismic_data": has_teleseismic_data,
        "n_teleseismic_events": n_teleseismic_events,
        "n_teleseismic_arrivals": n_teleseismic_arrivals,
        **az_stats,
        "src_rec": str(src_rec_path),
    }


def _write_phase_outputs(
    phase: str,
    phase_src_rec: Path,
    region_dir: Path,
    region: rscan.RegionBounds,
    base_metadata: dict,
    args: argparse.Namespace,
) -> None:
    phase_dir = region_dir / phase.upper()
    phase_dir.mkdir(parents=True, exist_ok=True)
    phase_src = phase_dir / "src_rec.dat"
    shutil.copy2(phase_src_rec, phase_src)

    sr_phase = SrcRec.read(str(phase_src))
    fig_dir = phase_dir / "figures"
    region_box = (region.lon_min, region.lon_max, region.lat_min, region.lat_max)
    rscan.isc_tools._generate_coverage_figures(sr_phase, fig_dir, region_box=region_box)
    rscan.isc_tools._plot_lateral_depth_connections(sr_phase, fig_dir / "source_receiver_connections.png")
    rscan.isc_tools._plot_longitudinal_depth_connections(sr_phase, fig_dir / "source_receiver_connections_lon.png")

    if args.ray_coverage:
        rc_dir = phase_dir / args.rc_output_dir
        rscan.run_ray_coverage(phase_src, rc_dir, args, region)

    phase_metadata = dict(base_metadata)
    phase_metadata["phase"] = phase.upper()
    computed = _compute_phase_metadata(sr_phase, region, phase_metadata, args, phase_src)
    if computed:
        rscan.write_region_metadata(phase_dir / "region_metadata.json", computed)


def _round_down_tenth(value: float) -> float:
    return math.floor(value * 10.0) / 10.0


def _round_up_tenth(value: float) -> float:
    return math.ceil(value * 10.0) / 10.0


def _get_data_bounds(arrivals: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Return (lat_min, lat_max, lon_min, lon_max) from event+station inputs."""
    ev = arrivals.loc[:, ["evla", "evlo"]].rename(columns={"evla": "lat", "evlo": "lon"})
    st = arrivals.loc[:, ["stla", "stlo"]].rename(columns={"stla": "lat", "stlo": "lon"})
    coords = pd.concat([ev, st], ignore_index=True).apply(pd.to_numeric, errors="coerce").dropna()
    if coords.empty:
        raise ValueError("No valid event/station coordinates found in arrivals.")

    lat_min = float(coords["lat"].min())
    lat_max = float(coords["lat"].max())
    lon_min = float(coords["lon"].min())
    lon_max = float(coords["lon"].max())
    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError("Event/station bounds are degenerate; cannot build a region.")
    return lat_min, lat_max, lon_min, lon_max


def _compute_depth_bounds(arrivals: pd.DataFrame, depth_range: Tuple[float, float] | None) -> Tuple[float, float]:
    if depth_range:
        depth_min, depth_max = sorted(depth_range)
        return float(depth_min), float(depth_max)
    depth_series = pd.to_numeric(arrivals["evdp"], errors="coerce")
    if not depth_series.notna().any():
        return 0.0, 0.0
    return float(depth_series.min(skipna=True)), float(depth_series.max(skipna=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src-rec",
        default="CWBcatalog/src_rec_CWB.dat",
        help="Input src_rec.dat (default: CWBcatalog/src_rec_CWB.dat).",
    )
    parser.add_argument(
        "--output-root",
        default="tomo_regions_cwb_square",
        help="Output directory for the region results.",
    )
    parser.add_argument("--phases", default="P", help="Comma-separated list of phases to keep.")
    parser.add_argument(
        "--include-s",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include S picks alongside P picks (default true).",
    )
    parser.add_argument(
        "--phases-anycase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize phases to uppercase (default true).",
    )
    parser.add_argument(
        "--use-dist-col",
        action="store_true",
        help="Use dist_deg column in src_rec (if present).",
    )
    parser.add_argument(
        "--central-area-fraction",
        type=float,
        default=0.90,
        help="Central area fraction for events (0<frac<=1).",
    )
    parser.add_argument(
        "--lat-range",
        type=float,
        nargs=2,
        metavar=("MIN_LAT", "MAX_LAT"),
        help="Override latitude bounds for region construction (degrees).",
    )
    parser.add_argument(
        "--lon-range",
        type=float,
        nargs=2,
        metavar=("MIN_LON", "MAX_LON"),
        help="Override longitude bounds for region construction (degrees).",
    )
    parser.add_argument(
        "--region-border-deg",
        type=float,
        default=0.2,
        help="Border added around auto-derived lat/lon bounds (degrees, default 0.2).",
    )
    parser.add_argument(
        "--decluster-block-km",
        type=float,
        default=0.1,
        help="Block size for declustering (km); set <=0 to disable.",
    )
    parser.add_argument("--min-p-picks", type=int, default=4, help="Minimum number of P picks per event.")
    parser.add_argument("--min-s-picks", type=int, default=4, help="Minimum number of S picks per event.")
    parser.add_argument(
        "--min-events-region",
        type=int,
        default=0,
        help="Minimum number of events required for the region (0 disables).",
    )
    parser.add_argument("--azimuth-bin-deg", type=float, default=30.0, help="Azimuth bin size (degrees).")
    parser.add_argument("--min-stations-per-bin", type=int, default=1, help="Min stations per azimuth bin.")
    parser.add_argument(
        "--require-all-azimuth-bins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require each azimuth bin to meet the minimum station count (default true).",
    )
    parser.add_argument(
        "--event-coverage-bin-deg",
        type=float,
        help="Bin size (degrees) for event coverage percentage.",
    )
    parser.add_argument(
        "--region-step-deg",
        type=float,
        default=0.5,
        help="Fallback bin size for event coverage when --event-coverage-bin-deg is unset.",
    )
    parser.add_argument(
        "--depth-range",
        type=float,
        nargs=2,
        metavar=("MIN_DEPTH", "MAX_DEPTH"),
        help="Optional depth bounds override (km).",
    )
    parser.add_argument(
        "--teleseismic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep out-of-region sources/events while removing out-of-region "
            "receivers."
        ),
    )
    parser.add_argument(
        "--ray-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable ray coverage outputs (default false).",
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
    args = build_parser().parse_args()
    _require_pytomoatt()
    args.phases = rscan._normalize_phase_list(args.phases, args.include_s)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    src_rec_path = Path(args.src_rec).expanduser()
    if not src_rec_path.exists():
        raise FileNotFoundError(f"src_rec file not found: {src_rec_path}")

    sr_all = SrcRec.read(str(src_rec_path))
    if args.phases_anycase and hasattr(sr_all, "rec_points") and "phase" in sr_all.rec_points.columns:
        sr_all.rec_points["phase"] = sr_all.rec_points["phase"].astype(str).str.upper()

    arrivals = rscan.build_arrivals_from_src_rec(sr_all)
    if "weight" in arrivals.columns:
        arrivals["weight"] = _remap_cwb_weights(arrivals["weight"])
    phase_list = [ph.strip().upper() for ph in args.phases.split(",") if ph.strip()]
    if phase_list:
        arrivals = arrivals[arrivals["phase"].astype(str).str.upper().isin(phase_list)].copy()
    if arrivals.empty:
        raise SystemExit("No arrivals remain after phase filtering.")

    data_lat_min, data_lat_max, data_lon_min, data_lon_max = _get_data_bounds(arrivals)
    auto_lat_min = data_lat_min - args.region_border_deg
    auto_lat_max = data_lat_max + args.region_border_deg
    auto_lon_min = data_lon_min - args.region_border_deg
    auto_lon_max = data_lon_max + args.region_border_deg

    if args.lat_range:
        lat_min_raw, lat_max_raw = sorted(args.lat_range)
    else:
        lat_min_raw, lat_max_raw = auto_lat_min, auto_lat_max
    if args.lon_range:
        lon_min_raw, lon_max_raw = sorted(args.lon_range)
    else:
        lon_min_raw, lon_max_raw = auto_lon_min, auto_lon_max

    lat_min = _round_down_tenth(lat_min_raw)
    lat_max = _round_up_tenth(lat_max_raw)
    lon_min = _round_down_tenth(lon_min_raw)
    lon_max = _round_up_tenth(lon_max_raw)

    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError("Invalid lat/lon bounds (min must be < max).")

    LOGGER.info(
        (
            "Data bounds lat %.4f-%.4f, lon %.4f-%.4f; "
            "buffered bounds lat %.4f-%.4f, lon %.4f-%.4f; "
            "final rounded bounds lat %.4f-%.4f, lon %.4f-%.4f"
        ),
        data_lat_min,
        data_lat_max,
        data_lon_min,
        data_lon_max,
        lat_min_raw,
        lat_max_raw,
        lon_min_raw,
        lon_max_raw,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    )

    depth_min, depth_max = _compute_depth_bounds(arrivals, args.depth_range)
    region = rscan.RegionBounds(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        depth_min=depth_min,
        depth_max=depth_max,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = rscan.process_region(1, region, arrivals, args, output_root)
    summaries = [summary] if summary else []
    rscan.write_region_summary_csv(output_root / "region_summary.csv", summaries)
    rscan.write_region_rank_csv(output_root / "region_rankings.csv", summaries)

    if not summaries:
        LOGGER.warning("No region outputs written (region skipped by filters).")
    else:
        region_dir = output_root / "region_001"
        src_rec_phase = region_dir / "src_rec.dat"
        base_metadata_path = region_dir / "region_metadata.json"
        base_metadata = {}
        if base_metadata_path.exists():
            base_metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
        region_bounds = base_metadata.get("region_bounds")
        if region_bounds:
            region = rscan.RegionBounds(**region_bounds)

        if src_rec_phase.exists():
            phase_outputs = {
                "P": region_dir / "src_rec_P.dat",
                "S": region_dir / "src_rec_S.dat",
            }
            for phase, phase_path in phase_outputs.items():
                wrote = _write_phase_src_rec(src_rec_phase, phase, phase_path)
                if not wrote:
                    LOGGER.warning("No %s arrivals found; %s not written.", phase, phase_path.name)
                    continue
                _write_phase_outputs(phase, phase_path, region_dir, region, base_metadata, args)
        LOGGER.info("Done. Processed 1 region in %s", output_root)


if __name__ == "__main__":
    main()
