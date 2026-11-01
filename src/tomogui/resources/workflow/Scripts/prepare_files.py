#!/usr/bin/env python3
"""Prepare checkerboard/inversion folders from region metadata using TEST templates.

This script creates a timestamped folder under RESULTS with:
  - dataset/ containing phase-specific src_rec files
  - checkerboard_P, checkerboard_S
  - inversion_P, inversion_S
  - inversion_VpVs (when both P and S are present)

Folders are copied from RESULTS/TEST templates, then domain and runtime
settings are updated for the selected region.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
RESULTS_DIR = ROOT_DIR / "RESULTS"
TEST_DIR = RESULTS_DIR / "TEST"

TEMPLATE_CHECKERBOARD = {
    "P": TEST_DIR / "checkerboard_P",
    "S": TEST_DIR / "checkerboard_S",
}
TEMPLATE_INVERSION = {
    "P": TEST_DIR / "inversion_P",
    "S": TEST_DIR / "inversion_S",
}
TEMPLATE_VPVS = TEST_DIR / "inversion_VpVs"
VPVS_SCRIPT = BASE_DIR / "generate_vpvs_folder.py"

DEFAULT_PREFIX = "datatest"
KM_PER_DEG = 25.0
DEG_TO_KM = 111.32
LATERAL_BUFFER_DEG = 0.2

REQUIRED_TEMPLATE_FILES = (
    "prepare_input_files.py",
    "plot_output.py",
    "run_this_example_slurm.sh",
)
OPTIONAL_TEMPLATE_FILES = (
    "assign_gaussian_noise.py",
    "run_this_example.sh",
)
INVERSION_SUPPORT_SCRIPTS = (
    "plot_qc.py",
    "build_azcov_archive.py",
    "plot_azimuthal_coverage.py",
    "plot_single_ray_trace.py",
    "run_postprocess.py",
)
RUN_ROOT_SUPPORT_FILES = (
    "clean_run_results.sh",
)


def _fmt(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _fmt_list(values: Sequence[float]) -> str:
    return "[" + ",".join(_fmt(v) for v in values) + "]"


def _compute_base_inv_grid(n_rtp: Sequence[int]) -> list[int]:
    base_grid: list[int] = []
    for value in n_rtp:
        target = int(round(0.75 * float(value)))
        target = max(1, min(int(value), target))
        base_grid.append(target)
    return base_grid


def _replace_regex(
    text: str,
    pattern: str,
    replacement: str,
    desc: str,
    count: int = 0,
    allow_missing: bool = False,
) -> str:
    new_text, num = re.subn(pattern, replacement, text, count=count, flags=re.MULTILINE)
    if num == 0 and not allow_missing:
        raise RuntimeError(f"Could not update {desc} (pattern {pattern!r} not found).")
    return new_text


def _parse_phases(value: str) -> list[str]:
    phases = [token.strip().upper() for token in value.split(",") if token.strip()]
    if not phases:
        phases = ["P"]
    return list(dict.fromkeys(phases))


def _resolve_metadata_path(metadata_path: Path, phase: str) -> Path:
    phase_path = metadata_path.parent / phase.upper() / "region_metadata.json"
    return phase_path if phase_path.exists() else metadata_path


def _resolve_src_rec(src_rec: str | Path, metadata_path: Path) -> Path:
    src_rec_path = Path(src_rec)
    if not src_rec_path.is_absolute():
        src_rec_path = (metadata_path.parent / src_rec_path).resolve()
    return src_rec_path


def _resolve_phase_src_rec(base_src: Path, metadata_path: Path, phase: str, strict: bool) -> Path:
    phase = phase.upper()
    phase_parent = metadata_path.parent
    if phase_parent.name.upper() == phase:
        candidates = [
            phase_parent / "src_rec.dat",
            phase_parent / f"src_rec_{phase}.dat",
        ]
    else:
        candidates = [
            phase_parent / phase / "src_rec.dat",
            phase_parent / f"src_rec_{phase}.dat",
        ]

    if base_src.exists():
        base_text = base_src.as_posix().upper()
        if f"/{phase}/" in base_text or base_src.name.upper().endswith(f"_{phase}.DAT"):
            return base_src

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if strict:
        expected = ", ".join(str(p) for p in candidates)
        raise SystemExit(f"Phase-specific src_rec for {phase} not found. Expected one of: {expected}")

    if base_src.exists():
        print(f"Warning: phase-specific src_rec for {phase} not found; using {base_src}")
        return base_src

    raise SystemExit(f"src_rec file not found: {base_src}")


def _load_region_metadata(
    path: Path,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], Path, float | None, bool]:
    data = json.loads(path.read_text(encoding="utf-8"))
    bounds = data.get("region_bounds") or {}
    lat_range = (float(bounds["lat_min"]), float(bounds["lat_max"]))
    lon_range = (float(bounds["lon_min"]), float(bounds["lon_max"]))
    dep_range = (float(bounds["depth_min"]), float(bounds["depth_max"]))
    src_rec = data.get("src_rec")
    if not src_rec:
        raise SystemExit("region_metadata.json missing 'src_rec' path.")
    src_path = _resolve_src_rec(src_rec, path)
    avg_spacing = data.get("avg_receiver_spacing_km")
    try:
        avg_spacing_val = float(avg_spacing) if avg_spacing is not None else None
    except (TypeError, ValueError):
        avg_spacing_val = None
    if avg_spacing_val is not None and (not math.isfinite(avg_spacing_val) or avg_spacing_val <= 0.0):
        avg_spacing_val = None
    has_teleseismic_data = bool(data.get("has_teleseismic_data", False))
    return lat_range, lon_range, dep_range, src_path, avg_spacing_val, has_teleseismic_data


def _compute_n_rtp(
    dep_range: tuple[float, float],
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    avg_receiver_spacing_km: float | None,
) -> list[int]:
    dep_span = max(dep_range[1] - dep_range[0], 0.0)
    lat_span = max(lat_range[1] - lat_range[0], 0.0)
    lon_span = max(lon_range[1] - lon_range[0], 0.0)
    n_dep = max(1, int(round(dep_span)) + 1)
    if avg_receiver_spacing_km is not None and avg_receiver_spacing_km > 0.0:
        lat_km = lat_span * DEG_TO_KM
        lon_km = lon_span * DEG_TO_KM
        n_lat = max(1, int(math.floor(lat_km / avg_receiver_spacing_km)))
        n_lon = max(1, int(math.floor(lon_km / avg_receiver_spacing_km)))
    else:
        n_lat = max(1, int(round(lat_span * KM_PER_DEG)) + 1)
        n_lon = max(1, int(round(lon_span * KM_PER_DEG)) + 1)
    return [n_dep, n_lat, n_lon]


def _expand_lateral_range(range_pair: tuple[float, float], buffer_deg: float) -> tuple[float, float]:
    low, high = sorted(range_pair)
    return (low - buffer_deg, high + buffer_deg)


def _count_unique_stations(src_rec: Path) -> int:
    stations: set[str] = set()
    for line in src_rec.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        if line[0].isspace():
            parts = line.split()
            if len(parts) >= 3:
                stations.add(parts[2])
    return len(stations)


def _receiver_depth_range(src_rec: Path) -> tuple[float, float] | None:
    depths: list[float] = []
    for line in src_rec.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        if line[0].isspace():
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                depths.append(float(parts[5]))
            except ValueError:
                continue
    if not depths:
        return None
    max_abs = max(abs(value) for value in depths)
    assume_meters = max_abs > 50.0
    if assume_meters:
        depths_km = [-(value / 1000.0) for value in depths]
    else:
        depths_km = depths
    return min(depths_km), max(depths_km)


def _write_src_rec(
    src_path: Path,
    dest_path: Path,
    include_weights: bool,
    fixed_weight: float | None,
) -> None:
    lines = src_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    output: list[str] = []
    for line in lines:
        if not line.strip():
            output.append(line)
            continue
        if line[0].isspace():
            prefix = line[: len(line) - len(line.lstrip())]
            parts = line.split()
            if include_weights:
                if fixed_weight is not None and len(parts) >= 1:
                    parts[-1] = f"{fixed_weight:.4f}".rstrip("0").rstrip(".")
            else:
                if len(parts) >= 9:
                    parts = parts[:-1]
            output.append(prefix + " ".join(parts))
        else:
            output.append(line)
    dest_path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _update_yaml_domain(
    path: Path,
    dep_range: tuple[float, float],
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    n_rtp: Sequence[int],
) -> None:
    text = path.read_text(encoding="utf-8")
    dep_list = _fmt_list(dep_range)
    lat_list = _fmt_list(lat_range)
    lon_list = _fmt_list(lon_range)
    n_rtp_list = f"[{n_rtp[0]}, {n_rtp[1]}, {n_rtp[2]}]"
    n_base_inv = _compute_base_inv_grid(n_rtp)
    n_base_inv_list = f"[{n_base_inv[0]}, {n_base_inv[1]}, {n_base_inv[2]}]"

    text = _replace_regex(text, r"(^\s*min_max_dep:\s*)\[[^\]]+\]", fr"\g<1>{dep_list}", "min_max_dep", allow_missing=True)
    text = _replace_regex(text, r"(^\s*min_max_lat:\s*)\[[^\]]+\]", fr"\g<1>{lat_list}", "min_max_lat", allow_missing=True)
    text = _replace_regex(text, r"(^\s*min_max_lon:\s*)\[[^\]]+\]", fr"\g<1>{lon_list}", "min_max_lon", allow_missing=True)
    text = _replace_regex(text, r"(^\s*n_rtp:\s*)\[[^\]]+\]", fr"\g<1>{n_rtp_list}", "n_rtp", allow_missing=True)

    text = _replace_regex(text, r"(^\s*min_max_dep_inv:\s*)\[[^\]]+\]", fr"\g<1>{dep_list}", "min_max_dep_inv", allow_missing=True)
    text = _replace_regex(text, r"(^\s*min_max_lat_inv:\s*)\[[^\]]+\]", fr"\g<1>{lat_list}", "min_max_lat_inv", allow_missing=True)
    text = _replace_regex(text, r"(^\s*min_max_lon_inv:\s*)\[[^\]]+\]", fr"\g<1>{lon_list}", "min_max_lon_inv", allow_missing=True)
    text = _replace_regex(
        text,
        r"(^\s*n_inv_dep_lat_lon:\s*)\[[^\]]+\]",
        fr"\g<1>{n_base_inv_list}",
        "n_inv_dep_lat_lon",
        allow_missing=True,
    )
    text = _replace_regex(text, r"(^\s*min_max_dep_inv_ani:\s*)\[[^\]]+\]", fr"\g<1>{dep_list}", "min_max_dep_inv_ani", allow_missing=True)
    text = _replace_regex(text, r"(^\s*min_max_lat_inv_ani:\s*)\[[^\]]+\]", fr"\g<1>{lat_list}", "min_max_lat_inv_ani", allow_missing=True)
    text = _replace_regex(text, r"(^\s*min_max_lon_inv_ani:\s*)\[[^\]]+\]", fr"\g<1>{lon_list}", "min_max_lon_inv_ani", allow_missing=True)
    text = _replace_regex(
        text,
        r"(^\s*uniform_inv_grid_dep:\s*)(true|false)",
        r"\g<1>true",
        "uniform_inv_grid_dep",
        allow_missing=True,
    )

    text = _replace_regex(
        text,
        r"(model_(?:init|ckb)_N)\d+_\d+_\d+",
        fr"\g<1>{n_rtp[0]}_{n_rtp[1]}_{n_rtp[2]}",
        "model path sizes",
        allow_missing=True,
    )
    path.write_text(text, encoding="utf-8")


def _update_n_sims(path: Path, n_sims: int) -> None:
    text = path.read_text(encoding="utf-8")
    text = _replace_regex(text, r"(^\s*n_sims:\s*)\d+", fr"\g<1>{n_sims}", "n_sims", allow_missing=True)
    path.write_text(text, encoding="utf-8")


def _update_swap_src_rec(path: Path, value: bool) -> None:
    text = path.read_text(encoding="utf-8")
    text = _replace_regex(
        text,
        r"(^\s*swap_src_rec:\s*)(true|false)",
        fr"\g<1>{str(value).lower()}",
        "swap_src_rec",
        allow_missing=True,
    )
    path.write_text(text, encoding="utf-8")


def _update_have_tele_data(path: Path, value: bool) -> None:
    text = path.read_text(encoding="utf-8")
    text = _replace_regex(
        text,
        r"(^\s*have_tele_data:\s*)(true|false)",
        fr"\g<1>{str(value).lower()}",
        "have_tele_data",
        allow_missing=True,
    )
    path.write_text(text, encoding="utf-8")


def _update_ignore_velocity_discontinuity(path: Path, value: bool = True) -> None:
    text = path.read_text(encoding="utf-8")
    flag = "true" if value else "false"
    line = (
        f"ignore_velocity_discontinuity: {flag} # Allow runs to continue if the input model "
        "has common SALUTE velocity jumps.\n"
    )
    if re.search(r"^\s*ignore_velocity_discontinuity:\s*(true|false)", text, flags=re.MULTILINE):
        text = _replace_regex(
            text,
            r"(^\s*ignore_velocity_discontinuity:\s*)(true|false)",
            fr"\g<1>{flag}",
            "ignore_velocity_discontinuity",
            allow_missing=True,
        )
    elif re.search(r"^\s*have_tele_data:", text, flags=re.MULTILINE):
        text = re.sub(
            r"(^\s*have_tele_data:.*\n)",
            rf"\g<1>{line}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text = re.sub(
            r"(^\s*run_mode:\s*\d+\s*(?:#.*)?\n)",
            rf"\g<1>{line}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    path.write_text(text, encoding="utf-8")


def _update_slurm_ntasks(path: Path, n_sims: int) -> None:
    text = path.read_text(encoding="utf-8")
    text = _replace_regex(text, r"(^#SBATCH\s+-n\s+)\d+", fr"\g<1>{n_sims}", "slurm ntasks", allow_missing=True)
    text = re.sub(r'(NTASKS="\$\{SLURM_NTASKS:-)\d+(\}")', fr"\g<1>{n_sims}\g<2>", text)
    path.write_text(text, encoding="utf-8")


def _update_output_flags(path: Path, enabled: bool) -> None:
    text = path.read_text(encoding="utf-8")
    flag = "true" if enabled else "false"
    for field in ("output_source_field", "output_kernel", "output_in_process", "output_in_process_data"):
        text = _replace_regex(
            text,
            rf"(^\s*{field}:\s*)(true|false)",
            fr"\g<1>{flag}",
            field,
            allow_missing=True,
        )
    path.write_text(text, encoding="utf-8")


def _update_prepare_input_file(
    path: Path,
    dep_range: tuple[float, float],
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    n_rtp: Sequence[int],
) -> None:
    def _inset_interval(limits: tuple[float, float], domain_limits: tuple[float, float], inset: float) -> list[float]:
        lo = max(float(limits[0]), float(domain_limits[0]))
        hi = min(float(limits[1]), float(domain_limits[1]))
        if lo <= float(domain_limits[0]) + 1.0e-12:
            lo += inset
        if hi >= float(domain_limits[1]) - 1.0e-12:
            hi -= inset
        if hi <= lo:
            mid = 0.5 * (float(domain_limits[0]) + float(domain_limits[1]))
            return [mid, mid]
        return [lo, hi]

    text = path.read_text(encoding="utf-8")
    lim_x_vals = _inset_interval((lon_range[0], lon_range[1]), (lon_range[0], lon_range[1]), 1.0e-3)
    lim_y_vals = _inset_interval((lat_range[0], lat_range[1]), (lat_range[0], lat_range[1]), 1.0e-3)
    lim_z_vals = _inset_interval(
        (max(0.0, dep_range[0]), max(0.0, dep_range[1])),
        (dep_range[0], dep_range[1]),
        1.0e-3,
    )
    lim_x = _fmt_list(lim_x_vals)
    lim_y = _fmt_list(lim_y_vals)
    lim_z = _fmt_list(lim_z_vals)

    text = _replace_regex(text, r"lim_x=\[[^\]]+\]", f"lim_x={lim_x}", "lim_x", allow_missing=True)
    text = _replace_regex(text, r"lim_y=\[[^\]]+\]", f"lim_y={lim_y}", "lim_y", allow_missing=True)
    text = _replace_regex(text, r"lim_z=\[[^\]]+\]", f"lim_z={lim_z}", "lim_z", allow_missing=True)
    text = _replace_regex(text, r"(\"lim_x\"\s*:\s*)\[[^\]]+\]", fr"\g<1>{lim_x}", "lim_x dict", allow_missing=True)
    text = _replace_regex(text, r"(\"lim_y\"\s*:\s*)\[[^\]]+\]", fr"\g<1>{lim_y}", "lim_y dict", allow_missing=True)
    text = _replace_regex(text, r"(\"lim_z\"\s*:\s*)\[[^\]]+\]", fr"\g<1>{lim_z}", "lim_z dict", allow_missing=True)
    text = re.sub(r'(--lim-x",\s*default=")([^"]+)(")', fr"\g<1>{lim_x_vals[0]},{lim_x_vals[1]}\g<3>", text)
    text = re.sub(r'(--lim-y",\s*default=")([^"]+)(")', fr"\g<1>{lim_y_vals[0]},{lim_y_vals[1]}\g<3>", text)
    text = re.sub(r'(--lim-z",\s*default=")([^"]+)(")', fr"\g<1>{lim_z_vals[0]},{lim_z_vals[1]}\g<3>", text)
    text = _replace_regex(
        text,
        r"(model_(?:init|ckb)_N)\d+_\d+_\d+",
        fr"\g<1>{n_rtp[0]}_{n_rtp[1]}_{n_rtp[2]}",
        "prepare_input model sizes",
        allow_missing=True,
    )
    path.write_text(text, encoding="utf-8")


def _update_plot_checkerboard(
    path: Path,
    dep_range: tuple[float, float],
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    n_rtp: Sequence[int],
) -> None:
    text = path.read_text(encoding="utf-8")
    text = _replace_regex(
        text,
        r"(^\s*Ngrid\s*=\s*)\[[^\]]+\]",
        fr"\g<1>[{n_rtp[0]}, {n_rtp[1]}, {n_rtp[2]}]",
        "Ngrid",
        allow_missing=True,
    )
    text = _replace_regex(
        text,
        r"(model_(?:init|ckb)_N)\d+_\d+_\d+",
        fr"\g<1>{n_rtp[0]}_{n_rtp[1]}_{n_rtp[2]}",
        "plot model sizes",
        allow_missing=True,
    )
    path.write_text(text, encoding="utf-8")


def _update_plot_inversion(
    path: Path,
    dep_range: tuple[float, float],
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    n_rtp: Sequence[int],
) -> None:
    text = path.read_text(encoding="utf-8")
    plan_region = _fmt_list([lon_range[0], lon_range[1], lat_range[0], lat_range[1]])
    cross_min = max(0.0, dep_range[0])
    cross_max = max(cross_min, dep_range[1])
    cross_region = _fmt_list([cross_min, cross_max, lat_range[0], lat_range[1]])
    start = _fmt_list([lon_range[0], lat_range[0]])
    end = _fmt_list([lon_range[1], lat_range[1]])
    depth_slice = cross_min + 0.5 * (cross_max - cross_min) if cross_max > cross_min else cross_min

    text = _replace_regex(
        text,
        r"(^\s*Ngrid\s*=\s*)\[[^\]]+\]",
        fr"\g<1>[{n_rtp[0]}, {n_rtp[1]}, {n_rtp[2]}]",
        "Ngrid",
        allow_missing=True,
    )
    text = _replace_regex(text, r"(^\s*region\s*=\s*)\[[^\]]+\]", fr"\g<1>{plan_region}", "region", allow_missing=True)
    text = _replace_regex(text, r"(^\s*region_sec\s*=\s*)\[[^\]]+\]", fr"\g<1>{cross_region}", "region_sec", allow_missing=True)
    text = _replace_regex(text, r"(^\s*start\s*=).*$", fr"\g<1> {start}", "start", allow_missing=True, count=1)
    text = _replace_regex(text, r"(^\s*end\s*=).*$", fr"\g<1> {end}", "end", allow_missing=True, count=1)
    text = _replace_regex(
        text,
        r"(^\s*depth\s*=\s*)[0-9.+-]+",
        fr"\g<1>{_fmt(depth_slice)}",
        "depth slice",
        allow_missing=True,
        count=1,
    )
    text = _replace_regex(
        text,
        r"(model_(?:init|ckb)_N)\d+_\d+_\d+",
        fr"\g<1>{n_rtp[0]}_{n_rtp[1]}_{n_rtp[2]}",
        "plot model sizes",
        allow_missing=True,
    )
    path.write_text(text, encoding="utf-8")


def _copy_template_files(template_dir: Path, target_dir: Path) -> None:
    missing = [name for name in REQUIRED_TEMPLATE_FILES if not (template_dir / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(f"Template {template_dir} missing required files: {missing_text}")

    params_dir = template_dir / "3_input_params"
    if not params_dir.exists():
        raise SystemExit(f"Template {template_dir} missing 3_input_params")

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(params_dir, target_dir / "3_input_params")

    for filename in REQUIRED_TEMPLATE_FILES:
        shutil.copy2(template_dir / filename, target_dir / filename)
    for filename in OPTIONAL_TEMPLATE_FILES:
        src = template_dir / filename
        if src.exists():
            shutil.copy2(src, target_dir / filename)


def _copy_support_scripts(target_dir: Path, kind: str) -> None:
    if kind != "inversion":
        return
    for filename in INVERSION_SUPPORT_SCRIPTS:
        src = BASE_DIR / filename
        if not src.exists():
            raise SystemExit(f"Support script not found: {src}")
        shutil.copy2(src, target_dir / filename)


def _copy_run_root_support_files(output_root: Path) -> None:
    for filename in RUN_ROOT_SUPPORT_FILES:
        src = BASE_DIR / filename
        if not src.exists():
            raise SystemExit(f"Run-root support file not found: {src}")
        shutil.copy2(src, output_root / filename)


def _update_domain_and_runtime(
    target_dir: Path,
    dep_range: tuple[float, float],
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    n_rtp: Sequence[int],
    n_sims: int,
    has_teleseismic_data: bool,
    kind: str,
    enable_azcov_outputs: bool,
) -> None:
    params_dir = target_dir / "3_input_params"
    for yaml_path in params_dir.glob("*.yaml"):
        _update_yaml_domain(yaml_path, dep_range, lat_range, lon_range, n_rtp)
        _update_n_sims(yaml_path, n_sims)
        _update_swap_src_rec(yaml_path, True)
        _update_have_tele_data(yaml_path, has_teleseismic_data)
        _update_ignore_velocity_discontinuity(yaml_path, True)
        if kind == "inversion" and enable_azcov_outputs:
            _update_output_flags(yaml_path, True)

    prepare_file = target_dir / "prepare_input_files.py"
    if prepare_file.exists():
        _update_prepare_input_file(prepare_file, dep_range, lat_range, lon_range, n_rtp)

    plot_file = target_dir / "plot_output.py"
    if plot_file.exists():
        if kind == "checkerboard":
            _update_plot_checkerboard(plot_file, dep_range, lat_range, lon_range, n_rtp)
        elif kind == "inversion":
            _update_plot_inversion(plot_file, dep_range, lat_range, lon_range, n_rtp)

    slurm_file = target_dir / "run_this_example_slurm.sh"
    if slurm_file.exists():
        _update_slurm_ntasks(slurm_file, n_sims)


def _build_phase_folder(
    template_dir: Path,
    target_dir: Path,
    src_rec: Path,
    dep_range: tuple[float, float],
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    avg_receiver_spacing_km: float | None,
    n_sims: int,
    has_teleseismic_data: bool,
    include_weights: bool,
    fixed_weight: float | None,
    kind: str,
    enable_azcov_outputs: bool,
) -> None:
    _copy_template_files(template_dir, target_dir)
    _copy_support_scripts(target_dir, kind)
    n_rtp = _compute_n_rtp(dep_range, lat_range, lon_range, avg_receiver_spacing_km)

    src_dir = target_dir / "1_src_rec_files"
    src_dir.mkdir(parents=True, exist_ok=True)
    _write_src_rec(src_rec, src_dir / "src_rec_config.dat", include_weights, fixed_weight)

    _update_domain_and_runtime(
        target_dir,
        dep_range,
        lat_range,
        lon_range,
        n_rtp,
        n_sims,
        has_teleseismic_data,
        kind,
        enable_azcov_outputs,
    )


def _build_vpvs_folder(
    output_root: Path,
    dep_range: tuple[float, float],
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    avg_receiver_spacing_km: float | None,
    n_sims: int,
    has_teleseismic_data: bool,
) -> None:
    target_dir = output_root / "inversion_VpVs"
    if not TEMPLATE_VPVS.exists():
        raise SystemExit(f"Vp/Vs template not found: {TEMPLATE_VPVS}")
    target_dir.mkdir(parents=True, exist_ok=True)

    ref_yaml = TEMPLATE_VPVS / "3_input_params" / "input_params_ref.yaml"
    plot_file = TEMPLATE_VPVS / "plot_output.py"
    if not ref_yaml.exists() or not plot_file.exists():
        raise SystemExit("Vp/Vs template missing input_params_ref.yaml or plot_output.py")

    shutil.copytree(TEMPLATE_VPVS / "3_input_params", target_dir / "3_input_params")
    shutil.copy2(plot_file, target_dir / "plot_output.py")

    ref_path = target_dir / "3_input_params" / "input_params_ref.yaml"
    n_rtp = _compute_n_rtp(dep_range, lat_range, lon_range, avg_receiver_spacing_km)
    _update_yaml_domain(ref_path, dep_range, lat_range, lon_range, n_rtp)
    _update_n_sims(ref_path, n_sims)
    _update_swap_src_rec(ref_path, True)
    _update_have_tele_data(ref_path, has_teleseismic_data)
    _update_ignore_velocity_discontinuity(ref_path, True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-metadata", required=True, help="Path to region_metadata.json")
    parser.add_argument("--phases", default="P,S", help="Comma-separated phases to prepare (default: P,S).")
    parser.add_argument("--src-rec", help="Override src_rec path (single phase only).")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help=f"Output folder prefix (default: {DEFAULT_PREFIX}).")
    parser.add_argument("--include-weights", action="store_true", help="Keep weight column in src_rec.")
    parser.add_argument("--fixed-weight", type=float, help="Override all pick weights with this value.")
    parser.add_argument(
        "--enable-azcov-outputs",
        action="store_true",
        help="Enable heavy wavefield/kernel/in-process outputs needed for azimuth coverage and ray tracing.",
    )
    args = parser.parse_args()

    phases = _parse_phases(args.phases)
    if args.src_rec and len(phases) > 1:
        raise SystemExit("--src-rec only supports a single phase. Use --phases P or --phases S.")

    required_templates: set[Path] = set()
    for phase in phases:
        if phase not in TEMPLATE_CHECKERBOARD or phase not in TEMPLATE_INVERSION:
            raise SystemExit(f"Unsupported phase: {phase}")
        required_templates.add(TEMPLATE_CHECKERBOARD[phase])
        required_templates.add(TEMPLATE_INVERSION[phase])
    for template in sorted(required_templates):
        if not template.exists():
            raise SystemExit(f"Template folder not found: {template}")

    metadata_path = Path(args.region_metadata).expanduser()
    if not metadata_path.exists():
        raise SystemExit(f"region_metadata.json not found: {metadata_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = RESULTS_DIR / f"{args.prefix}_{timestamp}"
    if output_root.exists():
        raise SystemExit(f"Output folder already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_run_root_support_files(output_root)

    dataset_dir = output_root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    phase_info = []
    for phase in phases:
        phase_metadata = _resolve_metadata_path(metadata_path, phase)
        lat_range, lon_range, dep_range, base_src, avg_receiver_spacing_km, has_teleseismic_data = _load_region_metadata(
            phase_metadata
        )
        lat_range = _expand_lateral_range(tuple(sorted(lat_range)), LATERAL_BUFFER_DEG)
        lon_range = _expand_lateral_range(tuple(sorted(lon_range)), LATERAL_BUFFER_DEG)
        dep_range = tuple(sorted(dep_range))

        if args.src_rec:
            src_path = _resolve_src_rec(args.src_rec, metadata_path)
        else:
            src_path = _resolve_phase_src_rec(base_src, phase_metadata, phase, strict=len(phases) > 1)

        if not src_path.exists():
            raise SystemExit(f"src_rec file not found: {src_path}")

        recv_range = _receiver_depth_range(src_path)
        if recv_range:
            dep_range = (min(dep_range[0], recv_range[0]), max(dep_range[1], recv_range[1]))
        dep_range = (math.floor(dep_range[0]), math.ceil(dep_range[1]))
        dep_range = tuple(sorted(dep_range))

        station_count = _count_unique_stations(src_path)
        if station_count >= 56:
            n_sims = 56
        elif station_count > 0:
            n_sims = station_count
        else:
            n_sims = 1

        dataset_dest = dataset_dir / f"src_rec_{phase}.dat"
        _write_src_rec(src_path, dataset_dest, args.include_weights, args.fixed_weight)

        phase_info.append(
            {
                "phase": phase,
                "lat_range": lat_range,
                "lon_range": lon_range,
                "dep_range": dep_range,
                "src_rec": src_path,
                "avg_receiver_spacing_km": avg_receiver_spacing_km,
                "n_sims": n_sims,
                "station_count": station_count,
                "has_teleseismic_data": has_teleseismic_data,
            }
        )

    for info in phase_info:
        phase = info["phase"]
        checker_dir = output_root / f"checkerboard_{phase}"
        inversion_dir = output_root / f"inversion_{phase}"

        _build_phase_folder(
            TEMPLATE_CHECKERBOARD[phase],
            checker_dir,
            info["src_rec"],
            info["dep_range"],
            info["lat_range"],
            info["lon_range"],
            info["avg_receiver_spacing_km"],
            info["n_sims"],
            info["has_teleseismic_data"],
            args.include_weights,
            args.fixed_weight,
            kind="checkerboard",
            enable_azcov_outputs=False,
        )
        _build_phase_folder(
            TEMPLATE_INVERSION[phase],
            inversion_dir,
            info["src_rec"],
            info["dep_range"],
            info["lat_range"],
            info["lon_range"],
            info["avg_receiver_spacing_km"],
            info["n_sims"],
            info["has_teleseismic_data"],
            args.include_weights,
            args.fixed_weight,
            kind="inversion",
            enable_azcov_outputs=args.enable_azcov_outputs,
        )

        print(
            f"Prepared phase {phase}: {checker_dir} and {inversion_dir} "
            f"(stations={info['station_count']}, n_sims={info['n_sims']})"
        )

    phase_set = {info["phase"] for info in phase_info}
    if {"P", "S"}.issubset(phase_set):
        ref = next(item for item in phase_info if item["phase"] == "P")
        has_teleseismic_data = any(bool(item.get("has_teleseismic_data", False)) for item in phase_info)
        _build_vpvs_folder(
            output_root,
            ref["dep_range"],
            ref["lat_range"],
            ref["lon_range"],
            ref["avg_receiver_spacing_km"],
            ref["n_sims"],
            has_teleseismic_data,
        )

    if VPVS_SCRIPT.exists():
        shutil.copy2(VPVS_SCRIPT, output_root / VPVS_SCRIPT.name)

    print(f"Dataset files copied to: {dataset_dir}")


if __name__ == "__main__":
    main()
