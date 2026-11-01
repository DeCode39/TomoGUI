"""Read current workspace defaults; never replace the maintained templates."""
from __future__ import annotations

import ast
import math
from pathlib import Path
import yaml

from .workspace import load_module


def literal_setting(path: Path, name: str):
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f'{name} was not found in {path}.')


def resolution_defaults(root: Path) -> dict:
    sweep = root / 'RESULTS/resolution_kd_sweep/prepare_resolution_kd_sweep.py'
    current = root / 'RESULTS/JpGU/build_jpgu_resolution_sweeps.py'
    source = current if current.is_file() else sweep
    return {'candidates': list(literal_setting(source, 'KDENSITY_VALUES' if source == current else 'DEFAULT_KDENSITY')),
            'waves': literal_setting(sweep, 'DEFAULT_WAVES'), 'source': str(source.relative_to(root)),
            'waves_source': str(sweep.relative_to(root))}


def checker_defaults(root: Path, phase: str = "P") -> dict:
    path = root / f"RESULTS/TEST/checkerboard_{phase}/prepare_input_files.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_base_checkerboard_defaults":
            for item in node.body:
                if isinstance(item, ast.Return):
                    return ast.literal_eval(item.value)
    raise ValueError(f"Checkerboard defaults not found in {path}")


def current_defaults(root: Path) -> dict:
    prep = load_module(root / "Scripts/prepare_files.py")
    path = root / "RESULTS/TEST/inversion_P/3_input_params/input_params_joint_step2.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Read argparse defaults without importing plotting/scanning dependencies.
    scanner = ast.parse((root / "Scripts/region_scanner_cwb_square.py").read_text(encoding="utf-8"))
    region = {}
    for node in ast.walk(scanner):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            names = [n.value for n in node.args if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            for kw in node.keywords:
                if kw.arg == "default":
                    try:
                        value = ast.literal_eval(kw.value)
                    except (ValueError, TypeError):
                        continue
                    if names:
                        region[names[0].lstrip("-").replace("-", "_")] = value
    return {"checkerboard": checker_defaults(root), "region": region,
            "grid": {"spacing_scale": 1.0, "depth_spacing": 1.0, "inversion_scale": 0.75,
                     "lateral_buffer": prep.LATERAL_BUFFER_DEG},
            "keep_pick_weights": False,  # prepare_files.py's unchanged --include-weights default
            "iterations": config["model_update"]["max_iterations"],
            "kdensity": config["model_update"]["optim_method_0"]["Kdensity_coe"],
            "sequence": ["resolution_kd", "checkerboard", "inversion"],
            "kd_candidates": resolution_defaults(root)['candidates'],
            "kd_workflow": resolution_defaults(root)}


def grid_estimate(root: Path, metadata: dict, tuning: dict | None = None) -> dict:
    tuning = tuning or {}
    prep = load_module(root / "Scripts/prepare_files.py")
    bounds = metadata["region_bounds"]
    spacing = metadata.get("avg_receiver_spacing_km")
    if spacing is not None and float(spacing) <= 0:
        spacing = None
    override = tuning.get("receiver_spacing")
    spacing = float(override) if override not in (None, "") else spacing
    scale = float(tuning.get("spacing_scale", 1.0))
    inv_scale = float(tuning.get("inversion_scale", .75))
    depth_spacing = float(tuning.get("depth_spacing", 1.0))
    if not all(math.isfinite(n) for n in (scale, inv_scale, depth_spacing)):
        raise ValueError("Grid settings must be finite.")
    if scale <= 0 or depth_spacing <= 0 or not 0 < inv_scale <= 1:
        raise ValueError("Spacings must be positive; inversion scale must be in (0, 1].")
    if spacing is not None and (not math.isfinite(float(spacing)) or float(spacing) <= 0):
        raise ValueError("Receiver spacing must be positive and finite.")
    lat = prep._expand_lateral_range((bounds['lat_min'], bounds['lat_max']), prep.LATERAL_BUFFER_DEG)
    lon = prep._expand_lateral_range((bounds['lon_min'], bounds['lon_max']), prep.LATERAL_BUFFER_DEG)
    dep = (bounds['depth_min'], bounds['depth_max'])
    source = Path(metadata.get('src_rec', ''))
    if not source.is_absolute():
        source = root / source
    if source.is_file():
        receiver = prep._receiver_depth_range(source)
        if receiver:
            dep = (min(dep[0], receiver[0]), max(dep[1], receiver[1]))
    dep = (math.floor(dep[0]), math.ceil(dep[1]))
    counts = prep._compute_n_rtp(dep, lat, lon, float(spacing) * scale if spacing else None)
    if depth_spacing != 1:
        counts[0] = max(1, round((dep[1] - dep[0]) / depth_spacing) + 1)
    inversion = prep._compute_base_inv_grid(counts) if inv_scale == .75 else [max(1, min(n, round(n * inv_scale))) for n in counts]
    return {"n_rtp": counts, "n_inv_dep_lat_lon": inversion,
            "measured_receiver_spacing_km": metadata.get("avg_receiver_spacing_km"),
            "effective_receiver_spacing_km": float(spacing) * scale if spacing else None,
            "domain": {"min_max_dep": list(dep), "min_max_lat": list(lat), "min_max_lon": list(lon)},
            "tuning": {"spacing_scale": scale, "depth_spacing": depth_spacing, "inversion_scale": inv_scale,
                       "receiver_spacing": override}, "fallback": spacing is None}
