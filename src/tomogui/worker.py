"""Scientific adapters. Each job runs in its own interpreter and artifact folder."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

from .data import canonicalize, detect, read_src_rec, summarize, write_src_rec
from .defaults import checker_defaults, grid_estimate
from .quality import quality_weight
from .workspace import Workspace, load_module


def run(command, cwd):
    print("Running:", " ".join(str(s) for s in command), flush=True)
    subprocess.run([str(s) for s in command], cwd=cwd, check=True)


def prepare_data(ws: Workspace, payload: dict) -> dict:
    source = ws.path(payload["path"])
    info = detect(source, ws.root)
    folder = ws.artifact("datasets")
    output = folder / "src_rec.dat"
    quality = payload.get("quality") or {}
    if info["kind"] == "src_rec":
        if quality.get("column"):
            raise ValueError("An existing src_rec has weights, not a quality column. Choose a table quality column before conversion.")
        shutil.copy2(source, output)
    elif info["kind"] in {"cwb", "salute"}:
        station = ws.path(payload.get("stations", ""))
        if not station.is_file():
            raise ValueError("Select the station metadata file for this catalog.")
        if quality.get("column"):
            raise ValueError("Column quality rules apply to arrival tables; catalog conversion uses the existing converter's quality mapping.")
        command = ([sys.executable, ws.root / "Scripts/convert_cwb_to_src_rec.py", "--cwb", source, "--sta", station, "--out", output]
                   if info["kind"] == "cwb" else
                   [sys.executable, ws.root / "Scripts/convert_salute_to_src_rec.py", "--events", source, "--stations", station, "--out", output])
        run(command, folder)
    elif info["kind"] == "csv":
        import pandas as pd
        module = load_module(ws.root / "Scripts/region_scanner_pipeline.py")
        with source.open(encoding='utf-8-sig') as handle:
            is_isc = any(line.lstrip().startswith('EVENTID') for line in handle)
        raw = module.read_isc_csv(str(source)) if is_isc else pd.read_csv(source, sep=info.get('delimiter',','), dtype=str)
        raw.columns = [s.strip().upper() for s in raw.columns]
        aliases = {"EVENT_ID": "EVENTID", "STATION_LAT": "STLA", "STATION_LON": "STLO", "ORIGIN_LAT": "EVLA",
                   "ORIGIN_LON": "EVLO", "ORIGIN_DATE": "ORIGIN DATE", "ORIGIN_TIME": "ORIGIN TIME", "ARRIVAL_DATE": "DATE", "ARRIVAL_TIME": "TIME"}
        raw = raw.rename(columns={key: value for key, value in aliases.items() if key in raw and value not in raw})
        for key, name in (payload.get("mapping") or {}).items():
            if name and name.upper() in raw and key in {"EVENTID", "PHASE", "DATE", "TIME", "STA", "STLA", "STLO", "ORIGIN DATE", "ORIGIN TIME", "EVLA", "EVLO", "DEPTH"}:
                raw[key] = raw[name.upper()]
        mapped = module.extract_columns(raw)
        mapped.setdefault('elev', pd.Series(0.0, index=raw.index))
        args = module.build_parser().parse_args([])
        arrivals = module.build_src_rec_df(mapped, args)
        original = module.build_src_rec_object(arrivals, args, output)
        # Apply custom PICK weights only after the original helper computes its
        # source weights; the event-weight semantics must remain unchanged.
        if quality.get("column"):
            column = quality["column"].upper()
            if column not in raw:
                raise ValueError("Selected quality column does not exist.")
            lookup = defaultdict(deque)
            for index, row in raw.iterrows():
                station = str(mapped["station"].iloc[index]).strip()
                if args.name_netsta and "net" in mapped and pd.notna(mapped['net'].iloc[index]) and str(mapped['net'].iloc[index]).strip():
                    station = str(mapped['net'].iloc[index]).strip() + "." + station
                lookup[(str(mapped['event_id'].iloc[index]), station, str(mapped['phase'].iloc[index]).upper())].append(row[column])
            for index, row in original.rec_points.iterrows():
                event_id = str(original.src_points.loc[row['src_index'], 'event_id'])
                values = lookup[(event_id, str(row['staname']), str(row['phase']).upper())]
                if len(values) != 1:
                    raise ValueError("Quality mapping is ambiguous for duplicate event/station/phase arrivals. Deduplicate the table first.")
                original.rec_points.loc[index, 'weight'] = quality_weight(values[0], row['weight'], quality)
        original.write(str(output))
    else:
        raise ValueError(info.get("message", "Unsupported data structure."))
    canonicalize(output)
    summary = summarize(output)
    changes = {"data": summary, "region": None, "model": None, "run": None,
               "custom_pick_weights": bool(quality.get('column'))}
    if quality.get("column"):
        changes["keep_pick_weights"] = True
    (folder / "provenance.json").write_text(json.dumps({"source": str(source), "detection": info['kind'], "quality": quality}, indent=2), encoding="utf-8")
    return {"summary": summary, "state_updates": changes}


def filter_region(ws: Workspace, payload: dict) -> dict:
    data = ws.read_state().get("data")
    if not data:
        raise ValueError("Prepare a src_rec dataset first.")
    source = ws.path(data["path"])
    folder = ws.artifact("regions")
    settings = payload.get("settings", {})
    bounds = payload["bounds"]
    for lo, hi in (("lat_min", "lat_max"), ("lon_min", "lon_max"), ("depth_min", "depth_max")):
        if not math.isfinite(float(bounds[lo])) or not math.isfinite(float(bounds[hi])) or float(bounds[lo]) >= float(bounds[hi]):
            raise ValueError("Region bounds must be finite and increasing.")
    script = "region_scanner_pipeline.py" if payload.get("scan") else "region_scanner_cwb_square.py"
    command = [sys.executable, ws.root / "Scripts" / script, "--src-rec", source, "--output-root", folder,
               "--lat-range", bounds['lat_min'], bounds['lat_max'], "--lon-range", bounds['lon_min'], bounds['lon_max'],
               "--depth-range", bounds['depth_min'], bounds['depth_max']]
    numeric = {"min_p_picks", "min_s_picks", "decluster_block_km", "central_area_fraction", "azimuth_bin_deg"}
    for key in numeric:
        if key in settings:
            command += ["--" + key.replace("_", "-"), settings[key]]
    for key in ("require_all_azimuth_bins", "teleseismic", "ray_coverage"):
        if key in settings:
            command.append("--" + ("" if settings[key] else "no-") + key.replace("_", "-"))
    if payload.get("scan"):
        command += ["--scan-regions", "--include-s"]
    if payload.get('phases') == 'P':
        command += ["--no-include-s", "--min-s-picks", "0"] if not payload.get('scan') else ["--min-s-picks", "0"]
    elif payload.get('phases') == 'S':
        command += ["--phases", "S", "--min-p-picks", "0"]
    if ws.read_state().get('custom_pick_weights') and script == 'region_scanner_cwb_square.py':
        # A user-defined quality conversion already produced physical weights.
        # Disable only the scanner's code/legacy-quarter-step interpretation for
        # this opt-in job; do not edit the scanner or its default CLI behavior.
        sys.path.insert(0, str(ws.root / 'Scripts'))
        scanner = load_module(ws.root / 'Scripts' / script)
        scanner._remap_cwb_weights = lambda values: values
        previous = sys.argv
        try:
            sys.argv = [str(s) for s in command[1:]]
            scanner.main()
        finally:
            sys.argv = previous
    else:
        run(command, ws.root)
    candidates = sorted(folder.glob("region_*/region_metadata.json"))
    if not candidates:
        raise ValueError("No region passed the current filters. Review minimum picks and coverage settings.")
    for source_copy in folder.rglob('src_rec.dat'):
        canonicalize(source_copy)
    metadata = json.loads(candidates[0].read_text(encoding="utf-8"))
    final = candidates[0].parent / "src_rec.dat"
    summary = summarize(final)
    metadata.update(src_rec=str(final), avg_receiver_spacing_km=summary['avg_receiver_spacing_km'])
    candidates[0].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    region = {"metadata_path": str(candidates[0]), "metadata": metadata, "summary": summary,
              "candidates": [str(p) for p in candidates]}
    return {"state_updates": {"region": region, "model": None, "run": None}, "region": region}


def model_preview(ws: Workspace, payload: dict) -> dict:
    state = ws.read_state()
    region = state.get("region")
    if not region:
        raise ValueError("Apply a filtered region first; model grids use the final src_rec.")
    phase = payload.get("phase", "P")
    if phase not in {"P", "S"}:
        raise ValueError("Select P or S.")
    if phase not in region['summary']['phases']:
        raise ValueError(f'The filtered region contains no {phase} arrivals.')
    phase_file = Path(region['metadata_path']).parent / phase / 'region_metadata.json'
    metadata = json.loads(phase_file.read_text(encoding='utf-8')) if phase_file.exists() else region['metadata']
    grid = grid_estimate(ws.root, metadata, payload.get('grid'))
    if math.prod(grid['n_rtp']) > 20_000_000:
        raise ValueError("This preview exceeds 20 million grid nodes. Increase spacing for an interactive preview.")
    folder = ws.artifact("models")
    template = ws.root / f"RESULTS/TEST/checkerboard_{phase}"
    params = yaml.safe_load((template / "3_input_params/input_params_signal.yaml").read_text(encoding="utf-8"))
    params['domain'].update(grid['domain'], n_rtp=grid['n_rtp'])
    param_file = folder / "input_params.yaml"
    param_file.write_text(yaml.safe_dump(params, sort_keys=False), encoding="ascii")
    module = load_module(template / "prepare_input_files.py")
    builder = module.BuildInitialModel(par_file=str(param_file), output_dir=str(folder))
    builder.build_initial_model()
    initial = folder / "initial_model.h5"
    builder.am.write(str(initial))
    kind = payload.get('kind', 'checkerboard')
    if kind == 'initial':
        output = initial
    elif kind == 'dual_layer':
        spec = (ws.root / "custom_models/model_definitions/dual_anisotropy_ns_ew.txt").read_text(encoding="utf-8")
        for key, value in {'par_file': param_file, 'background_model': initial, 'output_model': folder / 'model.h5'}.items():
            spec = re.sub(rf'(?m)^{key}=.*$', lambda _: f'{key}={value.as_posix()}', spec)
        definition = folder / 'dual_layer.txt'
        definition.write_text(spec, encoding="ascii")
        custom = load_module(ws.root / 'custom_models/build_custom_model.py')
        output = custom.build_model(definition)
    elif kind == 'checkerboard':
        config = module._default_checkerboard_config(str(param_file))
        values = payload.get('checkerboard', {})
        for name in ('n_pert_x', 'n_pert_y', 'n_pert_z'):
            value = values.get(name, config[name])
            if isinstance(value, bool) or int(value) != float(value) or not 1 <= int(value) <= 100:
                raise ValueError("Checkerboard counts must be integers from 1 to 100.")
            config[name] = int(value)
        for name in ('pert_vel', 'pert_ani', 'ani_dir'):
            value = float(values.get(name, config[name]))
            if not math.isfinite(value) or (name != 'ani_dir' and not 0 <= value < 1):
                raise ValueError("Perturbation strengths must be finite fractions in [0, 1).")
            config[name] = value
        checker = module.Checker(str(initial), para_fname=str(param_file))
        checker.checkerboard(**config)
        output = folder / 'model.h5'
        checker.write(str(output))
    else:
        raise ValueError("Unknown model preset.")
    model = {"path": str(output), "initial": str(initial), "phase": phase, "kind": kind,
             "settings": payload, "grid": grid}
    return {"model": model, "state_updates": {"model": model}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    ws = Workspace(Path(args.workspace))
    spec = json.loads(ws.path(args.spec).read_text(encoding='utf-8'))
    operations = {'prepare_data': prepare_data, 'filter_region': filter_region, 'model_preview': model_preview}
    from .runs import generate, execute, score, select_kd, postprocess
    operations.update(generate=generate, execute=execute, score=score, select_kd=select_kd, postprocess=postprocess)
    if spec['operation'] not in operations:
        raise ValueError('Unknown workflow operation.')
    result = operations[spec['operation']](ws, spec['payload'])
    Path(args.result).write_text(json.dumps(result, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
