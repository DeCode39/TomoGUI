#!/usr/bin/env python3
"""Prepare an automated sinusoidal checkerboard / Kdensity_coe sweep.

The prepared layout is intentionally standalone under RESULTS/resolution_kd_sweep
so it can be uploaded and run on the HPC without modifying the normal SALUTE
run folders.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = ROOT.parent / "SALUTEv5_FULL_reg_20260325_121824_Ptests_split_step2" / "checkerboard_P_paper"
DEFAULT_REAL_YAML = (
    ROOT.parent
    / "SALUTEv5_FULL_reg_20260325_121824_Ptests_split_step2"
    / "inversion_P"
    / "3_input_params"
    / "input_params_joint_step2.yaml"
)
DEFAULT_KDENSITY = (0.0, 0.3, 0.6, 0.9)
DEFAULT_WAVES = {
    "wl0p5": {
        "velocity_period_deg": 0.375,
        "anisotropy_period_deg": 0.5,
        "depth_period_km": 5.0,
        "weight": 0.25,
        "seed": 170501,
    },
    "wl1p0": {
        "velocity_period_deg": 0.75,
        "anisotropy_period_deg": 1.0,
        "depth_period_km": 10.0,
        "weight": 0.50,
        "seed": 170502,
    },
    "wl2p0": {
        "velocity_period_deg": 1.5,
        "anisotropy_period_deg": 2.0,
        "depth_period_km": 20.0,
        "weight": 0.25,
        "seed": 170503,
    },
}
COPY_FILES = (
    "prepare_input_files.py",
    "assign_gaussian_noise.py",
    "build_paper_resolution_mask.py",
    "plot_checkerboard_preview.py",
    "plot_checkerboard_results_mpl.py",
    "plot_output.py",
)
COPY_DIRS = ("1_src_rec_files", "2_models", "3_input_params")


def parse_number_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def kd_tag(value: float) -> str:
    return f"kd{value:.4g}".replace("-", "m").replace(".", "p")


def read_grid_from_yaml(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"(?m)^\s*n_inv_dep_lat_lon:\s*\[([^\]]+)\]", text)
    if not match:
        raise SystemExit(f"Could not find n_inv_dep_lat_lon in {path}")
    values = [int(round(float(part.strip()))) for part in match.group(1).split(",")]
    if len(values) != 3:
        raise SystemExit(f"Expected three n_inv_dep_lat_lon values in {path}")
    return values


def patch_regex(text: str, pattern: str, repl: str, label: str) -> str:
    new_text, count = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if count == 0:
        raise SystemExit(f"Could not patch {label}")
    return new_text


def patch_yaml_value(path: Path, *, kdensity: float | None = None, n_inv: list[int] | None = None, src_rec_file: str | None = None) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if kdensity is not None:
        text = patch_regex(
            text,
            r"(^\s*Kdensity_coe:\s*)[-+0-9.eE]+",
            rf"\g<1>{kdensity:g}",
            f"Kdensity_coe in {path}",
        )
    if n_inv is not None:
        n_inv_text = f"[{n_inv[0]}, {n_inv[1]}, {n_inv[2]}]"
        text = patch_regex(
            text,
            r"(^\s*n_inv_dep_lat_lon:\s*)\[[^\]]+\]",
            rf"\g<1>{n_inv_text}",
            f"n_inv_dep_lat_lon in {path}",
        )
    if src_rec_file is not None:
        text = patch_regex(
            text,
            r"(^\s*src_rec_file:\s*).*$",
            rf"\g<1>{src_rec_file} # source receiver file path",
            f"src_rec_file in {path}",
        )
    path.write_text(text, encoding="utf-8")


def patch_prepare_input(path: Path, wave: dict[str, float]) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    replacements = {
        "VELOCITY_PERIOD_DEG": wave["velocity_period_deg"],
        "ANISOTROPY_PERIOD_DEG": wave["anisotropy_period_deg"],
        "DEPTH_PERIOD_KM": wave["depth_period_km"],
    }
    for name, value in replacements.items():
        text = patch_regex(
            text,
            rf"(?m)^{name}\s*=\s*[-+0-9.eE]+",
            f"{name} = {float(value):g}",
            f"{name} in {path}",
        )
    path.write_text(text, encoding="utf-8")


def patch_noise_seed(path: Path, seed: int) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "NOISE_SEED" in text:
        text = re.sub(r"(?m)^NOISE_SEED\s*=\s*\d+", f"NOISE_SEED = {seed}", text)
    else:
        text = text.replace("from pytomoatt.src_rec import SrcRec\n", "from pytomoatt.src_rec import SrcRec\nimport numpy as np\n\nNOISE_SEED = %d\n" % seed)
        text = text.replace("    sr = SrcRec.read(in_fname)\n", "    np.random.seed(NOISE_SEED)\n    sr = SrcRec.read(in_fname)\n")
    path.write_text(text, encoding="utf-8")


def regenerate_wave_checkerboard(wave_dir: Path) -> None:
    """Keep local true checkerboards consistent with the patched wave periods."""
    subprocess.run([sys.executable, "prepare_input_files.py"], cwd=wave_dir, check=True)


def remove_outputs(path: Path) -> None:
    for name in ("OUTPUT_FILES", "img", "paper_resolution_mask", "resolution_mask", "__pycache__"):
        target = path / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for pattern in ("tomoatt_*.out", "tomoatt_*.err", "*.log"):
        for target in path.glob(pattern):
            if target.is_file():
                target.unlink()


def copy_missing_tree(src: Path, dest: Path) -> None:
    """Copy template contents that are absent without touching run outputs."""
    if not src.exists():
        raise SystemExit(f"Required template path not found: {src}")
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dest / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def ensure_template_files(template: Path, dest: Path) -> None:
    """Repair an existing generated case without deleting completed outputs."""
    for dirname in COPY_DIRS:
        copy_missing_tree(template / dirname, dest / dirname)
    for filename in COPY_FILES:
        src = template / filename
        if not src.exists():
            raise SystemExit(f"Required template file not found: {src}")
        target = dest / filename
        if not target.exists():
            shutil.copy2(src, target)


def copy_template(template: Path, dest: Path, *, force: bool) -> None:
    if dest.exists():
        if not force:
            ensure_template_files(template, dest)
            return
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for dirname in COPY_DIRS:
        shutil.copytree(template / dirname, dest / dirname)
    for filename in COPY_FILES:
        src = template / filename
        if src.exists():
            shutil.copy2(src, dest / filename)
    remove_outputs(dest)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def merge_manifest_rows(existing: list[dict[str, object]], new_rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for row in existing:
        merged[str(row[key])] = row
    for row in new_rows:
        merged[str(row[key])] = row
    return list(merged.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--real-inversion-yaml", type=Path, default=DEFAULT_REAL_YAML)
    parser.add_argument("--kd-values", default=",".join(str(v) for v in DEFAULT_KDENSITY))
    parser.add_argument("--n-inv-dep-lat-lon", help="Override base inversion grid, e.g. 41,46,46.")
    parser.add_argument("--force", action="store_true", help="Recreate existing generated wave/run folders.")
    args = parser.parse_args()

    template = args.template.resolve()
    if not template.exists():
        raise SystemExit(f"Template folder not found: {template}")

    if args.n_inv_dep_lat_lon:
        n_inv = [int(round(v)) for v in parse_number_list(args.n_inv_dep_lat_lon)]
        if len(n_inv) != 3:
            raise SystemExit("--n-inv-dep-lat-lon must contain three values")
    else:
        n_inv = read_grid_from_yaml(args.real_inversion_yaml.resolve())

    kd_values = sorted(set(round(v, 6) for v in parse_number_list(args.kd_values)))
    if not kd_values:
        raise SystemExit("No Kdensity_coe values were requested")

    waves_root = ROOT / "waves"
    runs_root = ROOT / "runs"
    wave_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []

    for wave_tag, wave in DEFAULT_WAVES.items():
        wave_dir = waves_root / wave_tag
        copy_template(template, wave_dir, force=args.force)
        patch_prepare_input(wave_dir / "prepare_input_files.py", wave)
        patch_noise_seed(wave_dir / "assign_gaussian_noise.py", int(wave["seed"]))
        patch_yaml_value(wave_dir / "3_input_params" / "input_params_inv.yaml", n_inv=n_inv)
        regenerate_wave_checkerboard(wave_dir)

        wave_rows.append(
            {
                "wave_tag": wave_tag,
                "velocity_period_deg": wave["velocity_period_deg"],
                "anisotropy_period_deg": wave["anisotropy_period_deg"],
                "depth_period_km": wave["depth_period_km"],
                "weight": wave["weight"],
                "noise_seed": wave["seed"],
                "wave_dir": str(wave_dir.relative_to(ROOT)),
                "true_model": str((wave_dir / "2_models" / "model_ckb_N55_61_61.h5").relative_to(ROOT)),
                "noisy_src_rec": str((wave_dir / "OUTPUT_FILES" / "OUTPUT_FILES_signal" / "src_rec_file_forward_noisy.dat").relative_to(ROOT)),
            }
        )

        for kdensity in kd_values:
            run_tag = f"{wave_tag}_{kd_tag(kdensity)}"
            run_dir = runs_root / run_tag
            copy_template(template, run_dir, force=args.force)

            # Avoid stale checkerboard models in inversion folders. The true
            # model for scoring always comes from the wavelength-specific wave folder.
            for stale_model in (run_dir / "2_models").glob("model_ckb_*.h5"):
                stale_model.unlink()

            noisy_from_run = f"../../waves/{wave_tag}/OUTPUT_FILES/OUTPUT_FILES_signal/src_rec_file_forward_noisy.dat"
            patch_yaml_value(
                run_dir / "3_input_params" / "input_params_inv.yaml",
                kdensity=kdensity,
                n_inv=n_inv,
                src_rec_file=noisy_from_run,
            )

            run_rows.append(
                {
                    "run_tag": run_tag,
                    "wave_tag": wave_tag,
                    "kdensity_coe": f"{kdensity:g}",
                    "velocity_period_deg": wave["velocity_period_deg"],
                    "anisotropy_period_deg": wave["anisotropy_period_deg"],
                    "depth_period_km": wave["depth_period_km"],
                    "wave_weight": wave["weight"],
                    "run_dir": str(run_dir.relative_to(ROOT)),
                    "wave_dir": str(wave_dir.relative_to(ROOT)),
                    "initial_model": str((run_dir / "2_models" / "model_init_N55_61_61.h5").relative_to(ROOT)),
                    "true_model": str((wave_dir / "2_models" / "model_ckb_N55_61_61.h5").relative_to(ROOT)),
                    "recovered_model": str((run_dir / "OUTPUT_FILES" / "OUTPUT_FILES_inv" / "final_model.h5").relative_to(ROOT)),
                    "noisy_src_rec": str((wave_dir / "OUTPUT_FILES" / "OUTPUT_FILES_signal" / "src_rec_file_forward_noisy.dat").relative_to(ROOT)),
                }
            )

    all_wave_rows = merge_manifest_rows(read_manifest(ROOT / "manifest_signal.csv"), wave_rows, "wave_tag")
    all_run_rows = merge_manifest_rows(read_manifest(ROOT / "manifest_inversion.csv"), run_rows, "run_tag")
    pending_wave_rows = [
        row for row in wave_rows if not (ROOT / str(row["noisy_src_rec"])).exists()
    ]
    pending_run_rows = [
        row for row in run_rows if not (ROOT / str(row["recovered_model"])).exists()
    ]

    write_manifest(ROOT / "manifest_signal.csv", all_wave_rows)
    write_manifest(ROOT / "manifest_inversion.csv", all_run_rows)
    write_manifest(ROOT / "manifest_signal_pending.csv", pending_wave_rows)
    write_manifest(ROOT / "manifest_inversion_pending.csv", pending_run_rows)

    print(f"Prepared resolution/Kdensity sweep under {ROOT}")
    print(f"Base inversion grid: {n_inv}")
    print(f"Wavelength cases: {', '.join(DEFAULT_WAVES)}")
    print(f"Kdensity_coe values: {', '.join(f'{v:g}' for v in kd_values)}")
    print(f"Signal tasks total/pending: {len(all_wave_rows)}/{len(pending_wave_rows)}")
    print(f"Inversion tasks total/pending: {len(all_run_rows)}/{len(pending_run_rows)}")
    print("Next HPC step: bash submit_resolution_kd_sweep.sh")


if __name__ == "__main__":
    main()
