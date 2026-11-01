#!/usr/bin/env python3
"""Create an inversion_VpVs folder from inversion P/S results."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DEFAULT_TEMPLATE = ROOT_DIR / "RESULTS" / "TEST" / "inversion_VpVs" / "plot_output.py"


def _copy_if_exists(src: Path, dest: Path) -> None:
    if not src.exists():
        raise SystemExit(f"Required file not found: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _pick_best_joint_step(root: Path) -> tuple[str, Path, Path, Path]:
    for step in ("joint_step3", "joint_step2", "joint_step1"):
        inv_p = root / "inversion_P" / "OUTPUT_FILES" / f"OUTPUT_FILES_{step}" / "final_model.h5"
        inv_s = root / "inversion_S" / "OUTPUT_FILES" / f"OUTPUT_FILES_{step}" / "final_model.h5"
        ref_yaml = root / "inversion_P" / "3_input_params" / f"input_params_{step}.yaml"
        if inv_p.exists() and inv_s.exists() and ref_yaml.exists():
            return step, inv_p, inv_s, ref_yaml
    raise SystemExit("No common joint-step final_model.h5 outputs found in inversion_P and inversion_S.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inversion_VpVs folder from inversion outputs.")
    parser.add_argument(
        "--dataset-root",
        default=".",
        help="Dataset root containing inversion_P and inversion_S (default: current directory).",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE),
        help="Template plot_output.py to copy into inversion_VpVs.",
    )
    parser.add_argument(
        "--force-plot",
        action="store_true",
        help="Overwrite existing inversion_VpVs/plot_output.py if present.",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    step, inv_p, inv_s, ref_yaml = _pick_best_joint_step(root)

    vpvs_dir = root / "inversion_VpVs"
    model_dir = vpvs_dir / "2_models"
    params_dir = vpvs_dir / "3_input_params"
    img_dir = vpvs_dir / "img"
    model_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(ref_yaml, params_dir / "input_params_ref.yaml")
    _copy_if_exists(inv_p, model_dir / "model_final_P.h5")
    _copy_if_exists(inv_s, model_dir / "model_final_S.h5")

    plot_dest = vpvs_dir / "plot_output.py"
    if args.force_plot or not plot_dest.exists():
        template = Path(args.template).expanduser()
        _copy_if_exists(template, plot_dest)

    print(f"Vp/Vs folder ready: {vpvs_dir} (from {step})")


if __name__ == "__main__":
    main()
