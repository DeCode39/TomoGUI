#!/usr/bin/env python3
"""Run default TomoATT postprocess helpers without failing the whole workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _infer_phase(run_dir: Path) -> str:
    name = run_dir.name.upper()
    if name.endswith("_S"):
        return "S"
    return "P"


def _run(script_dir: Path, script_name: str, extra_args: list[str]) -> int:
    script_path = script_dir / script_name
    if not script_path.exists():
        print(f"Skipping {script_name}: file not present.")
        return 0
    cmd = [sys.executable, str(script_path), *extra_args]
    print(f"Running {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Warning: {script_name} exited with code {result.returncode}; continuing.")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent, help="Path to inversion_P or inversion_S.")
    parser.add_argument("--phase", default=None, help="Phase tag used by azimuth and trace helpers.")
    parser.add_argument("--step", default="joint_step2", help="Step tag used by azimuth and trace helpers.")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    phase = args.phase or _infer_phase(run_dir)

    _run(script_dir, "plot_qc.py", ["--run-dir", str(run_dir)])
    _run(script_dir, "build_azcov_archive.py", ["--run-dir", str(run_dir), "--phase", phase, "--step", args.step])
    _run(script_dir, "plot_azimuthal_coverage.py", ["--run-dir", str(run_dir), "--phase", phase, "--step", args.step])

    archive_path = run_dir / "azcov_archive" / f"azcov_{phase}_{args.step}_ray_archive.h5"
    _run(
        script_dir,
        "plot_single_ray_trace.py",
        [
            "--run-dir",
            str(run_dir),
            "--phase",
            phase,
            "--archive",
            str(archive_path),
            "--wavefield",
        ],
    )


if __name__ == "__main__":
    main()
