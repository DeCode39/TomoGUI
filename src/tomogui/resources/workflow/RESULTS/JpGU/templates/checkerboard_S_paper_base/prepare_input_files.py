import os
import sys
from pathlib import Path

import h5py
import numpy as np

ATTModel = None
SrcRec = None

VELOCITY_PERIOD_DEG = 0.75
ANISOTROPY_PERIOD_DEG = 1.0
DEPTH_PERIOD_KM = 10.0
BOTTOM_BUFFER_KM = 5.0
PERTURBATION_FRACTION = 0.03
FAST_AZIMUTH_NEG_DEG = 60.0
FAST_AZIMUTH_POS_DEG = 150.0


def require_pytomoatt():
    global ATTModel, SrcRec
    if ATTModel is not None and SrcRec is not None:
        return
    try:
        from pytomoatt.model import ATTModel as _ATTModel
        from pytomoatt.src_rec import SrcRec as _SrcRec
    except ImportError:
        print("ERROR: PyTomoATT is required only when generating missing input files.")
        print("The existing src_rec_config.dat/model_init_N*.h5 files were not enough for this run.")
        sys.exit(1)
    ATTModel = _ATTModel
    SrcRec = _SrcRec


class BuildInitialModel:
    def __init__(self, par_file="./3_input_params/input_params_signal.yaml", output_dir="2_models"):
        self.am = ATTModel(par_file)
        self.output_dir = output_dir

    def build_initial_model(self, vel_min=2.5, vel_max=5.0):
        self.am.vel[self.am.depths < 0, :, :] = vel_min
        idx = np.where((0 <= self.am.depths) & (self.am.depths < 40.0))[0]
        self.am.vel[idx, :, :] = np.linspace(vel_min, vel_max, idx.size)[::-1][:, np.newaxis, np.newaxis]
        self.am.vel[self.am.depths >= 40.0, :, :] = vel_max


def _sinusoid_3d(lat_rel, lon_rel, dep_rel, latlon_period_deg):
    return (
        np.sin(np.pi * lat_rel / latlon_period_deg)
        * np.sin(np.pi * lon_rel / latlon_period_deg)
        * np.sin(np.pi * dep_rel / DEPTH_PERIOD_KM)
    )


def build_paper_checkerboard(init_model_path: Path, out_path: Path) -> None:
    with h5py.File(init_model_path, "r") as src:
        lon = np.asarray(src["lon"][...], dtype=float)
        lat = np.asarray(src["lat"][...], dtype=float)
        dep = np.asarray(src["dep"][...], dtype=float)
        vel0 = np.asarray(src["vel"][...], dtype=float)
        xi0 = np.asarray(src["xi"][...], dtype=float)
        eta0 = np.asarray(src["eta"][...], dtype=float)
        zeta0 = np.asarray(src["zeta"][...], dtype=float) if "zeta" in src else np.zeros_like(vel0)

        lat_rel = lat[np.newaxis, :, np.newaxis] - float(np.min(lat))
        lon_rel = lon[np.newaxis, np.newaxis, :] - float(np.min(lon))
        dep_grid = dep[:, np.newaxis, np.newaxis]
        dep_rel = np.maximum(dep_grid, 0.0)
        max_perturbation_depth = float(np.nanmax(dep)) - BOTTOM_BUFFER_KM
        inside_model = (dep_grid >= 0.0) & (dep_grid <= max_perturbation_depth)

        dv_frac = PERTURBATION_FRACTION * _sinusoid_3d(lat_rel, lon_rel, dep_rel, VELOCITY_PERIOD_DEG)
        dv_frac = np.where(inside_model, dv_frac, 0.0)
        vel = vel0 * (1.0 + dv_frac)

        epsilon = PERTURBATION_FRACTION * _sinusoid_3d(lat_rel, lon_rel, dep_rel, ANISOTROPY_PERIOD_DEG)
        epsilon = np.where(inside_model, epsilon, 0.0)
        fast_azimuth_deg = np.where(epsilon < 0.0, FAST_AZIMUTH_NEG_DEG, FAST_AZIMUTH_POS_DEG)
        two_psi = np.deg2rad(2.0 * fast_azimuth_deg)
        xi = epsilon * np.cos(two_psi)
        eta = epsilon * np.sin(two_psi)

        if out_path.exists():
            out_path.unlink()

        with h5py.File(out_path, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value

            for key in src.keys():
                data = src[key][...]
                if key == "vel":
                    data = vel
                elif key == "xi":
                    data = xi
                elif key == "eta":
                    data = eta
                elif key == "zeta":
                    data = zeta0
                elif key == "lon":
                    data = lon
                elif key == "lat":
                    data = lat
                elif key == "dep":
                    data = dep
                dst.create_dataset(key, data=data)

    print(f"Wrote paper-style checkerboard model to {out_path}")
    print(
        "Checkerboard periods: "
        f"velocity={VELOCITY_PERIOD_DEG:.3f} deg, "
        f"anisotropy={ANISOTROPY_PERIOD_DEG:.3f} deg, "
        f"depth={DEPTH_PERIOD_KM:.3f} km; "
        f"bottom buffer={BOTTOM_BUFFER_KM:.1f} km"
    )
    print(
        "Velocity perturbation range: "
        f"{float(np.nanmin(dv_frac)) * 100.0:.2f}% to {float(np.nanmax(dv_frac)) * 100.0:.2f}%"
    )
    epsilon_mag = np.sqrt(xi * xi + eta * eta)
    print(
        "Anisotropy magnitude range: "
        f"{float(np.nanmin(epsilon_mag)) * 100.0:.2f}% to {float(np.nanmax(epsilon_mag)) * 100.0:.2f}%"
    )


if __name__ == "__main__":
    url = "https://zenodo.org/records/14053821/files/src_rec_config.dat"
    src_rec_path = Path("1_src_rec_files/src_rec_config.dat")
    src_rec_path.parent.mkdir(parents=True, exist_ok=True)
    if not src_rec_path.exists():
        require_pytomoatt()
        sr = SrcRec.read(url)
        sr.write(str(src_rec_path))

    output_dir = Path("2_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    init_model = next(output_dir.glob("model_init_N*.h5"), None)
    if init_model is None:
        require_pytomoatt()
        bim = BuildInitialModel(output_dir=str(output_dir))
        bim.build_initial_model()
        init_model = output_dir / f"model_init_N{bim.am.n_rtp[0]}_{bim.am.n_rtp[1]}_{bim.am.n_rtp[2]}.h5"
        bim.am.write(str(init_model))
    else:
        print(f"Initial model already exists at {init_model}; preserving it.")

    build_paper_checkerboard(init_model, output_dir / "model_ckb_N55_61_61.h5")
