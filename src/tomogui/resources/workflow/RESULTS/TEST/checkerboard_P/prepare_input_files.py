# download src_ref_files from Zenodo
import argparse
import json
import os
from pathlib import Path
import numpy as np
import sys
import yaml
try:
    from pytomoatt.model import ATTModel
    from pytomoatt.checkerboard import Checker
    from pytomoatt.src_rec import SrcRec
except:
    print("ERROR: ATTModel not found. Please install pytomoatt first."
          "See https://tomoatt.github.io/PyTomoATT/installation.html for details.")
    sys.exit(1)


class BuildInitialModel():
    def __init__(self, par_file="./3_input_params/input_params_signal.yaml", output_dir="2_models"):
        """
        Build initial model for tomography inversion
        """
        self.am = ATTModel(par_file)
        self.output_dir = output_dir

    def build_initial_model(self, vel_min=5.0, vel_max=8.0):
        """
        Build initial model for tomography inversion
        """
        self.am.vel[self.am.depths < 0, :, :] = vel_min
        idx = np.where((0 <= self.am.depths) & (self.am.depths < 40.0))[0]
        #idx = np.where((0 <= self.am.depths) & (self.am.depths < 20.0))[0]
        self.am.vel[idx, :, :] = np.linspace(vel_min, vel_max, idx.size)[::-1][:, np.newaxis, np.newaxis]
        self.am.vel[self.am.depths >= 40.0, :, :] = vel_max


CHECKERBOARD_EDGE_INSET_XY = 1.0e-3
CHECKERBOARD_EDGE_INSET_Z = 1.0e-3


def _base_checkerboard_defaults():
    return {
        "n_pert_x": 2,
        "n_pert_y": 2,
        "n_pert_z": 2,
        "pert_vel": 0.2,
        "pert_ani": 0.1,
        "ani_dir": 60.0,
    }


def _load_domain_limits(par_file="./3_input_params/input_params_signal.yaml"):
    data = yaml.safe_load(Path(par_file).read_text(encoding="utf-8"))
    domain = data["domain"]
    lon = [float(domain["min_max_lon"][0]), float(domain["min_max_lon"][1])]
    lat = [float(domain["min_max_lat"][0]), float(domain["min_max_lat"][1])]
    dep = [float(domain["min_max_dep"][0]), float(domain["min_max_dep"][1])]
    return lon, lat, dep


def _inset_interval(limits, domain_limits, inset):
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


def _default_checkerboard_config(par_file="./3_input_params/input_params_signal.yaml"):
    config = _base_checkerboard_defaults()
    lon_domain, lat_domain, dep_domain = _load_domain_limits(par_file)
    dep_requested = [max(0.0, dep_domain[0]), max(0.0, dep_domain[1])]
    config["lim_x"] = _inset_interval(lon_domain, lon_domain, CHECKERBOARD_EDGE_INSET_XY)
    config["lim_y"] = _inset_interval(lat_domain, lat_domain, CHECKERBOARD_EDGE_INSET_XY)
    config["lim_z"] = _inset_interval(dep_requested, dep_domain, CHECKERBOARD_EDGE_INSET_Z)
    return config


def build_ckb_model(output_dir="2_models"):
    config = _default_checkerboard_config("./3_input_params/input_params_signal.yaml")
    cbk = Checker(f'{output_dir}/model_init_N61_61_61.h5', para_fname="./3_input_params/input_params_signal.yaml")
    cbk.checkerboard(
        n_pert_x=config["n_pert_x"],
        n_pert_y=config["n_pert_y"],
        n_pert_z=config["n_pert_z"],
        pert_vel=config["pert_vel"],
        pert_ani=config["pert_ani"],
        ani_dir=config["ani_dir"],
        lim_x=config["lim_x"],
        lim_y=config["lim_y"],
        lim_z=config["lim_z"],
    )
    out_path = f'{output_dir}/model_ckb_N61_61_61.h5'
    cbk.write(out_path)
    return out_path


def _parse_list(text, expected_len=None, item_type=float):
    items = [item_type(v.strip()) for v in str(text).split(",") if v.strip() != ""]
    if expected_len is not None and len(items) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(items)} in '{text}'.")
    return items


def _load_yaml(text):
    try:
        from ruamel.yaml import YAML
        yaml = YAML(typ="safe")
        return yaml.load(text)
    except Exception:
        try:
            import yaml
            return yaml.safe_load(text)
        except Exception as exc:
            raise ValueError("Failed to parse YAML; use JSON instead.") from exc


def _load_checkerboard_configs(path):
    raw = Path(path).read_text(encoding="utf-8")
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        data = json.loads(raw)
    elif suffix in {".yml", ".yaml"}:
        data = _load_yaml(raw)
    else:
        data = json.loads(raw)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Checkerboard config must be a list or dict.")
    return data


def _normalize_checkerboard_config(config):
    merged = _default_checkerboard_config("./3_input_params/input_params_signal.yaml")
    merged.update(config or {})
    if "n_pert" in merged:
        n_pert = merged.pop("n_pert")
        if isinstance(n_pert, str):
            n_pert = _parse_list(n_pert, expected_len=3, item_type=int)
        if len(n_pert) != 3:
            raise ValueError("n_pert must have three values.")
        merged["n_pert_x"], merged["n_pert_y"], merged["n_pert_z"] = [int(v) for v in n_pert]
    for key in ("n_pert_x", "n_pert_y", "n_pert_z"):
        merged[key] = int(merged[key])
    for key in ("pert_vel", "pert_ani", "ani_dir"):
        merged[key] = float(merged[key])
    for key in ("lim_x", "lim_y", "lim_z"):
        value = merged[key]
        if isinstance(value, str):
            value = _parse_list(value, expected_len=2, item_type=float)
        if len(value) != 2:
            raise ValueError(f"{key} must have two values.")
        merged[key] = [float(value[0]), float(value[1])]
    return merged


def _build_ckb_models(output_dir, n_rtp, configs):
    model_init_path = f'{output_dir}/model_init_N{n_rtp[0]}_{n_rtp[1]}_{n_rtp[2]}.h5'
    results = []
    for idx, config in enumerate(configs, start=1):
        tag = config.get("tag")
        if not tag and len(configs) > 1:
            tag = f"ckb{idx:02d}"
        cbk = Checker(model_init_path, para_fname="./3_input_params/input_params_signal.yaml")
        cbk.checkerboard(
            n_pert_x=config["n_pert_x"],
            n_pert_y=config["n_pert_y"],
            n_pert_z=config["n_pert_z"],
            pert_vel=config["pert_vel"],
            pert_ani=config["pert_ani"],
            ani_dir=config["ani_dir"],
            lim_x=config["lim_x"],
            lim_y=config["lim_y"],
            lim_z=config["lim_z"],
        )
        if tag:
            filename = f'model_ckb_{tag}_N{n_rtp[0]}_{n_rtp[1]}_{n_rtp[2]}.h5'
        else:
            filename = f'model_ckb_N{n_rtp[0]}_{n_rtp[1]}_{n_rtp[2]}.h5'
        out_path = f'{output_dir}/{filename}'
        cbk.write(out_path)
        results.append(out_path)
    return results


def _write_checkerboard_list(path, entries):
    list_path = Path(path)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(entries) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare input files and checkerboard models.")
    parser.add_argument("--checkerboards", help="Path to JSON/YAML checkerboard config file.")
    parser.add_argument("--list-file", help="Write checkerboard model paths to this file.")
    parser.add_argument("--n-pert", default="2,2,2", help="Checkerboard perturbations (x,y,z).")
    parser.add_argument("--pert-vel", type=float, default=0.2, help="Velocity perturbation amplitude.")
    parser.add_argument("--pert-ani", type=float, default=0.1, help="Anisotropy perturbation amplitude.")
    parser.add_argument("--ani-dir", type=float, default=60.0, help="Anisotropy direction (degrees).")
    parser.add_argument("--lim-x", default="", help="Checkerboard longitude limits (min,max). Default: domain range inset slightly from the edges.")
    parser.add_argument("--lim-y", default="", help="Checkerboard latitude limits (min,max). Default: domain range inset slightly from the edges.")
    parser.add_argument("--lim-z", default="", help="Checkerboard depth limits (min,max). Default: 0..max-depth inset slightly from the upper edge.")
    parser.add_argument("--tag", help="Optional tag for single checkerboard output.")
    args = parser.parse_args()

    # download src_rec_config.dat
    url = 'https://zenodo.org/records/14053821/files/src_rec_config.dat'
    path = "1_src_rec_files/src_rec_config.dat"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        sr = SrcRec.read(url)
        sr.write(path)

    # build initial model
    output_dir = "2_models"
    os.makedirs(output_dir, exist_ok=True)
    bim = BuildInitialModel(output_dir=output_dir)
    existing_model = next(Path(output_dir).glob("model_init_N*.h5"), None)
    if existing_model is None:
        bim.build_initial_model()
        bim.am.write('{}/model_init_N{:d}_{:d}_{:d}.h5'.format(bim.output_dir, *bim.am.n_rtp))
    else:
        print(f"Initial model already exists at {existing_model}; preserving it.")

    # build ckb model(s)
    default_config = _default_checkerboard_config("./3_input_params/input_params_signal.yaml")
    if args.checkerboards:
        raw_configs = _load_checkerboard_configs(args.checkerboards)
        configs = [_normalize_checkerboard_config(cfg) for cfg in raw_configs]
        outputs = _build_ckb_models(output_dir, bim.am.n_rtp, configs)
    elif len(sys.argv) == 1:
        outputs = [build_ckb_model(output_dir)]
    else:
        configs = [
            _normalize_checkerboard_config(
                {
                    "n_pert": _parse_list(args.n_pert, expected_len=3, item_type=int),
                    "pert_vel": args.pert_vel,
                    "pert_ani": args.pert_ani,
                    "ani_dir": args.ani_dir,
                    "lim_x": _parse_list(args.lim_x, expected_len=2, item_type=float) if args.lim_x else default_config["lim_x"],
                    "lim_y": _parse_list(args.lim_y, expected_len=2, item_type=float) if args.lim_y else default_config["lim_y"],
                    "lim_z": _parse_list(args.lim_z, expected_len=2, item_type=float) if args.lim_z else default_config["lim_z"],
                    "tag": args.tag,
                }
            )
        ]
        outputs = _build_ckb_models(output_dir, bim.am.n_rtp, configs)
    if args.list_file or len(outputs) > 1:
        list_path = args.list_file or f"{output_dir}/checkerboards.list"
        _write_checkerboard_list(list_path, outputs)
