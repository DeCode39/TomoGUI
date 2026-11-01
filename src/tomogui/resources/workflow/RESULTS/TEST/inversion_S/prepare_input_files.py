# download src_ref_files from Zenodo
import os
import shutil
from pathlib import Path
import numpy as np
import sys
try:
    from pytomoatt.model import ATTModel
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

    def build_initial_model(self, vel_min=2.5, vel_max=5.0):
        """
        Build initial model for tomography inversion
        """
        self.am.vel[self.am.depths < 0, :, :] = vel_min
        idx = np.where((0 <= self.am.depths) & (self.am.depths < 40.0))[0]
        self.am.vel[idx, :, :] = np.linspace(vel_min, vel_max, idx.size)[::-1][:, np.newaxis, np.newaxis]
        self.am.vel[self.am.depths >= 40.0, :, :] = vel_max
if __name__ == "__main__":
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
    existing_model = next(Path(output_dir).glob("model_init_N*.h5"), None)
    if existing_model is None:
        bim = BuildInitialModel(output_dir=output_dir)
        bim.build_initial_model()
        bim.am.write('{}/model_init_N{:d}_{:d}_{:d}.h5'.format(bim.output_dir, *bim.am.n_rtp))
    else:
        print(f"Initial model already exists at {existing_model}; preserving it.")

    ckb_src = Path(__file__).resolve().parents[3] / "TOMOATT/examples/eg3_joint_inversion/2_models/model_ckb_N61_61_61.h5"
    ckb_dst = Path(output_dir) / "model_ckb_N61_61_61.h5"
    if ckb_src.exists() and not ckb_dst.exists():
        shutil.copy2(ckb_src, ckb_dst)
