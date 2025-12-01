import numpy as np
from pathlib import Path
import h5py
from Papers.P1.Code.COMSOL.comsol_params import PARAMS_alpha140


def get_binary_image(PARAM):
    print(PARAM.prefix)
    comsol_params_map = PARAM.comsol_params_map

    Path_binary_raw = PARAM.Path_binary_raw

    with h5py.File(PARAM.Path_data_h5, "r") as f:
        hf_keys = [k for k in f.keys()]
        print(hf_keys)
        P_arr = f[hf_keys[0]][comsol_params_map["p"]]

    Path_binary_raw.parent.mkdir(exist_ok=True, parents=True)
    binary_raw = np.where(np.isnan(P_arr), 1, 0)
    binary_raw.astype(np.uint8).tofile(Path_binary_raw)


PARAMS = PARAMS_alpha140
for PARAM in PARAMS:
    get_binary_image(PARAM)
