import sys

sys.path.append("../../")


import numba as nb
import numpy as np
import pandas as pd
import h5py
from Papers.P1.Code.COMSOL.comsol_params import extract_Re_hf
import pickle
from mpnm_new import network as net
from Papers.P1.Code.COMSOL.comsol_params import (
    PARAMS_N5_000_marching_cube
)


def get_pore_center_velocity(PARAM):
    PARAM = PARAM()
    comsol_param_map = PARAM.comsol_params_map
    Path_mix_raw = PARAM.Path_mix_raw
    Path_data_h5 = PARAM.Path_data_h5
    img_shape = PARAM.raw_shape
    num_void = PARAM.num_void
    num_pore = PARAM.num_pore

    Path_net_dual = PARAM.Path_net_dual

    resolution = PARAM.resolution
    dualn = net.vtk2network(Path_net_dual)
    dualn_pore_coords = np.round(dualn["pore.coords"] / resolution).astype(int)
    params2cal = ("p", "T", "spf.U")
    image = np.fromfile(Path_mix_raw, dtype=np.int32).reshape(img_shape)
    image = np.where(image < 0, 0, image)
    image_label_solid_bool = np.zeros(num_pore, dtype=bool)
    image_label_solid_bool[num_void:] = True
    Path_PNdata = PARAM.Path_PNdata

    if not Path_PNdata.exists():
        PNdata = {}
    else:
        PNdata = pickle.load(open(Path_PNdata, "rb"))
    with h5py.File(Path_data_h5, "r", swmr=True) as f:
        for key in f.keys():
            print(key)
            data_i = f[key]
            _Re, _hf = extract_Re_hf(key, _Re_astype=np.float32, _hf_astype=np.int64)

            PNdata_name_i = f"_Re{_Re:.5f}_hf{_hf}"
            PNdata_data_i = PNdata.get(PNdata_name_i, {})
            for param_i in params2cal:
                param_i_data = data_i[comsol_param_map[param_i]]
                param_i_data_pore_centre = param_i_data[
                    dualn_pore_coords[:, 2],
                    dualn_pore_coords[:, 1],
                    dualn_pore_coords[:, 0],
                ]
                param_i_data_pore_centre = np.nan_to_num(
                    param_i_data_pore_centre, nan=0.0
                )
                PNdata_data_i[f"pore.{param_i}_pore_center_comsol"] = (
                    param_i_data_pore_centre
                )
                PNdata[PNdata_name_i] = PNdata_data_i
        pickle.dump(PNdata, open(Path_PNdata, "wb"))


if __name__ == "__main__":
    PARAMS = PARAMS_N5_000_marching_cube
    for PARAM in PARAMS:
        get_pore_center_velocity(PARAM)
