import numba as nb
import numpy as np
import pandas as pd
import h5py
from Papers.P1.Code.COMSOL.comsol_params import extract_Re_hf
import pickle
from Papers.P1.Code.COMSOL.comsol_params import (
    PARAMS_N5_000_marching_cube,
    PARAMS_N5_000_constrained_smooth,
    PARAMS_N5_000_voxel
)


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def calculate_min_max_ave(image, physical_fields):
    image = image.reshape(-1)
    physical_fields = physical_fields.reshape(-1)
    labels_count = np.bincount(image)
    min_max_sum = np.zeros((3, labels_count.size), dtype=np.float64)
    min_max_sum[0] = np.inf
    min_max_sum[1] = -np.inf
    for i in range(physical_fields.size):
        label_i = image[i]
        min_max_sum[0, label_i] = min(min_max_sum[0, label_i], physical_fields[i])
        min_max_sum[1, label_i] = max(min_max_sum[1, label_i], physical_fields[i])
        min_max_sum[2, label_i] += physical_fields[i]

    min_max_ave = min_max_sum
    min_max_ave[2] /= labels_count
    return min_max_ave


def get_min_max_ave(PARAM):
    comsol_param_map = PARAM.comsol_params_map
    Path_mix_raw = PARAM.Path_mix_raw
    Path_data_h5 = PARAM.Path_data_h5
    img_shape = PARAM.raw_shape
    num_void = PARAM.num_void
    num_pore = PARAM.num_void + PARAM.num_solid
    Path_comsol = PARAM.Path_comsol
    params_fluid = PARAM.comsol_params_fluid

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

            for param in params2cal:
                min_max_ave_i = calculate_min_max_ave(
                    image, data_i[comsol_param_map[param]]
                )
                min_max_ave_i = min_max_ave_i.astype(np.float32)[:, 1:]
                if param in params_fluid:
                    min_max_ave_i[:, image_label_solid_bool] = 0
                PNdata_data_i[f"pore.{param}_min_comsol"] = min_max_ave_i[0]
                PNdata_data_i[f"pore.{param}_max_comsol"] = min_max_ave_i[1]
                PNdata_data_i[f"pore.{param}_ave_comsol"] = min_max_ave_i[2]
            PNdata[PNdata_name_i] = PNdata_data_i
        pickle.dump(PNdata, open(Path_PNdata, "wb"))


PARAMS = PARAMS_N5_000_constrained_smooth
for PARAM in PARAMS:
    get_min_max_ave(PARAM)
