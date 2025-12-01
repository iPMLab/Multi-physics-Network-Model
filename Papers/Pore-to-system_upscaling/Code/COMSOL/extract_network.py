import sys

sys.path.append("../../../../")
import h5py
from mpnm_new import extraction, network as net, util
import numpy as np
import numba as nb
from Papers.P1.Code.COMSOL.comsol_params import (
    PARAMS_N2_500,
    PARAMS_N3_455,
    PARAMS_N4_353,
    PARAMS_N4_689,
    PARAMS_N4_869,
    PARAMS_N5_000,
    PARAMS_N10_000,
    PARAMS_N5_000_marching_cube,
    PARAMS_N5_000_voxel,
    PARAMS_N5_000_constrained_smooth,
    PARAMS_N5_000_minkowski,
)

import time


def extract_network(PARAM):
    PARAM = PARAM()
    # print(PARAM.prefix)
    Path_binary_raw = PARAM.Path_binary_raw
    raw_shape = PARAM.raw_shape
    resolution = PARAM.resolution
    Path_mix_raw = PARAM.Path_mix_raw
    Path_net_dual = PARAM.Path_net_dual
    Path_net_pore = PARAM.Path_net_pore
    Path_net_solid = PARAM.Path_net_solid
    image = np.fromfile(
        Path_binary_raw,
        dtype=np.uint8,
    ).reshape(raw_shape)

    config_0 = {
        "method": "pne",
        "target_value": 0,
        "n_workers": 32,
        "resolution": resolution,
        "mode": "image",
    }
    # config_0 = {"method": "snow", "target_value": 0,"r_max":5}
    # config_1 = {"method": "snow", "target_value": 1}
    config_1 = {
        "method": "pne",
        "target_value": 1,
        "n_workers": 32,
        "resolution": resolution,
        "mode": "image",
    }

    config_list = [config_0, config_1]

    image_mix, nets, seps = extraction.dualn_phase_extraction(
        image=image,
        resolution=resolution,
        config_list=config_list,
        n_workers_segmentation=len(config_list),
        n_workers_extraction=61,
        backend="loky",
    )
    Path_net_dual.parent.mkdir(exist_ok=True, parents=True)
    Path_mix_raw.parent.mkdir(exist_ok=True, parents=True)
    dualn, pn, sn = nets
    # print(pn["pore.radius"].sum())
    # SAVE
    # image_mix.tofile(Path_mix_raw)
    net.network2vtk(dualn, Path_net_dual)
    net.network2vtk(pn, Path_net_pore)
    net.network2vtk(sn, Path_net_solid)
    print("pore_num_dual:", dualn["pore.all"].size)
    print("pore_num_pn:", pn["pore.all"].size)
    print("pore_num_sn:", sn["pore.all"].size)


##### pypne == 0.0.26
PARAMS = [
    # *PARAMS_N2_500,
    # *PARAMS_N3_455,
    # *PARAMS_N4_353,
    # *PARAMS_N4_689,
    # *PARAMS_N4_869,
    # *PARAMS_N5_000,
    *PARAMS_N5_000_minkowski,
    # *PARAMS_N5_000_voxel,
    # *PARAMS_N5_000_marching_cube,
    # *PARAMS_N5_000_constrained_smooth,
    # *PARAMS_N10_000,
]

# extract_network(PARAMS[0])

for PARAM in PARAMS:
    t0 = time.time()
    extract_network(PARAM)
    print(PARAM.prefix, "time:", time.time() - t0)

# 19
# 25
# 26
# 27
# 28
