import sys

sys.path.append("../../../")
import numpy as np
from mpnm_new import topotool, algorithm, network as net, util
import pandas as pd
from Papers.P1.Code.COMSOL.comsol_params import extract_Re_hf
import pickle
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict
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
)

PARAM = PARAMS_N5_000[0]


dualn = net.vtk2network(PARAM.Path_net_dual)
volume_solid = dualn["pore.volume"][dualn["pore.solid"]]
volume_solid = np.repeat(volume_solid, 5)
volume_pore = dualn["pore.volume"][dualn["pore.void"]]
volume_pore = np.repeat(volume_pore, 5)

# 找到最大长度
max_len = max(len(volume_solid), len(volume_pore))


# 补全到相同长度（用 NaN）
def pad_with_nan(arr, length):
    if len(arr) < length:
        return np.concatenate([arr, np.full(length - len(arr), np.nan)])
    else:
        return arr  # 或者也可以截断：arr[:length]


volume_solid_padded = pad_with_nan(volume_solid, max_len)
volume_pore_padded = pad_with_nan(volume_pore, max_len)

# 构建 DataFrame
df = pd.DataFrame(
    {"volume_solid": volume_solid_padded, "volume_pore": volume_pore_padded}
)

df.to_csv(f"{PARAM.prefix}_volume.csv", index=False)
