from pathlib import Path
import numpy as np
import pandas as pd
import numba as nb
from scipy.spatial import cKDTree
import pickle
from tqdm import tqdm
import numba as nb
import matplotlib.pyplot as plt
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_N10_000_sample0 as PARAM, extract_u_hf
import h5py

from joblib import Parallel, delayed

image_mix = PARAM.Path_mix_raw
image_label_ravel = np.fromfile(image_mix, dtype=np.int32).reshape(-1)

Path_data_h5 = PARAM.Path_data_h5
pores = PARAM.num_pore
comsol_param_map = PARAM.comsol_params_map


@nb.njit(cache=True, nogil=True, fastmath=True)
def each_pore_value(indices, X_i):
    return X_i[indices]


@nb.njit(cache=True, parallel=True, nogil=True, fastmath=True)
def each_pore_index(pore, image_label_ravel):
    return np.where(image_label_ravel == pore)[0]


Indices_pore = Parallel(n_jobs=-1)(
    delayed(each_pore_index)(pore, image_label_ravel)
    for pore in tqdm(range(1, pores + 1))
)


with h5py.File(Path_data_h5, "r", swmr=True) as f:
    for key in f.keys():
        print(key)
        data_i = f[key][()]
        # _u, _hf = extract_u_hf(key, _u_astype=np.float32, _hf_astype=np.int64)
        P_i = data_i[comsol_param_map["p"]].reshape(-1)
        U_i = data_i[comsol_param_map["spf.U"]].reshape(-1)
        T_i = data_i[comsol_param_map["T"]].reshape(-1)

        P_i_pore = Parallel(n_jobs=-1)(
            delayed(each_pore_value)(indices, P_i) for indices in tqdm(Indices_pore)
        )
        U_i_pore = Parallel(n_jobs=-1)(
            delayed(each_pore_value)(indices, U_i) for indices in tqdm(Indices_pore)
        )
        T_i_pore = Parallel(n_jobs=-1)(
            delayed(each_pore_value)(indices, T_i) for indices in tqdm(Indices_pore)
        )
        dict_i = {
            pore: {"p": P_i_pore[pore], "spf.U": U_i_pore[pore], "T": T_i_pore[pore]}
            for pore in tqdm(range(pores))
        }
        pickle.dump(dict_i, open(Path_data_h5.parent / f"Poredata{key}.pkl", "wb"))
