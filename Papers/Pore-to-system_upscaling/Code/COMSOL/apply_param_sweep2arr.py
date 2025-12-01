from pathlib import Path
import numpy as np
import pandas as pd
import numba as nb
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from Papers.P1.Code.COMSOL.comsol_params import (num_params, params, params_map, img_shape, num_void,
                           num_solid, num_pore, u_inlets, Path_data_root,
                           Path_data_xyz, Path_raw)

Path_xyz = Path_data_xyz
print(np.round(u_inlets, 5))
coords = pd.read_csv(Path_data_root / "xyz.csv",
                     engine='pyarrow').loc[:, ["x", "y", "z"]].to_numpy()
suffix = ""
img_label = np.fromfile(Path_raw, dtype=np.int32).reshape(img_shape)
img_label = np.where(img_label < 0, 0, img_label)
P_void_bool = (pd.read_csv(Path_data_root /
                           f"Finney_6000_u{u_inlets[0]:.5f}.csv",
                           engine='pyarrow').loc[:, "P"].to_numpy() != -1)
P_solid_bool = ~P_void_bool
void_index = np.where(P_void_bool)[0]
kdtree_void = cKDTree(coords[void_index])

image_label_ravel = img_label.reshape(-1)
raw_nan_void = np.where((0 < image_label_ravel)
                        & (image_label_ravel <= num_void) & (P_solid_bool))[0]
map_index = np.concatenate(
    (
        raw_nan_void.reshape(1, -1),
        void_index[kdtree_void.query(coords[raw_nan_void], k=1,
                                     workers=-1)[1]].reshape(1, -1),
    ),
    axis=0,
)

for i, u_inlet in enumerate(u_inlets):
    print(f"Processing u_inlet={u_inlet:.5f}")
    df_i = pd.read_csv(Path_data_root / f"Finney_6000_u{u_inlet:.5f}.csv",
                       engine='pyarrow')
    data_i = np.empty((num_params, *img_shape))
    for param in params:
        param_i = df_i.loc[:, param].to_numpy().reshape(-1)
        param_i[map_index[0]] = param_i[map_index[1]]
        data_i[params_map[param]] = param_i.reshape(img_shape)
    np.savez_compressed(Path_data_root / f"Finney_6000_u{u_inlet:.5f}.npz",
                        arr=data_i)