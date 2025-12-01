import numpy as np
import numexpr as ne
import sys
import numba as nb

sys.path.append("../../")
import timeit
from mpnm_new.extraction._extraction_numba import nb_binary_dilation
from mpnm_new.util._utils_numba import nb_unique_uint
from mpnm_new.util import unique_rows, find_throat_conns_map
from mpnm_new import network as net
from skimage.measure import regionprops_table
from joblib import Parallel, delayed
from tqdm import tqdm
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_1_8_0 as PARAM, extract_u_hf
import h5py
import pandas as pd
import pickle

PARAM = PARAM()
d3q6_directions = np.array(
    (
        # (0, 0, 0),  # 静止粒子
        # 6个面方向（速度模=1）
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
)


@nb.njit(fastmath=True, cache=True, nogil=True, parallel=False, error_model="numpy")
def nb_find_different_neighbors_3d(labeled_region):
    """第二次遍历：填充预分配数组"""
    nz, ny, nx = labeled_region.shape
    self_coords = np.zeros((nz, ny, nx, 3), dtype=np.int64)
    neighbor_coords = np.zeros((nz, ny, nx, 3), dtype=np.int64)
    self_label = np.zeros((nz, ny, nx), dtype=labeled_region.dtype)
    neighbor_label = np.zeros((nz, ny, nx), dtype=labeled_region.dtype)
    for z in nb.prange(nz):
        for y in nb.prange(ny):
            for x in nb.prange(nx):
                current = labeled_region[z, y, x]
                if current > 0:
                    for i, (dz, dy, dx) in enumerate(d3q6_directions):
                        z_, y_, x_ = z + dz, y + dy, x + dx

                        # 确定邻居标签
                        if not (0 <= z_ < nz and 0 <= y_ < ny and 0 <= x_ < nx):
                            continue

                        neighbor = labeled_region[z_, y_, x_]

                        # 记录通量
                        if neighbor != current and neighbor > 0:
                            self_coords[z, y, x, 0] = z
                            self_coords[z, y, x, 1] = y
                            self_coords[z, y, x, 2] = x
                            neighbor_coords[z, y, x, 0] = z_
                            neighbor_coords[z, y, x, 1] = y_
                            neighbor_coords[z, y, x, 2] = x_
                            self_label[z, y, x] = current
                            neighbor_label[z, y, x] = neighbor

    return self_coords, neighbor_coords, self_label, neighbor_label


def find_different_neighbors_3d(arr, resolution=1, offset=0.5, sep=2):
    self_coords, neighbor_coords, self_label, neighbor_label = (
        nb_find_different_neighbors_3d(arr)
    )
    self_label = self_label.reshape(-1)
    neighbor_label = neighbor_label.reshape(-1)
    self_coords = self_coords.reshape(-1, 3)
    neighbor_coords = neighbor_coords.reshape(-1, 3)
    valid_bool = self_label > 0

    self_coords = self_coords[valid_bool]
    neighbor_coords = neighbor_coords[valid_bool]
    self_label = self_label[valid_bool]
    neighbor_label = neighbor_label[valid_bool]
    condition = ((self_label < sep) & (neighbor_label >= sep)) | (
        (self_label >= sep) & (neighbor_label < sep)
    )

    self_coords = self_coords[condition]
    neighbor_coords = neighbor_coords[condition]

    coords = np.concatenate((self_coords, neighbor_coords), axis=0)
    coords = unique_rows(coords)

    return (coords + offset) * resolution


a = np.zeros(27, dtype=int).reshape(3, 3, 3) - 1
a[0, 0, 0] = 1
a[0, 0, 1] = 2
a[0, 0, 2] = 1

print(find_different_neighbors_3d(a))


Path_data_h5 = PARAM.Path_data_h5
Path_comsol = PARAM.Path_comsol
comsol_params_map = PARAM.comsol_params_map
Path_mix_raw = PARAM.Path_mix_raw
raw_shape = PARAM.raw_shape
num_void = PARAM.num_void
Path_PNdata = PARAM.Path_PNdata
Path_interface_coords = PARAM.Path_interface_coords
resolution = PARAM.resolution

labeled_image = np.fromfile(Path_mix_raw, dtype=np.int32).reshape(raw_shape)
# labeled_image = np.where(
#     (labeled_image < 1) | (labeled_image > num_void), -1, labeled_image
# )
num_pore = PARAM.num_pore
labeled_image = np.where((labeled_image < 1), -1, labeled_image)
coords = find_different_neighbors_3d(
    labeled_image, resolution=resolution, offset=0.5, sep=num_void + 1
)

coords[:, 0], coords[:, 2] = coords[:, 2], coords[:, 0]

np.savetxt(Path_interface_coords, coords)
# if not Path_PNdata.exists():
#     PNdata = {}
# else:
#     PNdata = pickle.load(open(Path_PNdata, "rb"))
# dualn = net.vtk2network(PARAM.Path_net_dual)
# throat_conns = dualn["throat.conns"]
# throat_void = dualn["throat.void"]
# throat_solid = dualn["throat.solid"]
# throat_connect = dualn["throat.connect"]
# pore_surface_all = dualn["pore.surface_all"]
# throat_conns = np.sort(throat_conns, axis=1)
