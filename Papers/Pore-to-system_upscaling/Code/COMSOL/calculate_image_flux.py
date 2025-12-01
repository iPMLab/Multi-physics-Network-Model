import sys

sys.path.append("../../")
from mpnm_new import extraction
from skimage.measure import marching_cubes, regionprops_table
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import pandas as pd
import numpy as np
import os
import numba as nb
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm, trange
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_1_8_0
import pyvista as pv
from tqdm import tqdm, trange
import h5py
PARAM = ComsolParams_1_8_0()
raw_shape = PARAM.raw_shape
resolution = PARAM.resolution
params_map = PARAM.comsol_params_map
num_void = PARAM.num_void
num_solid = PARAM.num_solid
num_pore = PARAM.num_pore



@nb.njit(parallel=True, fastmath=True, nogil=True, cache=True)
def nb_calculate_flux(faces, vertices_coord, vector_field):
    """Numba加速的顶点速度通量计算
    Args:
        verts: (N,3)顶点坐标数组
        faces: (M,3)三角形面片顶点索引
        vertex_velocity: (N,3)顶点速度矢量数组
    Returns:
        total_flux_pos: 正向通量
        total_flux_neg: 负向通量
    """
    num_face = faces.shape[0]
    total_flux = np.zeros(num_face, dtype="float64")
    total_flux_pos = 0.0
    total_flux_neg = 0.0
    # 并行遍历所有面片
    for i in nb.prange(num_face):
        # 获取顶点索引

        vertice0_index = faces[i, 0]
        vertice1_index = faces[i, 1]
        vertice2_index = faces[i, 2]

        vertice0_coord = vertices_coord[vertice0_index]
        vertice1_coord = vertices_coord[vertice1_index]
        vertice2_coord = vertices_coord[vertice2_index]

        edge0 = vertice1_coord - vertice0_coord
        edge1 = vertice2_coord - vertice0_coord
        area_vector = 0.5 * np.cross(edge0, edge1)

        # 获取顶点速度并计算面片平均速度
        vector0 = vector_field[vertice0_index]
        vector1 = vector_field[vertice1_index]
        vector2 = vector_field[vertice2_index]
        vector_avg = (vector0 + vector1 + vector2) / 3.0
        # vector_avg = np.sum(vector_field[vertices_indice], axis=0)/3

        # 计算通量
        flux = np.dot(vector_avg, area_vector)
        total_flux[i] = flux
        if flux > 0:
            total_flux_pos += flux
        elif flux < 0:
            total_flux_neg += flux

    return total_flux, total_flux_pos, total_flux_neg


def calculate_flux(
    faces,
    vertices_coord,
    vector_field,
    return_flux=True,
    return_flux_pos=True,
    return_flux_neg=True,
):
    assert vertices_coord.shape[1] == 3
    assert faces.shape[1] == 3
    assert vertices_coord.shape[0] == vector_field.shape[0]
    total_flux, total_flux_pos, total_flux_neg = nb_calculate_flux(
        faces, vertices_coord, vector_field
    )
    res = []
    if return_flux:
        res.append(total_flux)
    if return_flux_pos:
        res.append(total_flux_pos)
    if return_flux_neg:
        res.append(total_flux_neg)

    res = tuple(res)
    return res[0] if len(res) == 1 else res


vertices_coord = np.indices(raw_shape).reshape(3, -1).T
img_label = np.fromfile(PARAM.Path_mix_raw, dtype=np.int32).reshape(raw_shape)
img_label = np.where(img_label < 0, 0, img_label)

kdtree_all = cKDTree(vertices_coord)


def get_slice(zyx, offset=0):
    zyx_min_i = np.clip(zyx[:3] - offset, 0, np.inf).astype(np.int64)
    z_min, y_min, x_min = zyx_min_i
    z_max, y_max, x_max = zyx[3:] + offset
    return (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))


pad = 5
min_coord = 0.5 * resolution
max_coord = 499.5 * resolution
indices_coord = np.arange(500) + 0.5
with h5py.File(PARAM.Path_data_h5, 'r') as f:
    # 查看文件结构
    print("HDF5 文件内容：")
    print(list(f.keys()))  # 打印所有顶层数据集/组
    
    # 读取某个数据集（假设 'data' 是数据集名称）
    if 'data' in f:
        dataset = f['data'][:]  # 读取整个数据集到内存
        print("数据集 shape:", dataset.shape)



for i, u_inlet in tqdm(enumerate(u_inlets), position=0):
    print(f"Processing u_inlet={u_inlet:.5f}")
    data_i = np.load(Path_comsol_data / f"Finney_6000_u{u_inlet:.5f}.npz")["arr"]
    u_i = data_i[params_map["u"]]
    v_i = data_i[params_map["v"]]
    w_i = data_i[params_map["w"]]
    u_i[img_solid_bool] = 0.0
    v_i[img_solid_bool] = 0.0
    w_i[img_solid_bool] = 0.0

    labeled_image = img_label
    phase_props_table = pd.DataFrame(
        regionprops_table(
            label_image=labeled_image, properties=("label", "bbox", "area")
        )
    )
    name_map = {
        "bbox-0": "z_min",
        "bbox-1": "y_min",
        "bbox-2": "x_min",
        "bbox-3": "z_max",
        "bbox-4": "y_max",
        "bbox-5": "x_max",
        "area": "volume",
    }
    interpolator_u = RegularGridInterpolator((indices_coord * resolution,) * 3, u_i)
    interpolator_v = RegularGridInterpolator((indices_coord * resolution,) * 3, v_i)
    interpolator_w = RegularGridInterpolator((indices_coord * resolution,) * 3, w_i)
    tree_u = cKDTree(np.column_stack((indices_coord,) * 3))
    phase_props_table.rename(columns=name_map, inplace=True)
    phase_props_table = phase_props_table.to_numpy(dtype=np.int32)
    df = pd.DataFrame()
    labels = np.zeros(num_pore, dtype=np.int32)
    flux_poses = np.zeros(num_pore)
    flux_negs = np.zeros(num_pore)
    flux_sums = np.zeros(num_pore)
    for i in tqdm(range(num_void), position=1):
        label_i = phase_props_table[i, 0]
        zyx_min_i = phase_props_table[i, 1:4]
        zyx_max_i = phase_props_table[i, 4:7]
        labeled_image_i = labeled_image[
            zyx_min_i[0] : zyx_max_i[0],
            zyx_min_i[1] : zyx_max_i[1],
            zyx_min_i[2] : zyx_max_i[2],
        ]
        labeled_image_i_bool = labeled_image_i == label_i
        labeled_image_i_bool = np.pad(
            labeled_image_i_bool, ((pad, pad),) * 3, "constant", constant_values=False
        )
        labeled_image_i_bool = extraction.constrained_smooth(labeled_image_i_bool)
        verts, faces, _, _ = marching_cubes(labeled_image_i_bool, 0.5)
        mesh = pv.PolyData(verts, np.insert(faces, 0, 3, axis=1))
        mesh.plot()
        verts_original = (verts + pad - 0.5 + zyx_min_i) * resolution
        # verts_original = np.clip(verts_original,
        #                          a_min=min_coord,
        #                          a_max=max_coord)
        verts_original_out_bool = np.any(
            (max_coord < verts_original) | (verts_original < min_coord), axis=1
        )
        verts_original_in_bool = ~verts_original_out_bool
        verts_original_in = verts_original[verts_original_in_bool]
        verts_original_out = verts_original[verts_original_out_bool]
        interpolated_data = np.empty((verts_original.shape[0], 3))
        if np.any(verts_original_out_bool):
            distances, indices = kdtree_all.query(
                verts_original[verts_original_out_bool], k=3, workers=-1
            )
            u_out = np.mean(u_i.reshape(-1)[indices], axis=1)
            v_out = np.mean(v_i.reshape(-1)[indices], axis=1)
            w_out = np.mean(w_i.reshape(-1)[indices], axis=1)
            interpolated_data[verts_original_in_bool] = np.column_stack(
                (
                    interpolator_u(verts_original_in),
                    interpolator_v(verts_original_in),
                    interpolator_w(verts_original_in),
                )
            )
            interpolated_data[verts_original_out_bool] = np.column_stack(
                (u_out, v_out, w_out)
            )
        else:
            interpolated_data = np.column_stack(
                (
                    interpolator_u(verts_original),
                    interpolator_v(verts_original),
                    interpolator_w(verts_original),
                )
            )

        mesh = pv.PolyData(verts_original, np.insert(faces, 0, 3, axis=1))
        mesh.plot()

        total_flux, total_flux_pos, total_flux_neg = calculate_flux(
            faces, verts_original[:, ::-1], interpolated_data
        )
        total_flux_sum = total_flux_pos + total_flux_neg
        tqdm.write(
            f"label={label_i}, total_flux_pos={total_flux_pos},total_flux_neg={total_flux_neg},total_flux_sum={total_flux_sum}"
        )
        labels[i] = label_i
        flux_poses[i] = total_flux_pos
        flux_negs[i] = total_flux_neg
        flux_sums[i] = total_flux_sum

    df["label"] = labels
    df["flux_pos"] = flux_poses
    df["flux_neg"] = flux_negs
    df["flux_sum"] = flux_sums
    df.to_csv(Path_root / f"flux_pos_neg_smoothed_u{u_inlet:.5f}.csv", index=False)
