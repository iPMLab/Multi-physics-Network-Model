import numpy as np
import numba as nb
import pyvista as pv
import sys

sys.path.append("../../")
from mpnm_new.util import unique_rows, find_throat_conns_map
from mpnm_new import network as net
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_1_8_0 as PARAM
import h5py

PARAM = PARAM()

comsol_params_map = {
    "P": 0,
    "T": 1,
    "U": 2,
    "u": 3,
    "v": 4,
    "w": 5,
    "ht.cfluxx": 6,
    "ht.cfluxy": 7,
    "ht.cfluxz": 8,
    "ht.dfluxx": 9,
    "ht.dfluxy": 10,
    "ht.dfluxz": 11,
    "ht.tfluxx": 12,
    "ht.tfluxy": 13,
    "ht.tfluxz": 14,
}


# @nb.njit(fastmath=True, cache=True)
# def nb_get_cell_interface(cells):
#     faces = np.empty((cells.shape[0], 4, 3), dtype=np.int32)
#     for i in nb.prange(cells.shape[0]):
#         cell = cells[i]
#         faces[i, 0, 0] = cell[0]
#         faces[i, 0, 1] = cell[1]
#         faces[i, 0, 2] = cell[2]
#         faces[i, 1, 0] = cell[0]
#         faces[i, 1, 1] = cell[1]
#         faces[i, 1, 2] = cell[3]
#         faces[i, 2, 0] = cell[0]
#         faces[i, 2, 1] = cell[2]
#         faces[i, 2, 2] = cell[3]
#         faces[i, 3, 0] = cell[1]
#         faces[i, 3, 1] = cell[2]
#         faces[i, 3, 2] = cell[3]

#     return faces


def get_cell_interface(cells, cell_voxel):
    assert cells.shape[1] == 4, "cells must have 4 indices per face"
    assert (
        cells.shape[0] == cell_voxel.shape[0]
    ), "cells and cells_data must have the same shape along axis 0"
    cells = np.sort(cells, axis=1)
    # 定义每个面的顶点索引（4个面 × 3个顶点）
    face_indices = np.array(
        [
            [0, 1, 2],  # 面 0
            [0, 1, 3],  # 面 1
            [0, 2, 3],  # 面 2
            [1, 2, 3],  # 面 3
        ],
        dtype=np.int32,
    )

    # 用 broadcasting 提取所有面的顶点（n_cells × 4 × 3）
    faces = cells[:, face_indices].reshape(-1, 3)
    face_voxel = np.repeat(cell_voxel, 4, axis=0).astype(np.int32)
    face_cell = np.repeat(np.arange(cells.shape[0]), 4, axis=0).astype(np.int32)
    # faces = faces.reshape(-1, 3)
    # datas = datas.reshape(-1)

    unique_faces, inverse_indices, counts = unique_rows(
        faces, return_inverse=True, return_counts=True, keepdims=False
    )
    sorted_indices = np.argsort(inverse_indices)
    sorted_face_voxel = face_voxel[sorted_indices]
    sorted_face_cell = face_cell[sorted_indices]

    start_indices = np.empty((counts.size), dtype=np.int32)
    start_indices[0] = 0
    start_indices[1:] = np.cumsum(counts[:-1])

    connection_pairs = np.full((unique_faces.shape[0], 2), -1, dtype=np.int32)
    connection_pairs[:, 0] = sorted_face_voxel[start_indices]

    # 初始化 face_cell_pairs（形状：n_unique_faces × 2）
    face_cell_pairs = np.full((unique_faces.shape[0], 2), -1, dtype=np.int32)
    face_cell_pairs[:, 0] = sorted_face_cell[start_indices]

    # 对 count=2 的面填充第二个体素值和单元索引
    count_is_2 = counts == 2
    connection_pairs[count_is_2, 1] = sorted_face_voxel[start_indices[count_is_2] + 1]
    face_cell_pairs[count_is_2, 1] = sorted_face_cell[start_indices[count_is_2] + 1]

    sorted_indices = np.lexsort((connection_pairs[:, 0], connection_pairs[:, 1]))

    unique_faces = unique_faces[sorted_indices]
    connection_pairs = connection_pairs[sorted_indices]
    face_cell_pairs = face_cell_pairs[sorted_indices]

    return unique_faces, connection_pairs, face_cell_pairs


@nb.njit(fastmath=True, cache=True)
def compute_oriented_normals_nb(points, faces, interface_datas, cell_center):
    """
    计算带方向的法线和面积(Numba加速版)
    确保法线方向始终从左单元指向右单元
    """
    n_faces = faces.shape[0]
    normals = np.zeros((n_faces, 3), dtype=np.float64)
    areas = np.zeros(n_faces, dtype=np.float64)

    for i in range(n_faces):
        # 获取面顶点
        p0 = points[faces[i, 0]]
        p1 = points[faces[i, 1]]
        p2 = points[faces[i, 2]]

        # 计算初始法线
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        area = 0.5 * norm

        # 获取左右单元ID
        left_id = interface_datas[i, 0]
        right_id = interface_datas[i, 1]

        # if left_id != -1 and right_id != -1:  # 只处理内部面
        # 计算左右单元质心
        left_centroid = cell_center[left_id]
        right_centroid = cell_center[right_id]

        # 计算从左到右的向量
        lr_vector = right_centroid - left_centroid

        # 检查法线方向是否需要翻转
        dot_product = (
            normal[0] * lr_vector[0]
            + normal[1] * lr_vector[1]
            + normal[2] * lr_vector[2]
        )
        if dot_product < 0:
            normal = -normal

        normal /= norm

        normals[i] = normal
        areas[i] = area

    return normals, areas


@nb.njit(fastmath=True, cache=True)
def compute_flow_rates_nb(face_velocities, normals, areas):
    """计算通过每个面的流量(Numba加速版)"""
    n_faces = face_velocities.shape[0]
    flow_rates = np.zeros(n_faces, dtype=np.float64)

    for i in range(n_faces):
        # 流量 = 速度·法线 * 面积
        flow_rates[i] = (
            face_velocities[i, 0] * normals[i, 0]
            + face_velocities[i, 1] * normals[i, 1]
            + face_velocities[i, 2] * normals[i, 2]
        ) * areas[i]

    return flow_rates


# 1. 读取VTU文件
# 方法1：直接读取本地文件


Path_data_h5 = PARAM.Path_data_h5
Path_comsol = PARAM.Path_comsol
comsol_params_map = PARAM.comsol_params_map
Path_mix_raw = PARAM.Path_mix_raw
Path_binary_raw = PARAM.Path_binary_raw
raw_shape = PARAM.raw_shape
num_void = PARAM.num_void
Path_PNdata = PARAM.Path_PNdata
resolution = PARAM.resolution
labeled_image = np.fromfile(Path_mix_raw, dtype=np.int32).reshape(raw_shape)
binary_image = np.fromfile(Path_binary_raw, dtype=np.uint8).reshape(raw_shape)
# labeled_image = np.where(
#     (labeled_image < 1) | (labeled_image > num_void), -1, labeled_image
# )
num_pore = PARAM.num_pore
labeled_image = np.where((labeled_image < 1), 0, labeled_image)
with h5py.File(Path_data_h5, "r") as f:
    group_mesh = f["mesh"]
    group_voxel = f["voxel"]
    group_mesh_i = group_mesh["_u0.05617_hf10000"]
    voxel_cell = group_voxel["voxel.cell"][:]
    cell_voxel = group_mesh["cell.voxel"][:]
    points = group_mesh["points"][:]
    cells = group_mesh["cells"][:]

    # 创建 PyVista 网格
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: cells}, points)  # 假设是四面体

    # 绑定 cell_voxel 值到单元
    grid.cell_data["voxel_value"] = cell_voxel

    # 上色并绘图
    plotter = pv.Plotter()
    plotter.add_mesh(
        grid,
        scalars="voxel_value",
        cmap="viridis",
        show_edges=True,  # 显示单元边界
        opacity=0.8,
        scalar_bar_args={"title": "Cell Voxel Value"},
    )
    plotter.show()

    cell_center = group_mesh["cell.center"][:]

    point_ux = group_mesh_i["point.ht.dfluxx"][:] + np.nan_to_num(
        group_mesh_i["point.ht.cfluxx"][:], nan=0.0
    )
    point_uy = group_mesh_i["point.ht.dfluxy"][:] + np.nan_to_num(
        group_mesh_i["point.ht.cfluxy"][:], nan=0.0
    )
    point_uz = group_mesh_i["point.ht.dfluxz"][:] + np.nan_to_num(
        group_mesh_i["point.ht.cfluxz"][:], nan=0.0
    )

    ##### dualn #####
    dualn = net.vtk2network(PARAM.Path_net_dual)
    throat_conns = dualn["throat.conns"]
    pore_surface_all = dualn["pore.surface_all"]
    throat_void = dualn["throat.void"]

    throat_faces, throat_connections, throat_cell = get_cell_interface(
        cells, cell_voxel
    )
    throat_interface_bool = throat_connections[:, 0] != throat_connections[:, 1]

    throat_bool = (
        throat_interface_bool
        & np.all(throat_connections > 0, axis=1)
        # & np.all(throat_connections >= num_void + 1, axis=1)
    )  # & throat_void_bool
    throat_faces = throat_faces[throat_bool]
    throat_connections = throat_connections[throat_bool]
    throat_cell = throat_cell[throat_bool]

    normals, areas = compute_oriented_normals_nb(
        points,
        throat_faces,
        throat_cell,
        cell_center,
    )
    throat_faces_vectors = np.column_stack(
        (
            point_ux[throat_faces].mean(axis=1),
            point_uy[throat_faces].mean(axis=1),
            point_uz[throat_faces].mean(axis=1),
        )
    )

    fluxes = compute_flow_rates_nb(throat_faces_vectors, normals, areas)

    throat_connections, inverse_indices = unique_rows(
        throat_connections, return_inverse=True, keepdims=False
    )
    fluxes = np.bincount(inverse_indices, weights=fluxes)
    throat_connections -= 1
    throat_inner_bool = np.all(throat_connections >= 0, axis=1)

    throat_connections = throat_connections[throat_inner_bool]
    fluxes = fluxes[throat_inner_bool]
    fluxes = np.nan_to_num(fluxes, nan=0.0, posinf=0.0, neginf=0.0)
    throat_unsorted_bool = throat_connections[:, 0] > throat_connections[:, 1]
    throat_connections[throat_unsorted_bool] = throat_connections[:, [1, 0]][
        throat_unsorted_bool
    ]
    fluxes[throat_unsorted_bool] *= -1

    throat_connections, inverse_indices, counts = unique_rows(
        throat_connections, return_inverse=True, return_counts=True, keepdims=False
    )
    fluxes = np.bincount(inverse_indices, weights=fluxes) / counts
    throat_conns = throat_connections
    throat_fluid_flux = fluxes
    # throat_map = find_throat_conns_map(throat_conns, throat_connections)

    # throat_fluid_flux = np.zeros(len(throat_conns), dtype=np.float32)
    # throat_fluid_flux[throat_map[:, 0]] = fluxes[throat_map[:, 1]]
    # throat_fluid_flux[~throat_void] = 0

    cell_sample = np.sort(cells[20])
    face_0 = cell_sample[[0, 1, 2]]
    face_1 = cell_sample[[0, 1, 3]]
    face_2 = cell_sample[[0, 2, 3]]
    face_3 = cell_sample[[1, 2, 3]]
    normal_0 = np.cross(
        points[face_0[2]] - points[face_0[0]],
        points[face_0[1]] - points[face_0[0]],
    )
    normal_1 = np.cross(
        points[face_1[2]] - points[face_1[0]],
        points[face_1[1]] - points[face_1[0]],
    )
    normal_2 = np.cross(
        points[face_2[2]] - points[face_2[0]],
        points[face_2[1]] - points[face_2[0]],
    )
    normal_3 = np.cross(
        points[face_3[2]] - points[face_3[0]],
        points[face_3[1]] - points[face_3[0]],
    )

    area_0 = 0.5 * np.sqrt(np.sum(normal_0**2))
    area_1 = 0.5 * np.sqrt(np.sum(normal_1**2))
    area_2 = 0.5 * np.sqrt(np.sum(normal_2**2))
    area_3 = 0.5 * np.sqrt(np.sum(normal_3**2))
    normal_0 /= np.sqrt(np.sum(normal_0**2))
    normal_1 /= np.sqrt(np.sum(normal_1**2))
    normal_2 /= np.sqrt(np.sum(normal_2**2))
    normal_3 /= np.sqrt(np.sum(normal_3**2))
    flux_0 = (
        np.dot(
            np.array(
                [
                    point_ux[face_0].mean(),
                    point_uy[face_0].mean(),
                    point_uz[face_0].mean(),
                ]
            ),
            normal_0,
        )
        * area_0
    )
    flux_1 = (
        np.dot(
            np.array(
                [
                    point_ux[face_1].mean(),
                    point_uy[face_1].mean(),
                    point_uz[face_1].mean(),
                ]
            ),
            normal_1,
        )
        * area_1
    )
    flux_2 = (
        np.dot(
            np.array(
                [
                    point_ux[face_2].mean(),
                    point_uy[face_2].mean(),
                    point_uz[face_2].mean(),
                ]
            ),
            normal_2,
        )
        * area_2
    )
    flux_3 = (
        np.dot(
            np.array(
                [
                    point_ux[face_3].mean(),
                    point_uy[face_3].mean(),
                    point_uz[face_3].mean(),
                ]
            ),
            normal_3,
        )
        * area_3
    )
    flux = np.array((flux_0, flux_1, flux_2, flux_3))
    flux_total = np.sum(flux)
    print(
        f"flux_0 = {flux_0}, flux_1 = {flux_1}, flux_2 = {flux_2}, flux_3 = {flux_3}, flux_total = {flux_total},err_relative = {flux_total/np.sum(flux[flux>0])}"
    )

    for i in range(num_pore):
        if ~pore_surface_all[i]:  # and dualn["pore.void"][i]
            throat_bool_i = np.any(throat_conns == i, axis=1)  # & throat_void
            throat_conns_i = throat_conns[throat_bool_i]
            throat_fluid_flux_i = throat_fluid_flux[throat_bool_i].copy()
            throat_unsorted_bool_i = throat_conns_i[:, 0] != i
            throat_fluid_flux_i[throat_unsorted_bool_i] *= -1
            pore_flux = throat_fluid_flux_i.sum()
            pore_flux_pos = throat_fluid_flux_i[throat_fluid_flux_i > 0].sum()
            err_relative = pore_flux / pore_flux_pos
            err_relative = np.nan_to_num(err_relative, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"pore_{i} 相对误差 = {err_relative:.4f}")
