import sys

sys.path.append("../../")
import numpy as np
import numba as nb
import timeit
from mpnm_new.extraction._extraction_numba import nb_binary_dilation
from mpnm_new.util._utils_numba import nb_unique_uint
from mpnm_new.util import unique_rows, find_throat_conns_map
from mpnm_new import network as net
from skimage.measure import regionprops_table
from joblib import Parallel, delayed
from tqdm import tqdm
from Papers.P1.Code.COMSOL.comsol_params import extract_Re_hf
import h5py
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from Papers.P1.Code.COMSOL.comsol_params import (
    PARAMS_N2_545,
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


@nb.njit(fastmath=True, cache=True, nogil=True, parallel=True, error_model="numpy")
def nb_flux(labeled_region, velocity_region):
    """第二次遍历：填充预分配数组"""
    nz, ny, nx = labeled_region.shape
    labels = np.empty((nz, ny, nx, 6), dtype=labeled_region.dtype)
    labels_neighbors = np.empty((nz, ny, nx, 6), dtype=labeled_region.dtype)
    flux_out = np.zeros((nz, ny, nx, 6), dtype=np.float32)
    for z in nb.prange(nz):
        for y in nb.prange(ny):
            for x in nb.prange(nx):
                current = labeled_region[z, y, x]
                # if current>num_void+1:
                #     print(1)
                if current > 0:
                    for i, (dz, dy, dx) in enumerate(d3q6_directions):
                        z_, y_, x_ = z + dz, y + dy, x + dx

                        # 确定邻居标签
                        if not (0 <= z_ < nz and 0 <= y_ < ny and 0 <= x_ < nx):
                            neighbor = 0
                        else:
                            neighbor = labeled_region[z_, y_, x_]

                        # 记录通量
                        if neighbor != current and neighbor >= 0:
                            if neighbor != 0:  # 如果不是边界
                                uz, uy, ux = 0.5 * (
                                    velocity_region[z, y, x]
                                    + velocity_region[z_, y_, x_]
                                )

                            else:
                                uz, uy, ux = velocity_region[z, y, x]

                            flux = uz * dz + uy * dy + ux * dx
                            labels[z, y, x, i] = current
                            labels_neighbors[z, y, x, i] = neighbor
                            flux_out[z, y, x, i] = flux

    # 统计通量
    flux_out = flux_out.reshape(-1)
    flux_out_bool = flux_out != 0
    flux_out = flux_out[flux_out_bool]
    labels = labels.reshape(-1)
    labels = labels[flux_out_bool]
    labels_neighbors = labels_neighbors.reshape(-1)
    labels_neighbors = labels_neighbors[flux_out_bool]

    table = np.empty((flux_out.shape[0], 3), dtype=flux_out.dtype)
    table[:, 0] = labels
    table[:, 1] = labels_neighbors
    table[:, 2] = flux_out

    return table


def calculate_flux(labeled_image, labeled_image_Uzyx):
    """计算流量"""
    table_o = nb_flux(labeled_image, labeled_image_Uzyx)
    throat_conns = table_o[:, :2].astype(np.int32)
    throat_flux = table_o[:, 2]
    sorted_indices = np.lexsort((throat_conns[:, 0], throat_conns[:, 1]))
    throat_conns = throat_conns[sorted_indices]
    throat_flux = throat_flux[sorted_indices]

    throat_conns, reverse_indices = unique_rows(
        throat_conns, return_inverse=True, keepdims=False
    )
    throat_flux = np.bincount(reverse_indices, weights=throat_flux)

    return throat_conns, throat_flux


def draw_err_hist(err_relative, title=None):
    err_relative = np.asarray(err_relative)
    err_relative *= 100

    # 计算1%和99%分位数
    lower_bound = np.percentile(err_relative, 0.1)
    upper_bound = np.percentile(err_relative, 99.9)

    # 过滤掉1%的极端值
    err_relative = err_relative[
        (err_relative >= lower_bound) & (err_relative <= upper_bound)
    ]

    # 计算±5%和±10%的概率
    probability_2_5 = np.count_nonzero(np.abs(err_relative) <= 2.5) / len(err_relative)
    probability_5 = np.count_nonzero(np.abs(err_relative) <= 5) / len(err_relative)

    counts, bins = np.histogram(err_relative, bins=20, density=True)
    probabilities = counts / len(err_relative)
    # plt.bar(bins[:-1], probabilities, width=np.diff(bins), align="edge", alpha=0.7)
    plt.hist(err_relative, bins=20, density=True, alpha=0.5)
    plt.ylabel("Probability Density")

    # 添加 ±5% 红线
    plt.axvline(x=2.5, color="r", linestyle="--", linewidth=1.5, label="±2.5%")
    plt.axvline(x=-2.5, color="r", linestyle="--", linewidth=1.5)

    # 添加 ±10% 蓝线
    plt.axvline(x=5, color="b", linestyle="--", linewidth=1.5, label="±5%")
    plt.axvline(x=-5, color="b", linestyle="--", linewidth=1.5)

    # 添加标签和图例
    plt.xlabel("Relative Error (%)")
    # plt.title("Distribution of Relative Errors")
    plt.legend(loc="upper right")

    # 在左上方添加概率标注
    plt.text(
        0.05,
        0.95,
        f"Proportion of [-2.5%, 2.5%]: {probability_2_5:.0%}\n"
        f"Proportion of [-5%, 5%]: {probability_5:.0%}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )
    if title is not None:
        pass
        # plt.title(title)
    plt.show()
    # plt.show()


def get_flux(PARAM, draw=False):
    PARAM = PARAM()
    Path_data_h5 = PARAM.Path_data_h5
    Path_comsol = PARAM.Path_comsol
    comsol_params_map = PARAM.comsol_params_map
    Path_mix_raw = PARAM.Path_mix_raw
    raw_shape = PARAM.raw_shape
    num_void = PARAM.num_void
    Path_PNdata = PARAM.Path_PNdata
    resolution = PARAM.resolution
    labeled_image = np.fromfile(Path_mix_raw, dtype=np.int32).reshape(raw_shape)
    # labeled_image = np.where(
    #     (labeled_image < 1) | (labeled_image > num_void), -1, labeled_image
    # )
    num_pore = PARAM.num_pore
    labeled_image = np.where((labeled_image < 1), -1, labeled_image)
    binary_image = np.fromfile(PARAM.Path_binary_raw, dtype=np.uint8).reshape(raw_shape)
    void_bool = binary_image == 0
    """
    u
    v
    w
    ht.cfluxx
    ht.cfluxy
    ht.cfluxz
    ht.dfluxx
    ht.dfluxy
    ht.dfluxz
    """

    if not Path_PNdata.exists():
        PNdata = {}
    else:
        PNdata = pickle.load(open(Path_PNdata, "rb"))
    dualn = net.vtk2network(PARAM.Path_net_dual)
    throat_conns = dualn["throat.conns"]
    throat_void = dualn["throat.void"]
    throat_solid = dualn["throat.solid"]
    throat_connect = dualn["throat.connect"]
    pore_surface_all = dualn["pore.surface_all"]
    throat_conns = np.sort(throat_conns, axis=1)

    with h5py.File(Path_data_h5, "r", swmr=True) as f:
        err_relative_flow_list = []
        err_relative_heat_list = []
        for key in f.keys():
            print(key)
            _Re, _hf = extract_Re_hf(key, _Re_astype=np.float32, _hf_astype=np.int64)
            PNdata_name_i = f"_Re{_Re:.5f}_hf{_hf}"

            PNdata_data_i = PNdata.get(PNdata_name_i, {})
            # ### fluid flux ###
            data_ux = f[key][comsol_params_map["u"]]
            data_uy = f[key][comsol_params_map["v"]]
            data_uz = f[key][comsol_params_map["w"]]
            labeled_image_Uzyx = np.stack((data_uz, data_uy, data_ux), axis=3)
            connections, fluxes = calculate_flux(labeled_image, labeled_image_Uzyx)
            fluxes *= resolution**2
            connections -= 1
            throat_pores_bool = np.all(connections >= 0, axis=1)

            connections = connections[throat_pores_bool]
            fluxes = fluxes[throat_pores_bool]
            fluxes = np.nan_to_num(fluxes, nan=0.0, posinf=0.0, neginf=0.0)
            throat_unsorted_bool = connections[:, 0] > connections[:, 1]
            connections[throat_unsorted_bool] = connections[:, [1, 0]][
                throat_unsorted_bool
            ]
            fluxes[throat_unsorted_bool] *= -1

            connections, reverse_indices, counts = unique_rows(
                connections, return_inverse=True, return_counts=True, keepdims=False
            )
            fluxes = np.bincount(reverse_indices, weights=fluxes) / counts
            throat_map = find_throat_conns_map(throat_conns, connections)

            throat_fluid_flux = np.zeros(len(throat_conns), dtype=np.float32)
            throat_fluid_flux[throat_map[:, 0]] = fluxes[throat_map[:, 1]]
            throat_fluid_flux[~throat_void] = 0
            err_relative_list = []
            for i in range(num_pore):
                if ~pore_surface_all[i] and dualn["pore.void"][i]:
                    throat_bool_i = np.any(throat_conns == i, axis=1) & throat_void
                    throat_conns_i = throat_conns[throat_bool_i]
                    throat_fluid_flux_i = throat_fluid_flux[throat_bool_i].copy()
                    throat_unsorted_bool_i = throat_conns_i[:, 0] != i
                    throat_fluid_flux_i[throat_unsorted_bool_i] *= -1
                    pore_flux = throat_fluid_flux_i.sum()
                    pore_flux_pos = throat_fluid_flux_i[throat_fluid_flux_i > 0].sum()
                    err_relative = pore_flux / pore_flux_pos
                    err_relative = np.nan_to_num(
                        err_relative, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    err_relative_list.append(err_relative)
                    print(f"pore_{i} 相对误差 = {err_relative:.4f}")
            if draw:
                draw_err_hist(err_relative_list)
            err_relative_flow_list.extend(err_relative_list)
            PNdata_data_i["throat.fluid_flux_comsol"] = throat_fluid_flux

            ### conductive heat flux ###
            data_dfluxx = f[key][comsol_params_map["ht.dfluxx"]]
            data_dfluxy = f[key][comsol_params_map["ht.dfluxy"]]
            data_dfluxz = f[key][comsol_params_map["ht.dfluxz"]]

            labeled_image_dfluxzyx = np.stack(
                (data_dfluxz, data_dfluxy, data_dfluxx), axis=3
            )
            connections, fluxes = calculate_flux(labeled_image, labeled_image_dfluxzyx)
            fluxes *= resolution**2
            connections -= 1
            throat_pores_bool = np.all(connections >= 0, axis=1)

            connections = connections[throat_pores_bool]
            fluxes = fluxes[throat_pores_bool]
            fluxes = np.nan_to_num(fluxes, nan=0.0, posinf=0.0, neginf=0.0)
            throat_unsorted_bool = connections[:, 0] > connections[:, 1]
            connections[throat_unsorted_bool] = connections[:, [1, 0]][
                throat_unsorted_bool
            ]
            fluxes[throat_unsorted_bool] *= -1

            connections, reverse_indices, counts = unique_rows(
                connections, return_inverse=True, return_counts=True, keepdims=False
            )
            fluxes = np.bincount(reverse_indices, weights=fluxes) / counts
            throat_map = find_throat_conns_map(throat_conns, connections)

            throat_conductive_flux = np.zeros(len(throat_conns), dtype=np.float32)
            throat_conductive_flux[throat_map[:, 0]] = fluxes[throat_map[:, 1]]

            # for i in range(num_pore):
            #     if ~pore_surface_all[i]:  # and dualn["pore.void"][i]:
            #         throat_bool_i = np.any(throat_conns == i, axis=1)  # & throat_void
            #         throat_conns_i = throat_conns[throat_bool_i]
            #         throat_conductive_flux_i = throat_conductive_flux[throat_bool_i].copy()
            #         throat_unsorted_bool_i = throat_conns_i[:, 0] != i
            #         throat_conductive_flux_i[throat_unsorted_bool_i] *= -1
            #         pore_flux = throat_conductive_flux_i.sum()
            #         pore_flux_pos = throat_conductive_flux_i[
            #             throat_conductive_flux_i > 0
            #         ].sum()
            #         err_relative = pore_flux / pore_flux_pos
            #         err_relative = np.nan_to_num(
            #             err_relative, nan=0.0, posinf=0.0, neginf=0.0
            #         )
            #         print(f"pore_{i} 相对误差 = {err_relative:.4f}")

            PNdata_data_i["throat.heat_flux_conductive_comsol"] = throat_conductive_flux

            ### convective heat flux ###
            data_cfluxx = f[key][comsol_params_map["ht.cfluxx"]]
            data_cfluxy = f[key][comsol_params_map["ht.cfluxy"]]
            data_cfluxz = f[key][comsol_params_map["ht.cfluxz"]]

            # data_cfluxx = (
            #     f[key][comsol_params_map["u"]]
            #     * f[key][comsol_params_map["T"]]
            #     * PARAM.Cp_void
            #     * PARAM.rho_void
            # )
            # data_cfluxy = (
            #     f[key][comsol_params_map["v"]]
            #     * f[key][comsol_params_map["T"]]
            #     * PARAM.Cp_void
            #     * PARAM.rho_void
            # )
            # data_cfluxz = (
            #     f[key][comsol_params_map["w"]]
            #     * f[key][comsol_params_map["T"]]
            #     * PARAM.Cp_void
            #     * PARAM.rho_void
            # )

            labeled_image_cfluxzyx = np.stack(
                (data_cfluxz, data_cfluxy, data_cfluxx), axis=3
            )

            labeled_image_cfluxzyx[~void_bool, :] = np.nan
            connections, fluxes = calculate_flux(labeled_image, labeled_image_cfluxzyx)
            fluxes *= resolution**2
            connections -= 1
            throat_pores_bool = np.all(connections >= 0, axis=1)

            connections = connections[throat_pores_bool]
            fluxes = fluxes[throat_pores_bool]
            fluxes = np.nan_to_num(fluxes, nan=0.0, posinf=0.0, neginf=0.0)
            throat_unsorted_bool = connections[:, 0] > connections[:, 1]
            connections[throat_unsorted_bool] = connections[:, [1, 0]][
                throat_unsorted_bool
            ]
            fluxes[throat_unsorted_bool] *= -1

            connections, reverse_indices, counts = unique_rows(
                connections, return_inverse=True, return_counts=True, keepdims=False
            )
            fluxes = np.bincount(reverse_indices, weights=fluxes) / counts
            throat_map = find_throat_conns_map(throat_conns, connections)

            throat_convective_flux = np.zeros(len(throat_conns), dtype=np.float32)
            throat_convective_flux[throat_map[:, 0]] = fluxes[throat_map[:, 1]]
            throat_fluid_flux[~throat_void] = 0

            # for i in range(num_pore):
            #     if ~pore_surface_all[i]:  # and dualn["pore.void"][i]:
            #         throat_bool_i = np.any(throat_conns == i, axis=1)  # & throat_void
            #         throat_conns_i = throat_conns[throat_bool_i]
            #         throat_convective_flux_i = throat_convective_flux[throat_bool_i].copy()
            #         throat_unsorted_bool_i = throat_conns_i[:, 0] != i
            #         throat_convective_flux_i[throat_unsorted_bool_i] *= -1
            #         pore_flux = throat_convective_flux_i.sum()
            #         pore_flux_pos = throat_convective_flux_i[
            #             throat_convective_flux_i > 0
            #         ].sum()
            #         err_relative = pore_flux / pore_flux_pos
            #         err_relative = np.nan_to_num(
            #             err_relative, nan=0.0, posinf=0.0, neginf=0.0
            #         )
            #         print(f"pore_{i} 相对误差 = {err_relative:.4f}")

            PNdata_data_i["throat.heat_flux_convective_comsol"] = throat_convective_flux

            ### total heat flux ###
            labeled_image_cfluxzyx = labeled_image_dfluxzyx + np.nan_to_num(
                labeled_image_cfluxzyx, nan=0.0
            )
            connections, fluxes = calculate_flux(labeled_image, labeled_image_cfluxzyx)
            fluxes *= resolution**2
            connections -= 1
            throat_pores_bool = np.all(connections >= 0, axis=1)

            connections = connections[throat_pores_bool]
            fluxes = fluxes[throat_pores_bool]
            fluxes = np.nan_to_num(fluxes, nan=0.0, posinf=0.0, neginf=0.0)
            throat_unsorted_bool = connections[:, 0] > connections[:, 1]
            connections[throat_unsorted_bool] = connections[:, [1, 0]][
                throat_unsorted_bool
            ]
            fluxes[throat_unsorted_bool] *= -1

            connections, reverse_indices, counts = unique_rows(
                connections, return_inverse=True, return_counts=True, keepdims=False
            )
            fluxes = np.bincount(reverse_indices, weights=fluxes) / counts
            throat_map = find_throat_conns_map(throat_conns, connections)

            throat_flux_total = np.zeros(len(throat_conns), dtype=np.float32)
            throat_flux_total[throat_map[:, 0]] = fluxes[throat_map[:, 1]]
            err_relative_list = []
            for i in range(num_pore):
                if ~pore_surface_all[i]:  # and dualn["pore.void"][i]:
                    throat_bool_i = np.any(throat_conns == i, axis=1)  # & throat_void
                    throat_conns_i = throat_conns[throat_bool_i]
                    throat_flux_total_i = throat_flux_total[throat_bool_i].copy()
                    throat_unsorted_bool_i = throat_conns_i[:, 0] != i
                    throat_flux_total_i[throat_unsorted_bool_i] *= -1
                    pore_flux = throat_flux_total_i.sum()
                    pore_flux_pos = throat_flux_total_i[throat_flux_total_i > 0].sum()
                    err_relative = pore_flux / pore_flux_pos
                    err_relative = np.nan_to_num(
                        err_relative, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    err_relative_list.append(err_relative)
                    print(f"pore_{i} 相对误差 = {err_relative:.4f}")
            if draw:
                draw_err_hist(err_relative_list)
            err_relative_heat_list.extend(err_relative_list)
            PNdata_data_i["throat.heat_flux_total_comsol"] = throat_flux_total

            PNdata[PNdata_name_i] = PNdata_data_i

        pickle.dump(PNdata, open(Path_PNdata, "wb"))
        return err_relative_flow_list, err_relative_heat_list


PARAMS = [
    *PARAMS_N2_545,
    *PARAMS_N3_455,
    *PARAMS_N4_353,
    *PARAMS_N4_689,
    *PARAMS_N4_869,
    *PARAMS_N5_000,
    *PARAMS_N10_000,
]


for PARAM in PARAMS:
    err_relative_flow_list, err_relative_heat_list = get_flux(PARAM, draw=False)
    # draw_err_hist(err_relative_flow_list, title="Relative Error of Fluid Flux")
    # draw_err_hist(err_relative_heat_list, title="Relative Error of Heat Flux")
