#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:55:27 2022

@author: yojeep
"""

import copy
import scipy.spatial as spt
import pandas as pd
import os
import math
import inspect
import numpy as np
from collections import OrderedDict
from typing import Union, List, Tuple, Literal


from ._topotools_numba import (
    nb_find_neighbor_ball,
    nb_species_balance_conv,
    nb_energy_balance_conv,
    nb_mass_balance_conv,
    nb_cal_pore_veloc,
    nb_calculate_pore_flux,
)
from ..enum import Throat_Types
from ..util import (
    is_inplace,
    unique_uint,
    pd_col_loc,
    ravel,
    check_mpn,
    pd_itertuples,
    hash_array,
)

type_mpn = dict
type_boundary_conditions = pd.DataFrame
if "update_inner_info" not in os.environ.keys():
    os.environ["update_inner_info"] = "False"
type_union_list_tuple_ndarray = Union[list, tuple, np.ndarray]


def correct_error_pores(
    mpn: type_mpn,
    field_keys: Union[str, List[str]],
    ratio=0.05,
    inplace=True,
    **kwargs,
):
    mpn = is_inplace(mpn, inplace)
    if type(field_keys) is str:
        field_keys = [field_keys]
    from sklearn.neighbors import LocalOutlierFactor

    for field_key in field_keys:
        x, y, z = mpn["pore.coords"].T
        data = pd.DataFrame(
            {
                "x": x,
                "y": y,
                "z": z,
                "pore.void": mpn["pore.void"],
                f"{field_key}": mpn[field_key],
            }
        )
        model = LocalOutlierFactor(n_neighbors=30, contamination=ratio)
        res = model.fit_predict(data)
        mpn[f"pore.error_{field_key}"] = res < 0
    return mpn


def analyze_connectivity(mpn: type_mpn, min_len=0):
    import igraph as ig

    graph = ig.Graph(edges=mpn["throat.conns"])
    # degrees = graph.degree()
    components = graph.connected_components(mode="strong")
    components = np.asarray(components.membership).reshape(-1)
    components_unique, length = np.unique(
        components, return_counts=True
    )  # sorted = False
    # 给每个簇分配不同的颜色
    # colors = plt.get_cmap('tab10').colors
    # vertex_colors = [colors[i % len(colors)] for i in components_membership]

    # 画出图，并给不同簇分配颜色
    # layout = graph.layout("fr")  # 使用Fruchterman-Reingold布局
    # ig.plot(graph, layout=layout, vertex_color=vertex_colors, target="output_graph.png")
    components_dict = {}
    for i in components_unique:
        components_dict[i] = (components == i).nonzero()[0]
    # 打印簇的数量和大小
    print(f"Number of clusters: {len(components_unique)}")
    print(f"Sizes of clusters: {length}")

    if min_len > 0:
        isolated_pores = np.array([], dtype=int)
        for i in components_unique:
            if length[i] <= min_len:
                isolated_pores = np.concatenate((isolated_pores, components_dict[i]))
        return components_dict, isolated_pores
    else:
        return components_dict


def get_boundary_conditions_scope_bool(
    mpn: type_mpn,
    boundary_conditions: type_boundary_conditions,
    inplace=True,
    check_scope=True,
):
    """
    "Starting from the end, remove duplicate scope_bool == True within the same field."
    boundary_conditions should be a pandas dataframe with columns:  phase_key, field, io, boundary_type, scope_key, scope_bool, value
    """
    # 关键字列表
    columns = ["phase_key", "field_key", "axis", "io", "boundary_type", "scope_key"]

    # 找到不在 columns 中的关键字
    missing_columns = [
        column for column in columns if column not in boundary_conditions.columns
    ]

    # 如果有任何关键字不在 columns 中，抛出异常
    if missing_columns:
        raise KeyError(
            f"The following keys are not in columns: {', '.join(missing_columns)}, boundary_conditions should have columns: {', '.join(columns)}"
        )
    if inplace:
        pass
    else:
        boundary_conditions = copy.deepcopy(boundary_conditions)
    if "scope_bool" in boundary_conditions.columns:
        pass
    else:
        boundary_conditions["scope_bool"] = pd.DataFrame(
            {
                "scope_bool": [
                    np.array([True]),
                ]
                * len(boundary_conditions)
            }
        )
    boundary_conditions["scope_bool"] = boundary_conditions["scope_bool"].astype(
        "object"
    )
    for i in range(len(boundary_conditions)):
        boundary_conditions.iat[i, pd_col_loc(boundary_conditions, "scope_bool")] = mpn[
            boundary_conditions.iloc[i]["scope_key"]
        ]
    if "phase_bool" in boundary_conditions.columns:
        pass
    else:
        boundary_conditions["phase_bool"] = pd.DataFrame(
            {
                "phase_bool": [
                    np.array([True]),
                ]
                * len(boundary_conditions)
            }
        )
    boundary_conditions["phase_bool"] = boundary_conditions["phase_bool"].astype(
        "object"
    )
    for i in range(len(boundary_conditions)):
        boundary_conditions.iat[i, pd_col_loc(boundary_conditions, "phase_bool")] = mpn[
            boundary_conditions.iloc[i]["phase_key"]
        ]
    if check_scope:
        boundary_conditions = check_boundary_conditions(
            mpn, boundary_conditions, inplace=inplace
        )
    return boundary_conditions


def check_boundary_conditions(mpn, boundary_conditions, inplace=True):
    if inplace:
        pass
    else:
        boundary_conditions = copy.deepcopy(boundary_conditions)

    """
    "Starting from the end, remove duplicate scope_bool == True within the same field and phase."
    """
    columns = boundary_conditions.columns
    if "scope_bool" not in columns:
        raise Exception("scope_bool column not found in boundary_conditions")

    # 获取唯一的 field_key 和 phase_key 组合
    unique_combinations = (
        boundary_conditions[["field_key", "phase_key"]].drop_duplicates().values
    )

    for field_key, phase_key in unique_combinations:
        # 找到相同 field_key 和 phase_key 的索引
        indexes = (
            (boundary_conditions["field_key"] == field_key)
            & (boundary_conditions["phase_key"] == phase_key)
        ).nonzero()[0]

        if len(indexes) > 1:
            # 创建一个与 scope_bool 大小相同的空布尔数组
            empty_bool = np.zeros_like(
                mpn[boundary_conditions.iloc[indexes[0]]["scope_key"]], dtype=bool
            )
            # reversed_indexes = indexes[::-1]

            for i, index in enumerate(indexes):
                scope_bool_i = boundary_conditions.iloc[index]["scope_bool"]

                # 根据条件更新 scope_bool
                boundary_conditions.iat[
                    index, pd_col_loc(boundary_conditions, "scope_bool")
                ] = np.where((scope_bool_i) & (~empty_bool), True, False)

                # 更新 empty_bool，确保后续重复位置被置为 False
                empty_bool = empty_bool | scope_bool_i

    return boundary_conditions


def get_nb_params(func: callable, local_params: dict = None) -> dict:
    """
    优化后的参数提取函数
    """
    prefix = "mpn_"
    len_prefix = len(prefix)

    # 1. 预存 local_params['mpn'] 避免重复查询
    mpn_dict = local_params.get("mpn", {}) if local_params else {}

    # 2. 单次遍历参数，直接生成关键数据
    param_names = inspect.signature(func).parameters.keys()
    mpn_pairs = []  # 存储 (原参数名, 转换后的参数名)
    other_names = []

    for name in param_names:
        if name.startswith(prefix):
            suffix = name[len_prefix:]
            # 合并操作：直接生成最终需要的 mpn_name
            mpn_name = suffix.replace("_", ".", 1)
            mpn_pairs.append((name, mpn_name))
        else:
            other_names.append(name)

    # 3. 直接通过键检查构建 nb_params（避免索引循环）
    nb_params = {}
    for nb_mpn_name, mpn_name in mpn_pairs:
        if mpn_name not in mpn_dict:
            raise ValueError(f"Parameter {mpn_name} not found in mpn")
        nb_params[nb_mpn_name] = mpn_dict[mpn_name]

    # 4. 批量更新其他参数（字典推导式）
    nb_params.update(
        {name: local_params[name] for name in other_names if name in local_params}
    )

    return nb_params


def read_surface_area(
    mpn,
    path: str,
    index_col=None,
    resolution: float = 1.0,
    labels: Union[List, Tuple] = None,
    inplace: bool = True,
):
    """
    according to right hand axis
            Z
            |
            |
            |
            |
            |
            |
            |________________ Y
            /
            /
        /
        /
        /
        /
    X

    """
    mpn = is_inplace(mpn, inplace)
    if "pore.all_surface" not in mpn.keys():
        mpn["pore.all_surface"] = np.zeros_like(mpn["pore.all"], dtype=bool)
    else:
        pass
    if labels is None:
        labels = (
            "x-_surface",
            "x+_surface",
            "y-_surface",
            "y+_surface",
            "z-_surface",
            "z+_surface",
        )
    axis_tuple = ("x-", "x+", "y-", "y+", "z-", "z+")
    Boundaries_areas = pd.read_csv(path, index_col=index_col).to_numpy()
    Boundaries_areas[:, 0] = Boundaries_areas[:, 0] - 1

    max_label = len(mpn["pore.all"]) - 1

    Boundaries_areas = Boundaries_areas[Boundaries_areas[:, 0] <= max_label]
    Boundaries_labels = Boundaries_areas[:, 0]
    colm_bools_list = []  # each column > 0
    colm_labels_list = []  # each column labels
    colm_areas_list = []  # each column areas
    mpn_pore_which_surface_area_list = []  # each mpn[pore.which_surface_area]
    mpn_pore_which_surface_list = []  # each mpn[pore.which_surface]
    for i, ax in enumerate(axis_tuple):
        index = i + 1
        colm_areas_list.append(Boundaries_areas[:, index])
        colm_bools_list.append(colm_areas_list[i] > 0)
        colm_labels_list.append(Boundaries_labels[colm_bools_list[i]])

        mpn_pore_which_surface_area = np.zeros_like(mpn["pore.all"], dtype=np.float32)
        mpn_pore_which_surface_area[colm_labels_list[i]] = colm_areas_list[i][
            colm_bools_list[i]
        ]
        mpn_pore_which_surface_area_list.append(mpn_pore_which_surface_area)

        mpn_pore_which_surface_list.append(
            mpn_pore_which_surface_area_list[i].astype(bool, copy=False)
        )

    for i, label in enumerate(labels):
        mpn["pore." + labels[i]] = mpn_pore_which_surface_list[i]
        mpn["pore.all_surface"][colm_labels_list[i]] = True
        mpn["pore." + labels[i] + "_area"] = (
            mpn_pore_which_surface_area_list[i] * resolution**2
        )

    return mpn


def find_surface_KDTree(
    mpn: type_mpn,
    axis: Union[List[str], Tuple[str, ...], str] = "x",
    imsize: np.ndarray = None,
    resolution: float = 0,
    label_1: Union[List[str], Tuple[str, ...], str] = "x-_surface",
    label_2: Union[List[str], Tuple[str, ...], str] = "x+_surface",
    workers: int = math.ceil(os.cpu_count() / 4),
    percentile: float = 1.0,
    inplace: bool = False,
) -> type_mpn:
    """
    find the surface of the micropore network using KDTree
    Parameters:
    -----------
    mpn: dict
        the micropore network
    axis: str or list or tuple
        the axis to find the surface, can be 'x', 'y', 'z', or a list or tuple of them
    imsize: float
        the size of the image
    resolution: float
        the resolution of the image
    label_1: str or list or tuple
        the label of the surface, can be 'x-_surface', 'y-_surface',  'z-_surface', or a list or tuple of them
    label_2: str or list or tuple
        the label of the surface, can be 'x+_surface', 'y+_surface',  'z+_surface', or a list or tuple of them
    workers: int
        the number of workers to use
    percentile: float
        the percentile of the distance to find the surface
    inplace: bool
        if the mpn is inplace or not
    return:
    -------
    mpn: dict
        the multi-physics network
    """
    # t1 = time.time()
    mpn = is_inplace(mpn, inplace)
    if "pore.all_surface" not in mpn.keys():
        mpn["pore.all_surface"] = np.zeros_like(mpn["pore.all"], dtype=bool)
    else:
        pass
    if hasattr(axis, "__len__"):
        if len(axis) > 1:
            for i, ax in enumerate(axis):
                find_surface_KDTree(
                    mpn, ax, imsize, resolution, label_1[i], label_2[i], workers
                )
            return mpn
    k = 1  # 寻找的邻点数
    distance_factor = 1.2  # 深度
    distance_factor2 = 0  # 平面外移
    id_coord = np.column_stack(
        (mpn["pore._id"], mpn["pore.coords"], mpn["pore.radius"])
    )
    coords = id_coord[:, 1:4]
    ckt = spt.cKDTree(coords)
    min_distances, _ = ckt.query(coords, k=2, workers=workers)
    min_distances = min_distances[:, 1][min_distances[:, 1] > 0]
    length_temp = np.percentile(min_distances, percentile)
    length_min = length_temp

    axis_min = np.min(coords, axis=0)
    axis_max = np.max(coords, axis=0)

    axis_block_num = np.ceil((axis_max - axis_min) / length_min).astype(int, copy=False)

    axis_dict = {"x": 0, "y": 1, "z": 2}
    if axis == "x":
        sequence = (2, 0, 1)
    elif axis == "y":
        sequence = (0, 2, 1)
    elif axis == "z":
        sequence = (0, 1, 2)
    else:
        raise Exception("error status should be x,y,z")
    axis_use = axis_dict.pop(axis)
    others_list = [0, 1, 2]
    others_list.remove(axis_use)
    axis_length = axis_max[axis_use] - axis_min[axis_use]
    side_0 = np.linspace(
        axis_min[others_list[0]],
        axis_max[others_list[0]],
        axis_block_num[others_list[0]],
    )
    side_1 = np.linspace(
        axis_min[others_list[1]],
        axis_max[others_list[1]],
        axis_block_num[others_list[1]],
    )
    side_0, side_1 = np.meshgrid(side_0, side_1)
    side_0 = side_0.reshape(-1)
    side_1 = side_1.reshape(-1)
    side_2 = np.full((len(side_0)), axis_min[axis_use] - distance_factor2 * axis_length)
    side_3 = np.full((len(side_0)), axis_max[axis_use] + distance_factor2 * axis_length)
    mesh_1 = np.column_stack((side_0, side_1, side_2))[:, sequence]
    mesh_2 = np.column_stack((side_0, side_1, side_3))[:, sequence]
    distance1, index1 = ckt.query(
        mesh_1, workers=workers, k=k
    )  # 返回最近邻点的距离d和在数组中的顺序x
    index1, distance1 = index1.reshape(-1), distance1.reshape(-1)
    index1 = unique_uint(index1[distance1 < distance_factor * np.mean(distance1)])
    distance2, index2 = ckt.query(mesh_2, workers=workers, k=k)
    index2, distance2 = index2.reshape(-1), distance2.reshape(-1)
    index2 = unique_uint(index2[distance2 < distance_factor * np.mean(distance2)])
    pore_number1 = np.zeros((id_coord.shape[0]), dtype=bool)
    pore_number2 = np.zeros((id_coord.shape[0]), dtype=bool)
    name_label1 = "pore." + label_1
    name_label2 = "pore." + label_2
    pore_number1[index1] = True
    mpn[name_label1] = pore_number1
    pore_number2[index2] = True
    mpn[name_label2] = pore_number2
    mpn["pore.all_surface"][[*index1, *index2]] = True
    return mpn


def trim_surface(mpn: type_mpn, inplace: bool = True) -> type_mpn:
    """
    trim the surface of the multi-physics network
    Parameters:
    -----------
    mpn: dict
        the multi-physics network
    inplace: bool
        if the mpn is inplace or not
    return:
    -------
    mpn: dict
    """
    mpn = is_inplace(mpn, inplace)
    for i in ["left", "right"]:
        for j in ["back", "front"]:
            for k in ["bottom", "top"]:
                back = np.copy(
                    mpn["pore." + i + "_surface"] * mpn["pore." + k + "_surface"]
                )
                mpn["pore." + i + "_surface"][back] = False
                mpn["pore." + k + "_surface"][back] = False
                back = np.copy(
                    mpn["pore." + j + "_surface"] * mpn["pore." + k + "_surface"]
                )
                mpn["pore." + j + "_surface"][back] = False
                mpn["pore." + k + "_surface"][back] = False
            back = np.copy(
                mpn["pore." + i + "_surface"] * mpn["pore." + j + "_surface"]
            )
            mpn["pore." + i + "_surface"][back] = False
            mpn["pore." + j + "_surface"][back] = False

    return mpn


def divide_layer(mpn: type_mpn, n: Union[List[int], Tuple[int]]) -> dict:
    """
    divide the multi-physics network pores into layers of x,y,z axis
    Parameters:
    -----------
    mpn: dict
        the multi-physics network
    n: list or tuple
        the number of layers in each direction
    return:
    -------
    layer: dict
        the layers of the multi-physics network
    """
    layer = {}
    # step=size*resulation/n
    layer[0] = {}  # x
    layer[1] = {}  # y
    layer[2] = {}  # z
    for i in np.arange(3):
        index = np.min(mpn["pore.coords"][:, i])
        step = (
            np.max(mpn["pore.coords"][:, i]) - np.min(mpn["pore.coords"][:, i])
        ) / n[i]
        for j in np.arange(n[i]):
            layer[i][j] = np.copy(mpn["pore.all"])
            layer[i][j][(mpn["pore.coords"][:, i] - index) < j * step] = False
            layer[i][j][(mpn["pore.coords"][:, i] - index) > (j + 1) * step] = False
    return layer


def divide_layer_throat(
    mpn: type_mpn,
    n: Union[List[int], Tuple[int]],
    size: List[float],
    resolution: float,
) -> dict:
    """
    divide the multi-physics network throats into layers of x,y,z axis
    Parameters:
    -----------
    mpn: dict
        the multi-physics network
    n: list or tuple
        the number of layers in each direction
    size: list or tuple
        the size of the image
    resolution: float
        the resolution of the image
    return:
    -------
    layer: dict
        the layers of the multi-physics network
    """
    layer = {}
    # step=size*resulation/n
    layer[0] = {}  # left right
    layer[1] = {}  # back front
    layer[2] = {}  # bottom top
    for i in np.arange(3):
        index = min(mpn["throat.coords"][:, i])
        step = size[i] * resolution / n[i]
        for j in np.arange(n[i]):
            layer[i][j] = np.copy(mpn["throat.all"])
            layer[i][j][(mpn["throat.coords"][:, i] - index) < j * step] = False
            layer[i][j][(mpn["throat.coords"][:, i] - index) > (j + 1) * step] = False
    return layer


def pore_health(mpn: type_mpn, connections: int = 1) -> dict:
    """
    check the health of the multi-physics network
    Parameters:
    -----------
    mpn: dict
        the multi-physics network
    connections: int
        if the number of connections of the pore is less than connections,
        then the pore is considered as single pore
    return:
    -------
    health: dict
    """
    mpn_throat_conns = mpn["throat.conns"]
    pore_counts = np.bincount(
        mpn_throat_conns.reshape(-1), minlength=mpn["pore.all"].size
    )
    single_pores = (pore_counts < connections).nonzero()[0]

    health = {
        "single_pore": single_pores,
        "single_throat": np.isin(mpn_throat_conns, single_pores, kind="table")
        .any(axis=1)
        .nonzero()[0],
    }

    return health


def pore_health_s(mpn: type_mpn) -> dict:
    """
    check the health of the multi-physics network
    Parameters:
    ----------
    mpn: dict
    return:
    -------
    health: dict
    """
    number = len(mpn["pore.all"])
    conns = np.copy(mpn["throat.conns"])
    health = {}
    health["single_pore"] = []
    health["single_throat"] = []

    for i in np.arange(number):
        val0 = len(conns[:, 0][conns[:, 0] == i])
        val1 = len(conns[:, 1][conns[:, 1] == i])

        if val1 + val0 <= 1:
            health["single_pore"].append(i)
            ind0 = np.argwhere(conns[:, 0] == i)
            ind1 = np.argwhere(conns[:, 1] == i)
            if len(ind0) > 0 or len(ind1) > 0:
                health["single_throat"].append(np.concatenate((ind0, ind1))[0][0])

    return health


def trim_pore(
    mpn: type_mpn,
    pores: Union[list, tuple, np.ndarray] = None,
    throats: Union[list, tuple, np.ndarray] = None,
    remove_iso_pore: bool = True,
    keep_inlets_outlets: bool = False,
    inplace: bool = True,
) -> type_mpn:
    mpn = is_inplace(mpn, inplace)
    if pores is None:
        pores = np.array([], dtype=int)
    if throats is None:
        throats = np.array([], dtype=int)

    pores = ravel(pores)
    throats = ravel(throats)
    throats_isolated = (
        np.isin(mpn["throat.conns"], pores, kind="table").any(axis=1).nonzero()[0]
    )

    throats = np.append(throats, throats_isolated)
    for i in mpn:
        if "pore" in i and "throat" not in i:
            mpn[i] = np.delete(mpn[i], pores, axis=0)
        if "throat" in i:
            mpn[i] = np.delete(mpn[i], throats, axis=0)

    if remove_iso_pore:
        pores_isolated = (
            np.isin(mpn["pore._id"], mpn["throat.conns"], kind="table", invert=True)
        ).nonzero()[0]
        if len(pores_isolated) > 0:
            print("Find isolated %d pore" % len(pores_isolated))
            for i in mpn:
                if "pore" in i and "throat" not in i:
                    mpn[i] = np.delete(mpn[i], pores_isolated, axis=0)

    mpn["pore._id"] = np.arange(len(mpn["pore.all"]))
    mpn["throat._id"] = np.arange(len(mpn["throat.all"]))
    throat_internal_index = mpn["throat._id"][(mpn["throat.conns"][:, 0] >= 0)]
    mpn["pore.label"] = np.sort(mpn["pore.label"], kind="stable")
    """
    https://stackoverflow.com/questions/13572448/replace-values-of-a-numpy-index-array-with-values-of-a-list
    pore.label is assumed sorted
    """
    mpn["throat.conns"][:, 0][throat_internal_index] = (
        np.digitize(mpn["throat.conns"][:, 0][throat_internal_index], mpn["pore.label"])
        - 1
    )
    mpn["throat.conns"][:, 1] = (
        np.digitize(mpn["throat.conns"][:, 1], mpn["pore.label"]) - 1
    )
    # label_id_map = {i:j for i,j in zip(mpn['pore.label'], mpn['pore._id'])}
    # mpn['throat.conns'] = fr.remap(mpn['throat.conns'], label_id_map,preserve_missing_labels=True)

    if keep_inlets_outlets:
        pass
    else:
        mpn["throat.conns"] = mpn["throat.conns"][throat_internal_index]
        for i in mpn:
            if "throat" in i and "conns" not in i:
                mpn[i] = mpn[i][throat_internal_index]

    mpn["throat._id"] = np.arange(len(mpn["throat.all"]))
    mpn["pore.label"] = np.arange(len(mpn["pore.all"]))
    mpn["throat.label"] = np.arange(len(mpn["throat.all"]))
    return mpn


def trim_phase(
    mpn: type_mpn,
    pores: type_union_list_tuple_ndarray = None,
    throat: type_union_list_tuple_ndarray = None,
) -> dict:
    # count=len(mpn)
    """
    trim the phase of the multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    pores: list or tuple or np.ndarray
        the pores to be removed
    throat: list or tuple or np.ndarray
        the throats to be removed
    return:
    -------
    backup: dict
        the backup of the original multi-physics network
    """
    backup = {}
    if pores is None:
        pores = np.array([], dtype=int)

    if throat is None:
        throat = np.array([], dtype=int)
    for i in mpn:
        if "pore" in i and "throat" not in i:
            backup[i] = np.delete(
                mpn[i], pores, axis=0
            )  # if len(mpn[i].shape)>1 else np.delete(mpn[i],pores,axis=0)
        elif "throat" in i:
            backup[i] = np.delete(
                mpn[i], throat, axis=0
            )  # if len(mpn[i].shape)>1 else np.delete(mpn[i],throat,axis=0)
    backup["pore._id"] = np.arange(len(backup["pore.all"]))
    backup["throat._id"] = np.arange(len(backup["throat.all"]))
    return backup


def find_if_surface(mpn: type_mpn, ids: type_union_list_tuple_ndarray):
    """
    find the surface of the given pores
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    ids: list or tuple or np.ndarray
        the pores to be checked
    return:
    -------
    res: list
        the surface of the given pores
    """
    ids = ravel(ids)
    res = []
    for j in ids:
        b = 0
        for i in ["right", "left", "back", "front", "top", "bottom"]:
            if mpn["pore." + i + "_surface"][j]:
                b = "pore." + i + "_surface"
                res.append(b)
    return res


def find_whereis_pore(mpn: type_mpn, parameter, index):
    """
    find the index of the given parameter in the multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    parameter: list or tuple or np.ndarray
        the parameter to be checked
    index: int
        the index of the parameter to be found
    return:
    -------
    res: list
        the index of the given parameter in the multi-physics network
    """
    index1 = np.sort(parameter)[index]
    index2 = np.argwhere(parameter == index1)[0]
    index3 = find_if_surface(mpn, index2)
    return [index1, index2, index3]


def find_throat(
    mpn: type_mpn, ids: Union[type_union_list_tuple_ndarray, int]
) -> np.ndarray:
    """
    find the throats connected to the given pores
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    ids: list or tuple or np.ndarray
        the pores to be checked
    return:
    -------
    res: np.ndarray
        the throats connected to the given pores
    """
    ids = ravel(ids)
    ind1 = np.where(mpn["throat.conns"][:, 0] == ids)[0]
    ind2 = np.where(mpn["throat.conns"][:, 1] == ids)[0]
    res = np.append(ind1, ind2)
    return res


#
# def find_neighbor_nodes(mpn: type_mpn, ids: Union[type_union_list_tuple_ndarray, int]) -> np.ndarray:
#     """
#     find the neighbor nodes of the given pores
#     Parameters:
#     ----------
#     mpn: dict
#         the multi-physics network
#     ids: list or tuple or np.ndarray
#         the pores to be checked
#     return:
#     -------
#     throat_inf: np.ndarray
#         the throats connected to the given pores
#     node_inf: np.ndarray
#         the neighbor nodes of the given pores
#     """
#     if type(ids) is np.ndarray:
#         ids = ids.reshape(-1)[0]
#     else:
#         ids = np.array(ids).reshape(-1)[0]
#
#     num_pore = len(mpn['pore._id'])
#     A = coo_matrix((mpn['throat._id'], (mpn['throat.conns'][:, 1], mpn['throat.conns'][:, 0])),
#                    shape=(num_pore, num_pore), dtype=np.float64).tolil()
#     A = (A.T + A).tolil()
#     throat_inf = np.sort(A[ids].toarray()[0])[1:]
#     throat_inf = unique_uint(throat_inf)
#     node_inf = np.argwhere(A[ids].toarray()[0] != 0)
#     node_inf = np.sort(node_inf.reshape(len(node_inf)))
#
#     return throat_inf, node_inf


def find_neighbor_ball(
    mpn: type_mpn, ids: Union[type_union_list_tuple_ndarray, int]
) -> np.ndarray:
    """
    find the neighbor nodes of the given pores
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    ids: list or tuple or np.ndarray
        the pores to be checked
    return:
    -------
    throat_inf: np.ndarray
        the throats connected to the given pores
    node_inf: np.ndarray
        the neighbor nodes of the given pores
    """
    ids = ravel(ids)
    res = {}
    res["total"] = find_throat(mpn, ids)
    res["solid"] = []
    res["pore"] = []
    res["interface"] = []
    # num_pore=np.count_nonzero(mpn['pore.void']) if 'pore.void' in mpn else 0
    if res["total"].size > 0:
        for i in res["total"]:
            index = mpn["throat.conns"][i]
            tem = index if index[0] == ids else [ids, index[0]]
            if mpn["pore.void"][ids]:
                if mpn["pore.void"][tem[1]]:
                    res["pore"].append(np.append(tem, i))
                else:
                    res["interface"].append(np.append(tem, i))
            else:
                if mpn["pore.solid"][tem[1]]:
                    res["solid"].append(np.append(tem, i))
                else:
                    res["interface"].append(np.append(tem, i))

        res["pore"] = np.array(res["pore"], dtype=np.int64)
        res["solid"] = np.array(res["solid"], dtype=np.int64)
        res["interface"] = np.array(res["interface"], dtype=np.int64)

        return res
    else:
        return 0


def find_neighbor_ball_non_info(
    mpn: type_mpn, ids: Union[type_union_list_tuple_ndarray, int]
) -> np.ndarray:
    """
    find the neighbor nodes of the given pores
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    ids: list or tuple or np.ndarray
        the pores to be checked
    return:
    -------
    throat_inf: np.ndarray
    """
    ids = ravel(ids)
    check_mpn(mpn, keys_in_mpn=("pore.void", "pore.solid", "throat.conns"))
    mpn_pore_void = mpn["pore.void"]
    mpn_pore_solid = mpn["pore.solid"]
    mpn_throat_conns = mpn["throat.conns"]

    res = nb_find_neighbor_ball(mpn_pore_void, mpn_pore_solid, mpn_throat_conns, ids)
    return res


#####   coefficent calculation   #####


def H_P_fun(radius, lengh, viscosity):
    """
    calculate the Hydraulic permeability of the given pore
    Parameters:
    ----------
    radius: float
        the radius of the pore
    lengh: float
        the length of the pore
    viscosity: float
        the viscosity of the fluid
    return:
    -------
    g: float
        the hydraulic permeability of the given pore
    """
    g = np.pi / 8 * radius**4 / viscosity / lengh
    return g


def Mass_conductivity(mpn):
    """
    calculate the mass conductivity of the given multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    return:
    -------
    g_ij: np.ndarray
        the mass conductivity of the given multi-physics network
    """
    check_mpn(
        mpn,
        keys_in_mpn=(
            "pore.radius",
            "pore.real_shape_factor",
            "pore.real_k",
            "pore.viscosity",
            "throat.radius",
            "throat.real_shape_factor",
            "throat.real_k",
            "throat.viscosity",
        ),
    )
    mpn_pore_radius_0 = mpn["pore.radius"][mpn["throat.conns"][:, 0]]
    mpn_pore_radius_1 = mpn["pore.radius"][mpn["throat.conns"][:, 1]]
    mpn_pore_real_shape_factor_0 = mpn["pore.real_shape_factor"][
        mpn["throat.conns"][:, 0]
    ]
    mpn_pore_real_shape_factor_1 = mpn["pore.real_shape_factor"][
        mpn["throat.conns"][:, 1]
    ]
    mpn_pore_real_k_0 = mpn["pore.real_k"][mpn["throat.conns"][:, 0]]
    mpn_pore_real_k_1 = mpn["pore.real_k"][mpn["throat.conns"][:, 1]]
    mpn_pore_viscosity_0 = mpn["pore.viscosity"][mpn["throat.conns"][:, 0]]
    mpn_pore_viscosity_1 = mpn["pore.viscosity"][mpn["throat.conns"][:, 1]]
    mpn_throat_radius = mpn["throat.radius"]
    mpn_throat_real_shape_factor = mpn["throat.real_shape_factor"]
    mpn_throat_real_k = mpn["throat.real_k"]
    mpn_throat_viscosity = mpn["throat.viscosity"]
    if "throat.conduit_lengths_pore1" in mpn.keys():
        li = mpn["throat.conduit_lengths_pore1"]
        lj = mpn["throat.conduit_lengths_pore2"]
        lt = mpn["throat.conduit_lengths_throat"]

    elif "throat.conduit_lengths.pore1" in mpn.keys():
        li = mpn["throat.conduit_lengths.pore1"]
        lj = mpn["throat.conduit_lengths.pore2"]
        lt = mpn["throat.conduit_lengths.throat"]
    else:
        raise Exception("No throat conduit lengths found in the network")

    def cond(r, G, k, v):
        return (r**4) / 16 / G * k / v

    g_i = cond(
        mpn_pore_radius_0,
        mpn_pore_real_shape_factor_0,
        mpn_pore_real_k_0,
        mpn_pore_viscosity_0,
    )
    g_j = cond(
        mpn_pore_radius_1,
        mpn_pore_real_shape_factor_1,
        mpn_pore_real_k_1,
        mpn_pore_viscosity_1,
    )
    g_t = cond(
        mpn_throat_radius,
        mpn_throat_real_shape_factor,
        mpn_throat_real_k,
        mpn_throat_viscosity,
    )

    g_ij = (li + lj + lt) / (li / g_i + lj / g_j + lt / g_t)
    return g_ij


#
# def func_g(mpn, flux_Throat_profile, RE_th, C, E, m, n):
#     index = flux_Throat_profile > 0
#     flux_Throat_profile_abs = np.fabs(flux_Throat_profile)
#     mpn_throat_radius = mpn["throat.radius"]
#     mpn_throat_total_length = mpn["throat.total_length"]
#     mpn_pore_radius = mpn["pore.radius"]
#     mpn_pore_radius_pore0 = mpn_pore_radius[mpn["throat.conns"][:, 0]]
#     mpn_pore_radius_pore1 = mpn_pore_radius[mpn["throat.conns"][:, 1]]
#     mpn_throat_density = mpn["throat.density"]
#     i_pore_e = np.abs(1 - (mpn_throat_radius / mpn_pore_radius_pore0) ** 2) ** (
#         2 * m
#     )
#     i_pore_c = np.abs(
#         (1 - (mpn_throat_radius / mpn_pore_radius_pore0) ** 2) / 2
#     ) ** (n)
#     j_pore_e = np.abs(1 - (mpn_throat_radius / mpn_pore_radius_pore1) ** 2) ** (
#         2 * m
#     )
#     j_pore_c = np.abs(
#         (1 - (mpn_throat_radius / mpn_pore_radius_pore1) ** 2) / 2
#     ) ** (n)

#     tem0 = mpn_throat_total_length / Mass_conductivity(mpn)
#     tem1 = (
#         mpn_throat_density
#         / 2
#         / (np.pi**2)
#         * flux_Throat_profile_abs
#         / mpn_throat_radius**4
#         * ((E / RE_th) ** m + np.where(index, j_pore_e, i_pore_e))
#     )
#     tem2 = (
#         mpn_throat_density
#         / 2
#         / (np.pi**2)
#         * flux_Throat_profile_abs
#         / mpn_throat_radius**4
#         * ((C / RE_th) ** n + np.where(index, i_pore_c, j_pore_c))
#     )

#     tem3 = (
#         mpn_throat_density
#         * flux_Throat_profile_abs
#         / 2
#         / np.pi**2
#         * (1 / mpn_pore_radius_pore1**4 - 1 / mpn_pore_radius_pore0**4)
#         * (2 * index - 1)
#     )
#     result = tem1 + tem2 - tem3
#     result[flux_Throat_profile_abs == 0] = 0
#     result = 1 / (tem0 + result)
#     result = np.nan_to_num(result)
#     return result


def func_g(mpn, throat_flux, Re_throat, C, E, n, m):
    """
    https://www.sciencedirect.com/science/article/pii/S0017931024014571?via%3Dihub#b11
    https://doi.org/10.1016/j.ijheatmasstransfer.2024.126630
    https://doi.org/10.1016/j.cej.2013.11.077
    """
    gamma = 1
    throat_flux_bool = throat_flux > 0
    throat_flux_abs = np.fabs(throat_flux)
    mpn_pore_radius = mpn["pore.radius"]
    mpn_pore_radius_pore0 = mpn_pore_radius[mpn["throat.conns"][:, 0]]
    mpn_pore_radius_pore1 = mpn_pore_radius[mpn["throat.conns"][:, 1]]
    mpn_throat_radius = mpn["throat.radius"]
    mpn_throat_total_length = mpn["throat.total_length"]
    mpn_throat_viscosity = mpn["throat.viscosity"]
    mpn_throat_density = mpn["throat.density"]
    mpn_throat_radius_square = mpn_throat_radius**2
    mpn_throat_radius_quartic = mpn_throat_radius**4
    pore_radius_from = np.where(
        throat_flux_bool, mpn_pore_radius_pore1, mpn_pore_radius_pore0
    )
    pore_radius_to = np.where(
        throat_flux_bool, mpn_pore_radius_pore0, mpn_pore_radius_pore1
    )
    hydraulic_conductance = (
        8
        / np.pi
        * mpn_throat_viscosity
        * mpn_throat_total_length
        / mpn_throat_radius_quartic
    )
    pore_conductance_common = (
        mpn_throat_density
        * throat_flux_abs
        / (2 * np.pi**2 * mpn_throat_radius_quartic)
    )
    pore_conductance_C = (C / Re_throat) ** n + (
        0.5 - 0.5 * mpn_throat_radius_square / pore_radius_from**2
    ) ** n
    pore_conductance_E = (E / Re_throat) ** m + (
        1 - mpn_throat_radius_square / pore_radius_to**2
    ) ** (2 * m)

    pore_conductance = pore_conductance_common * (
        pore_conductance_C + pore_conductance_E
    )
    flow_pattern = (
        gamma
        / (2 * np.pi**2)
        * mpn_throat_density
        * throat_flux_abs
        * (1 / pore_radius_from**4 - 1 / pore_radius_to**4)
    )
    conductance_flow = pore_conductance - flow_pattern
    conductance_flow = np.nan_to_num(
        conductance_flow, copy=False, nan=0, posinf=0, neginf=0
    )
    conductance = 1 / (hydraulic_conductance + conductance_flow)
    return conductance


def Boundary_cond_cal(throat_inlet1, throat_inlet2, fluid, newPore, Pores):
    throat_inlet_cond = []
    BndG1 = np.sqrt(3) / 36 + 0.00001
    BndG2 = 0.07
    for i in np.arange(len(throat_inlet1)):
        indP2 = newPore[throat_inlet1[i, 2].astype(int) - 1].astype(int)
        GT = (
            1
            / fluid["viscosity"]
            * (throat_inlet1[i, 3] ** 2 / 4 / (throat_inlet1[i, 4])) ** 2
            * throat_inlet1[i, 4]
            * (throat_inlet1[i, 4] < BndG1)
            * 0.6
        )
        GT += (
            1
            / fluid["viscosity"]
            * throat_inlet1[i, 3] ** 2
            * throat_inlet1[i, 3] ** 2
            / 4
            / 8
            / (1 / 4 / np.pi)
            * (throat_inlet1[i, 4] > BndG2)
        )
        GT += (
            1
            / fluid["viscosity"]
            * (throat_inlet1[i, 3] ** 2 / 4 / (1 / 16)) ** 2
            * (1 / 16)
            * ((throat_inlet1[i, 4] >= BndG1) * (throat_inlet1[i, 4] <= BndG2))
            * 0.5623
        )
        GP2 = (
            1
            / fluid["viscosity"]
            * (Pores[indP2, 2] ** 2 / 4 / (Pores[indP2, 3])) ** 2
            * Pores[indP2, 3]
            * (Pores[indP2, 3] < BndG1)
            * 0.6
        )
        GP2 += (
            1
            / fluid["viscosity"]
            * Pores[indP2, 2] ** 2
            * Pores[indP2, 2] ** 2
            / 4
            / 8
            / (1 / 4 / np.pi)
            * (Pores[indP2, 3] > BndG2)
        )
        GP2 += (
            1
            / fluid["viscosity"]
            * (Pores[indP2, 2] ** 2 / 4 / (1 / 16)) ** 2
            * (1 / 16)
            * ((Pores[indP2, 3] >= BndG1) * (Pores[indP2, 3] <= BndG2))
            * 0.5623
        )
        # LP1 = throat_inlet2[i,3]
        LP2 = throat_inlet2[i, 4]
        LT = throat_inlet2[i, 5]
        throat_inlet_cond.append([indP2, 1 / (LT / GT + LP2 / GP2)])

    return np.array(throat_inlet_cond)


def species_balance_conv(
    mpn: type_mpn,
    g_ij: np.ndarray,
    Tem: np.ndarray,
    thermal_con_dual: np.ndarray,
    P_profile: np.ndarray,
    ids: Union[type_union_list_tuple_ndarray, int],
):
    """
    calculate the species balance of the given multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    g_ij: np.ndarray
        the mass conductivity of the given multi-physics network
    Tem: np.ndarray
        the temperature of the given multi-physics network
    thermal_con_dual: np.ndarray
        the thermal conductivity of the given multi-physics network
    P_profile: np.ndarray
        the pressure of the given multi-physics network
    ids: list or tuple or np.ndarray
        the pores to be checked
    return:
    -------
    res: np.ndarray
        the species balance of the given multi-physics network
    """
    check_mpn(mpn=mpn, keys_in_mpn=("throat.radius", "throat.total_length"))
    ids = ravel(ids)

    mpn_throat_radius = mpn["throat.radius"]
    mpn_throat_length = mpn["throat.total_length"]

    inner_info, inner_start2end = update_inner_info(mpn)
    result = nb_species_balance_conv(
        inner_info,
        inner_start2end,
        mpn_throat_radius,
        mpn_throat_length,
        g_ij,
        Tem,
        thermal_con_dual,
        P_profile,
        ids,
    )
    return result


def calculate_species_flow(
    mpn: type_mpn,
    boundary_conditions: type_boundary_conditions,
    g_ij: np.ndarray,
    Tem_c: np.ndarray,
    thermal_con_dual: np.ndarray,
    P_profile: np.ndarray,
):
    """
    calculate the species flow of the given multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network

    boundary_conditions: list
        the boundary condition of the given multi-physics network
    g_ij: np.ndarray
        the mass conductivity of the given multi-physics network
    Tem_c: np.ndarray
        the temperature of the given multi-physics network
    thermal_con_dual: np.ndarray
        the thermal conductivity of the given multi-physics network
    P_profile: np.ndarray
        the pressure of the given multi-physics network
    return:
    -------
    output: dict
        the species flow of the given multi-physics network
    """
    check_mpn(mpn=mpn)
    output = {}
    total = 0
    for boundary_condition in pd_itertuples(boundary_conditions):
        result = species_balance_conv(
            mpn,
            g_ij,
            Tem_c,
            thermal_con_dual,
            P_profile,
            mpn["pore._id"][mpn[boundary_condition.scope_key]],
        )[:, 0]
        output.update({boundary_condition.scope_key: np.sum(result)})
    for i in output:
        total += output[i]
    output.update({"total": np.sum(total)})
    return output


def energy_balance_conv(
    mpn: type_mpn,
    g_ij: np.ndarray,
    Tem: np.ndarray,
    coe_B: np.ndarray,
    P_profile: np.ndarray,
    ids: Union[type_union_list_tuple_ndarray, int],
) -> np.ndarray:
    """
    calculate the energy balance of the given multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    g_ij: np.ndarray
        the mass conductivity of the given multi-physics network
    Tem: np.ndarray
        the temperature of the given multi-physics network
    thermal_con_dual: np.ndarray
        the thermal conductivity of the given multi-physics network
    P_profile: np.ndarray
        the pressure of the given multi-physics network
    ids: list or tuple or np.ndarray
        the pores to be checked
    return:
    -------
    res: np.ndarray
        the energy balance of the given multi-physics network
    """
    check_mpn(
        mpn,
        keys_in_mpn=(
            "throat.radius",
            "throat.length",
            "throat.Cp",
            "throat.density",
        ),
    )
    ids = ravel(ids)
    inner_info, inner_start2end = update_inner_info(mpn)
    mpn_throat_Cp = mpn["throat.Cp"]
    mpn_throat_density = mpn["throat.density"]
    result = nb_energy_balance_conv(
        inner_info=inner_info,
        inner_start2end=inner_start2end,
        mpn_throat_Cp=mpn_throat_Cp,
        mpn_throat_density=mpn_throat_density,
        g_ij=g_ij,
        Tem=Tem,
        coe_B=coe_B,
        P_profile=P_profile,
        ids=ids,
    )
    return result


def calculate_heat_flow2(
    mpn: type_mpn,
    boundary_conditions: type_boundary_conditions,
    g_ij: np.ndarray,
    Tem_c: np.ndarray,
    coe_B: np.ndarray,
    P_profile: np.ndarray,
) -> dict:
    """
    calculate the heat flow of the given multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    boundary_conditions: list
        the boundary condition of the given multi-physics network
    g_ij: np.ndarray
        the mass conductivity of the given multi-physics network
    Tem_c: np.ndarray
        the temperature of the given multi-physics network
    coe_B: np.ndarray
        the thermal conductivity of the given multi-physics network
    P_profile: np.ndarray
        the pressure of the given multi-physics network
    return:
    -------
    output: dict
        the heat flow of the given multi-physics network
    """
    output = {}
    grouped = boundary_conditions.groupby("names", as_index=False, sort=False)
    for name, group in grouped:
        ids = np.asarray(group["ids"])
        result = energy_balance_conv(
            mpn=mpn,
            g_ij=g_ij,
            Tem=Tem_c,
            coe_B=coe_B,
            P_profile=P_profile,
            ids=ids,
        )[:, 0]
        output[name] = np.sum(result)
    output["total"] = sum(output.values())
    return output


def calculate_heat_flow(
    mpn: type_mpn,
    boundary_conditions: type_boundary_conditions,
    g_ij: np.ndarray,
    Tem_c: np.ndarray,
    thermal_con_dual: np.ndarray,
    P_profile: np.ndarray,
) -> dict:
    """
    calculate the heat flow of the given multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    boundary_conditions: list
        the boundary condition of the given multi-physics network
    g_ij: np.ndarray
        the mass conductivity of the given multi-physics network
    Tem_c: np.ndarray
        the temperature of the given multi-physics network
    thermal_con_dual: np.ndarray
        the thermal conductivity of the given multi-physics network
    P_profile: np.ndarray
        the pressure of the given multi-physics network
    return:
    -------
    output: dict
        the heat flow of the given multi-physics network
    """
    boundary_conditions = get_boundary_conditions_scope_bool(mpn, boundary_conditions)
    output = {}
    total = 0
    for boundary_condition in pd_itertuples(boundary_conditions):
        result = energy_balance_conv(
            mpn,
            g_ij,
            Tem_c,
            thermal_con_dual,
            P_profile,
            mpn["pore._id"][
                boundary_condition.phase_bool & boundary_condition.scope_bool
            ],
        )[:, 0]
        output.update(
            {
                f"{boundary_condition.phase_key}_{boundary_condition.scope_key}": np.sum(
                    result
                )
            }
        )
    for i in output:
        total += output[i]
    output.update({"total": np.sum(total)})
    return output


def mass_balance_conv(
    mpn: type_mpn,
    g_ij: np.ndarray,
    P_profile: np.ndarray,
    ids: Union[type_union_list_tuple_ndarray, int],
) -> np.ndarray:
    """
    calculate the mass balance of the given multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    g_ij: np.ndarray
        the mass conductivity of the given multi-physics network
    P_profile: np.ndarray
        the pressure of the given multi-physics network
    ids: list or tuple or np.ndarray
        the pores to be checked
    return:
    -------
    res: np.ndarray
        the mass balance of the given multi-physics network
    """
    check_mpn(mpn)
    ids = ravel(ids)
    inner_info, inner_start2end = update_inner_info(mpn)
    result = nb_mass_balance_conv(inner_info, inner_start2end, g_ij, P_profile, ids)
    return result


def mass_balance_conv_o(
    mpn: type_mpn,
    g_ij: np.ndarray,
    P_profile: np.ndarray,
    ids: Union[type_union_list_tuple_ndarray, int],
) -> np.ndarray:
    """
    calculate the mass balance of the given multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    g_ij: np.ndarray
        the mass conductivity of the given multi-physics network
    P_profile: np.ndarray
        the pressure of the given multi-physics network
    ids: list or tuple or np.ndarray
        the pores to be checked
    return:
    -------
    res: np.ndarray
        the mass balance of the given multi-physics network
    """
    check_mpn(mpn)
    ids = ravel(ids)
    # res=find_neighbor_ball(mpn,[ids])
    res = find_neighbor_ball(mpn, ids)
    if res == 0:
        result = 0
    elif len(res["pore"]) >= 1:
        delta_p = P_profile[res["pore"][:, 0]] - P_profile[res["pore"][:, 1]]

        cond_f = g_ij[[res["pore"][:, 2]]]
        if len(delta_p) >= 1:
            flux = delta_p * cond_f
        else:
            flux = 0
        result = np.sum(flux)

    else:
        result = 0
    # print('h_conv_f=%f,h_cond_f=%f, h_cond_sf=%f,h_cond_s=%f'%(h_conv_f,h_cond_f, h_cond_sf,h_cond_s))
    return result


def calculate_mass_flow2(
    mpn: type_mpn,
    boundary_conditions: type_boundary_conditions,
    g_ij: np.ndarray,
    P_profile: np.ndarray,
) -> dict:
    """
    calculate the mass flow of the given multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    boundary_conditions: Boundary_Condition
        the boundary condition of the given multi-physics network
    g_ij: np.ndarray
        the mass conductivity of the given multi-physics network
    P_profile: np.ndarray
        the pressure of the given multi-physics network
    return:
    -------
    output: dict
        the mass flow of the given multi-physics network
    """
    output = {}
    grouped = boundary_conditions.groupby("names", as_index=False, sort=False)
    for name, group in grouped:
        ids = np.asarray(group["ids"])
        result = mass_balance_conv(mpn, g_ij, P_profile, ids)
        output[name] = np.sum(result)
    output["total"] = sum(output.values())
    return output


def calculate_mass_flow(
    mpn: type_mpn,
    boundary_conditions: type_boundary_conditions,
    g_ij: np.ndarray,
    P_profile: np.ndarray,
) -> dict:
    """
    calculate the mass flow of the given multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    boundary_conditions: Boundary_Condition
        the boundary condition of the given multi-physics network
    g_ij: np.ndarray
        the mass conductivity of the given multi-physics network
    P_profile: np.ndarray
        the pressure of the given multi-physics network
    return:
    -------
    output: dict
        the mass flow of the given multi-physics network
    """
    boundary_conditions = get_boundary_conditions_scope_bool(mpn, boundary_conditions)
    output = {}
    total = 0
    for boundary_condition in pd_itertuples(boundary_conditions):
        output.update(
            {
                boundary_condition.scope_key: np.sum(
                    mass_balance_conv(
                        mpn,
                        g_ij,
                        P_profile,
                        mpn["pore._id"][boundary_condition.scope_bool],
                    )
                )
            }
        )
    for i in output:
        total += output[i]
    output.update({"total": np.sum(total)})
    return output


def cal_pore_veloc(
    mpn: type_mpn,
    g_ij: np.ndarray,
    P_profile: np.ndarray,
    ids: np.ndarray,
):
    """
    calculate the pore velocity of the given multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    g_ij: np.ndarray
        the mass conductivity of the given multi-physics network
    P_profile: np.ndarray
        the pressure of the given multi-physics network
    ids: list or tuple or np.ndarray
        the pores to be checked
    return:
    -------
    res: np.ndarray
        the pore velocity of the given multi-physics network
    """

    check_mpn(mpn=mpn)
    inner_info, inner_start2end = update_inner_info(mpn)
    mpn_pore_radius = mpn["pore.radius"]
    result = nb_cal_pore_veloc(
        inner_info, inner_start2end, mpn_pore_radius, g_ij, P_profile, ids
    )
    return result


def calculate_pore_flux(
    mpn: type_mpn,
    g_ij: np.ndarray,
    P_profile: np.ndarray,
    ids: np.ndarray,
):
    """
    calculate the pore flux of the given multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    g_ij: np.ndarray
        the mass conductivity of the given multi-physics network
    P_profile: np.ndarray
        the pressure of the given multi-physics network
    ids: list or tuple or np.ndarray
        the pores to be checked
    return:
    -------
    res: np.ndarray
        the pore flux of the given multi-physics network
    """
    mpn = check_mpn(mpn)
    ids = ravel(ids)
    inner_info, inner_start2end = update_inner_info(mpn)
    result = nb_calculate_pore_flux(inner_info, inner_start2end, g_ij, P_profile, ids)
    return result


def cal_pore_flux(mpn, g_ij, P_profile, ids):
    """
    calculate the pore flux of the given multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    g_ij: np.ndarray
        the mass conductivity of the given multi-physics network
    P_profile: np.ndarray
        the pressure of the given multi-physics network
    return:
    -------
    res: np.ndarray
        the pore flux of the given multi-physics network
    """
    result = mass_balance_conv(mpn, g_ij, P_profile, ids)
    return result


"""
def add_pores(self,mpn1,mpn2,trail=True):#mpn=mpn1,mpn2
    num_p=len(mpn1['pore.all'])
    num_t=len(mpn1['throat.all'])
    mpn2['throat._id']+=num_t
    mpn2['pore.label']+=num_p

    mpn={}
    if trail:
        mpn2['throat.conns']+=num_p
    for i in mpn2:
        if i not in mpn1:
            mpn[i]=np.zeros(num_p).astype(bool)
            mpn[i]=np.concatenate((mpn[i],mpn2[i])) 

        else:
            mpn[i]=np.concatenate((mpn1[i],mpn2[i])) 
    return mpn
"""


def add_pores(mpn1, mpn2, trail=True, inplace=True):  # mpn=mpn1,mpn2
    """
    add two multi-physics networks together
    add 2 to 1
    Parameters:
    ----------
    mpn1: dict
        the first multi-physics network
    mpn2: dict
        the second multi-physics network
    trail: bool
        whether to add the throat connections of the second network to the first network
    return:
    -------
    mpn: dict
        the merged multi-physics network
    """
    num_p = len(mpn1["pore.all"])
    num_t = len(mpn1["throat.all"])
    mpn2["throat._id"] += num_t
    mpn2["pore.label"] += num_p
    mpn = is_inplace(mpn1, inplace)

    if trail:
        mpn2["throat.conns"] += num_p
    for i in mpn2:
        if i not in mpn1:
            mpn[i] = (
                np.zeros(num_p, dtype=bool)
                if "pore" in i
                else np.zeros(num_t, dtype=bool)
            )
            mpn[i] = np.concatenate((mpn[i], mpn2[i]))
        else:
            mpn[i] = np.concatenate((mpn1[i], mpn2[i]))
    mpn["throat._id"] = np.sort(mpn["throat._id"], kind="stable")
    mpn["pore._id"] = np.sort(mpn["pore.label"], kind="stable")
    mpn["throat.label"] = np.sort(mpn["throat._id"], kind="stable")
    mpn["pore.label"] = np.sort(mpn["pore.label"], kind="stable")
    return mpn


def clone_pores(mpn, pores, label="clone_p"):
    """
    clone pores which has the same properties with the original pores
    bool pore properties are False
    pore._id is range(0,num_pore)
    other dtype pore properties are the same as the original pores
    other throat properties are the average of the original throat properties
    """
    clone = {}
    num = len(mpn["pore.all"][pores])
    for i in mpn:
        if "pore" in i and "throat" not in i:
            if "_id" in i:
                clone[i] = np.arange(num)

            elif mpn[i].dtype == bool:
                clone[i] = np.zeros(num, dtype=bool)
            else:
                clone[i] = mpn[i][pores]

        elif "throat" in i:
            clone[i] = np.zeros_like(mpn[i])
    clone["pore.all"] = np.ones(num, dtype=bool)
    clone[label] = np.ones(num, dtype=bool)
    return clone


def merge_clone_boundary_pore(
    mpn,
    pores,
    radius,
    offset,
    side: Literal["x-", "x+", "y-", "y+", "z-", "z+"],
    label="clone_p",
    area_mode: Literal["radius", "surface_area"] = "radius",
):
    """
    modify clone pores throat properties, which is connected to the original pores by a throat
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    pores: list or tuple or np.ndarray
        the pores to be connected
    radius: float
        the radius of the throat
    offset: distance between the original pores and the clone pores
    side: which side of the original pores to connect the clone pores, 'x-', 'x+', 'y-', 'y+', 'z-', 'z+'
    """
    mpn_copy = mpn

    side_to_axis = {
        "x-": (0, -1),
        "x+": (0, 1),
        "y-": (1, -1),
        "y+": (1, 1),
        "z-": (2, -1),
        "z+": (2, 1),
    }

    if side in side_to_axis:
        # mpn_copy['pore.coords'] += offset * side_to_axis[side][1]
        clone = clone_pores(mpn_copy, pores, label=label)
        org_coords = np.copy(clone["pore.coords"])
        axis, direction = side_to_axis[side]
        index = (
            np.min(clone["pore.coords"][:, axis])
            if direction == -1
            else np.max(clone["pore.coords"][:, axis])
        )
        clone["pore.coords"][:, axis] = index + direction * offset
    else:
        raise ValueError(
            f"Invalid side value: {side}. Expected one of 'x-', 'x+', 'y-', 'y+', 'z-', 'z+'."
        )

    num = len(mpn_copy["pore.all"])
    clone["throat.conns"] = np.column_stack(
        (clone["pore.label"], clone["pore._id"] + num)
    )
    clone["pore.label"] = clone["pore._id"]
    num_T = len(clone["throat.conns"])
    clone["throat.solid"] = np.zeros(num_T, dtype=bool)
    # clone['throat.solid'][(clone['throat.conns'][:,0]>=num_pore)&(clone['throat.conns'][:,1]>=num_pore)]=True
    clone["throat.solid"][mpn_copy["pore.solid"][clone["throat.conns"][:, 0]]] = True
    clone["throat.connect"] = np.zeros(num_T, dtype=bool)
    clone["throat.connect"][mpn_copy["pore.void"][clone["throat.conns"][:, 0]]] = True
    clone["throat.void"] = np.zeros(num_T, dtype=bool)
    clone["throat.void"] = ~(clone["throat.solid"] | clone["throat.connect"])
    clone["throat.label"] = np.arange(num_T)
    clone["throat._id"] = np.arange(num_T)
    clone["throat.all"] = np.ones(num_T, dtype=bool)
    clone["throat.length"] = np.linalg.norm(org_coords - clone["pore.coords"], axis=1)
    clone["throat.radius"] = radius  # clone['throat.all']*radius
    clone["throat.volume"] = np.pi * radius**2 * clone["throat.length"]
    if area_mode == "surface_area":
        clone[f"{label}_area"] = mpn[f"pore.{side}_surface_area"][
            clone["throat.conns"][:, 0]
        ]
        clone["throat.radius"] = np.sqrt(clone[f"{label}_area"] / np.pi)
        clone["throat.volume"] = (
            np.pi * clone["throat.radius"] ** 2 * clone["throat.length"]
        )

    return clone


def connect_repu_network(mpn, side) -> dict:
    """
    connect the repu network to the given multi-physics network
    Parameters:
    ----------
    mpn: dict
        the multi-physics network
    side: str
        the side of the repu network to be connected, 'x-', 'x+', 'y-', 'y+', 'z-', 'z+'
    return:
    -------
    mpn: dict
        the multi-physics network with the repu network connected
    """
    if side in ["right", "left"]:
        way = 0
        side1 = np.array(["right", "left"])[~(np.array(["right", "left"]) == side)][0]
    elif side in ["back", "front"]:
        way = 1
        side1 = np.array(["back", "front"])[~(np.array(["back", "front"]) == side)][0]
    elif side in ["bottom", "top"]:
        way = 2
        side1 = np.array(["bottom", "top"])[~(np.array(["bottom", "top"]) == side)][0]
    else:
        raise Exception(
            "The side is not correct, should be right,left,back,front,top,bottom"
        )

    mpn2 = copy.deepcopy(mpn)
    copy_surf = np.max(mpn["pore.coords"][mpn["pore." + side + "_surface"]][:, way])
    num_node = len(mpn["pore.all"])
    mpn2["pore.coords"][:, way] = 2 * copy_surf - mpn2["pore.coords"][:, way]
    # conn_t,conn_n=topotools().find_neighbor_nodes(mpn, mpn['pore._id'][mpn['pore.'+side+'_surface']])

    for i in mpn["pore._id"][mpn["pore." + side + "_surface"]]:
        mpn2["throat.conns"][:, 0][mpn2["throat.conns"][:, 0] == i] = i - num_node
        mpn2["throat.conns"][:, 1][mpn2["throat.conns"][:, 1] == i] = i - num_node

    mpn2 = trim_phase(
        mpn2, mpn["pore._id"][mpn["pore." + side + "_surface"]].astype(int), []
    )
    mpn = add_pores(mpn, mpn2, trail=True)
    li, indices, counts = np.unique(
        mpn["throat.conns"], axis=0, return_counts=True, return_index=True
    )
    duplicates = indices[np.where(counts > 1)]
    mpn = trim_pore(mpn, [], duplicates.astype(int))

    mpn["pore._id"] = np.arange(len(mpn["pore._id"]))
    index = np.digitize(mpn["throat.conns"][:, 0], mpn["pore.label"]) - 1
    mpn["throat.conns"][:, 0] = mpn["pore._id"][index]
    index = np.digitize(mpn["throat.conns"][:, 1], mpn["pore.label"]) - 1
    mpn["throat.conns"][:, 1] = mpn["pore._id"][index]

    mpn["pore.label"] = np.arange(len(mpn["pore._id"]))
    del mpn["pore." + side + "_surface"], mpn["pore." + side1 + "_surface"]
    """
    if side1 in ['left','back','bottom']:

        topotools().find_surface(mpn,np.array(['x','y','z'])[way],
                                imsize[way]*2,resolution,label_1=side1+'_surface',label_2=side+'_surface')
    else:
        topotools().find_surface(mpn,np.array(['x','y','z'])[way],
                                imsize[way]*2,resolution,label_1=side+'_surface',label_2=side1+'_surface') 
    """
    return mpn


def update_inner_info(mpn, check_keys=None, enable_cache=True, lru_size=50):
    # t0 = time.time()
    mpn = check_mpn(mpn)
    if enable_cache:
        if not hasattr(update_inner_info, "cache"):
            update_inner_info.cache = OrderedDict()

        cache = update_inner_info.cache
        check_keys = check_keys or ("throat.conns", "pore.void", "pore.solid")

        # Create cache key
        cache_key = tuple(
            (
                (k, hash_array(mpn[k]), mpn[k].shape)
                if isinstance(mpn[k], np.ndarray)
                else (k, mpn[k])
            )
            for k in check_keys
        )
        cache_key = frozenset(cache_key)

        if cache_key in cache:
            return cache[cache_key]
    # print('Updating pore conns info')
    """
    0=Pore, 1=Solid, 2=Interface
    """
    mpn_throat_conns = mpn["throat.conns"]
    mpn_pore_void = mpn["pore.void"]
    mpn_pore_solid = mpn["pore.solid"]

    num_pore = mpn_pore_void.size
    num_throat = len(mpn_throat_conns)
    mpn_throat_id = np.arange(num_throat)
    total_information = np.empty((2 * num_throat, 4), dtype="int64", order="F")
    total_information[:num_throat, 0:2] = mpn_throat_conns
    total_information[num_throat:, 0:2] = mpn_throat_conns[:, ::-1]
    total_information[:num_throat, 2] = mpn_throat_id
    total_information[num_throat:, 2] = mpn_throat_id
    # total_information = total_information[
    #     np.argsort(total_information[:, 0], kind="stable")
    # ]

    total_information_throat_type = total_information[:, 3]
    total_information_throat_type[:] = Throat_Types.interface
    pore_0 = total_information[:, 0]
    pore_1 = total_information[:, 1]
    void_throat = mpn_pore_void[pore_0] & mpn_pore_void[pore_1]
    solid_throat = mpn_pore_solid[pore_0] & mpn_pore_solid[pore_1]
    total_information_throat_type[void_throat] = Throat_Types.void
    total_information_throat_type[solid_throat] = Throat_Types.solid

    # equals to np.lexsort(total_information[:, 3],total_information[:,0]), but faster
    indices_throat = np.argsort(total_information[:, 3], kind="stable")
    indices_pore = np.argsort(total_information[indices_throat][:, 0], kind="stable")
    index_sorted = indices_throat[indices_pore]
    total_information = total_information[index_sorted]

    elements, counts = unique_uint(total_information[:, 0], return_counts=True)
    intervals = np.cumsum(counts)
    start = np.empty(elements.size, dtype=np.int64)
    start[0] = 0
    start[1:] = intervals[:-1]
    end = intervals
    start2end = np.full(shape=(num_pore, 2), fill_value=-1, dtype=np.int64)
    start2end[elements, 0] = start
    start2end[elements, 1] = end

    # Update cache if enabled
    if enable_cache:
        if len(cache) >= lru_size:
            cache.popitem(last=False)
        cache[cache_key] = (total_information, start2end)

    return total_information, start2end
    # print('Finished updating pore info')
    # inner = {'inner_info': total_information, 'inner_start2end': start2end}
    # mpn.update(inner)
