#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:55:27 2022

@author: yojeep
"""

import copy
import scipy.spatial as spt
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra
import pandas as pd
import os
import math
import inspect
import numpy as np
from collections import OrderedDict
from typing import Union, List, Tuple, Literal, Optional, Dict, Sequence
from numpy.typing import NDArray
from collections import namedtuple
import warnings
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
    to_list,
    remap,
)

type_mpn = dict
type_boundary_conditions = pd.DataFrame
if "update_inner_info" not in os.environ.keys():
    os.environ["update_inner_info"] = "False"
type_union_list_tuple_ndarray = Union[list, tuple, np.ndarray]


########################
def correct_error_pores(
    mpn: type_mpn,
    field_keys: Union[Sequence[str], str],
    ratio: float = 0.05,
    inplace: bool = True,
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


def analyze_connectivity(mpn: type_mpn, min_len: int = 0):
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


def check_connectivity(
    mpn: type_mpn,
    node_inlet: NDArray[np.integer],
    node_outlet: NDArray[np.integer],
    throat_resistance: NDArray[np.floating],
):
    num_node = mpn["pore.all"].size
    sink_node = num_node
    throat_conns = mpn["throat.conns"]
    throat_conns = np.concat(
        (
            throat_conns,
            np.column_stack(
                (
                    node_outlet,
                    np.broadcast_to(sink_node, node_outlet.size),
                )
            ),
        ),
        axis=0,
    )
    throat_resistance = np.concat(
        (throat_resistance, np.broadcast_to(0.0, node_outlet.size)),
    )
    rows = np.concat((throat_conns[:, 0], throat_conns[:, 1]))
    cols = np.concat((throat_conns[:, 1], throat_conns[:, 0]))
    datas = np.tile(throat_resistance, 2)
    R = csr_array(
        (datas, (rows, cols)), shape=(num_node + 1, num_node + 1), dtype=datas.dtype
    )
    dist = dijkstra(R, directed=False, indices=sink_node)
    path_resistance = dist[node_inlet]
    return path_resistance


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
    axis: Union[Sequence[str], str] = "x",
    label_0: Union[Sequence[str], str] = "x-_surface",
    label_1: Union[Sequence[str], str] = "x+_surface",
    inplace: bool = False,
    workers: Optional[int] = None,
) -> type_mpn:
    """
    find the surface of the micropore network using KDTree
    Parameters:
    -----------
    mpn: dict
        the micropore network
    axis: str or list or tuple
        the axis to find the surface, can be 'x', 'y', 'z', or a list or tuple of them
    label_0: str or list or tuple
        the label of the surface, can be 'x-_surface', 'y-_surface',  'z-_surface', or a list or tuple of them
    label_1: str or list or tuple
        the label of the surface, can be 'x+_surface', 'y+_surface',  'z+_surface', or a list or tuple of them
    inplace: bool
        if the mpn is inplace or not
    workers: int
        the number of workers to use
    return:
    -------
    mpn: dict
        the multi-physics network
    """
    axis = to_list(axis)
    label_0 = to_list(label_0)
    label_1 = to_list(label_1)

    if len(axis) > 1:
        assert len(axis) == len(label_0) == len(label_1), (
            "the length of axis, label_0, and label_1 should be the same"
        )

    mpn = is_inplace(mpn, inplace)
    if "pore.all_surface" not in mpn.keys():
        mpn["pore.all_surface"] = np.zeros_like(mpn["pore.all"], dtype=bool)
    else:
        pass

    if workers is None:
        workers = max(1, math.ceil(os.cpu_count() / 4))
    _dirs = ("x", "y", "z")
    distance_factor = 1.2  # 深度
    distance_factor2 = 0  # 平面外移
    coords = mpn["pore.coords"]
    ckt = spt.cKDTree(coords)
    min_distances, _ = ckt.query(coords, k=2, workers=workers)
    min_distances = min_distances[:, 1][min_distances[:, 1] > 0]
    length_min = np.mean(min_distances)
    axis_min = np.min(coords, axis=0)
    axis_max = np.max(coords, axis=0)
    axis_block_num = np.ceil((axis_max - axis_min) / length_min).astype(int, copy=False)

    axis_min = {_d: _min for _d, _min in zip(_dirs, axis_min)}
    axis_max = {_d: _max for _d, _max in zip(_dirs, axis_max)}
    axis_block_num = {_d: _block_num for _d, _block_num in zip(_dirs, axis_block_num)}

    for ax_i, label_0_i, label_1_i in zip(axis, label_0, label_1):
        mesh = {}
        _dir_others = [_ for _ in _dirs if _ != ax_i]
        num_side_mesh_points = np.prod([axis_block_num[_d] for _d in _dir_others])
        for _d in _dir_others:
            mesh[_d] = np.linspace(
                axis_min[_d],
                axis_max[_d],
                axis_block_num[_d],
            )
        grids = np.meshgrid(*(mesh[_d] for _d in _dir_others), indexing="ij")
        for i, _d in enumerate(_dir_others):
            mesh[_d] = np.tile(grids[i].reshape(-1), 2)
        axis_length = axis_max[ax_i] - axis_min[ax_i]
        mesh[ax_i] = np.concatenate(
            (
                np.full(num_side_mesh_points, axis_min[ax_i])
                - distance_factor2 * axis_length,
                np.full(num_side_mesh_points, axis_max[ax_i])
                + distance_factor2 * axis_length,
            )
        )
        mesh_points = np.column_stack([mesh[_d] for _d in _dirs])
        distances, indices = ckt.query(mesh_points, workers=workers, k=1)
        indices, distances = indices.reshape(-1), distances.reshape(-1)
        for ind, dist, label in zip(
            np.split(indices, 2), np.split(distances, 2), [label_0_i, label_1_i]
        ):
            ind = ind[dist < distance_factor * np.mean(dist)]
            mpn[f"pore.{label}"] = np.zeros_like(mpn["pore.all"], dtype=bool)
            mpn[f"pore.{label}"][ind] = True
            mpn["pore.all_surface"][ind] = True

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
    surface_labels = [
        f"pore.{i}_surface" for i in ("left", "right", "back", "front", "bottom", "top")
    ]
    mask = np.zeros_like(mpn["pore.all"], dtype=bool)
    for surface_label in surface_labels:
        mask |= mpn[surface_label]
    for surface_label in surface_labels:
        if surface_label in mpn:
            mpn[surface_label][mask] = False
    if "pore.all_surface" in mpn:
        mpn["pore.all_surface"][mask] = False
    return mpn


def divide_layer_pore(
    mpn: type_mpn,
    x: Optional[int] = None,
    y: Optional[int] = None,
    z: Optional[int] = None,
) -> dict:
    """
    divide the multi-physics network pores into layers of x,y,z axis
    Parameters:
    -----------
    mpn: dict
        the multi-physics network
    x,y,z: int
        the number of layers in each direction
    return:
    -------
    layer: dict
        the layers of the multi-physics network
    """
    ax_to_idx = {"x": 0, "y": 1, "z": 2}
    axis_dict = {"x": x, "y": y, "z": z}

    coords = mpn["pore.coords"]  # shape: (num_pores, 3)
    pore_all = mpn["pore.all"]  # shape: (num_pores,)
    num_pore = pore_all.size

    # Map axis name to coordinate index
    ax_to_idx = {"x": 0, "y": 1, "z": 2}

    num_pore = mpn["pore.all"].size
    layer = {}
    for ax, n_layers in axis_dict.items():
        if n_layers is None or n_layers <= 0:
            layer[ax] = np.array([], dtype=bool)
            continue
        mask = np.zeros((n_layers, num_pore), dtype=bool)
        coord = coords[:, ax_to_idx[ax]]
        min_val = coord.min()
        max_val = coord.max()

        # Avoid division by zero if all coords are the same
        if min_val == max_val:
            warnings.warning(
                f"length of {ax} coordinate is zero, all pores in all layers",
                category=UserWarning,
                stacklevel=1,
            )
            # All pores in all layers
            layer[ax] = ~mask
            continue

        # Create bin edges: n_layers layers => n_layers+1 edges
        bins = np.linspace(min_val, max_val, n_layers + 1)
        indices = np.digitize(coord, bins, right=False) - 1
        # Assign max-valued pores to the last layer
        indices[indices > (n_layers - 1)] = n_layers - 1

        # Create one-hot boolean mask: shape (n_layers, num_pore)
        mask[indices, np.arange(num_pore)] = True

        layer[ax] = mask

    return layer


def divide_layer_throat(
    mpn: Dict[str, NDArray],
    x: Optional[int] = None,
    y: Optional[int] = None,
    z: Optional[int] = None,
) -> Dict[str, NDArray]:
    """
    Divide throats into layers along x, y, z axes using binning.

    Parameters:
    -----------
    mpn : dict
        Multi-physics network with "throat.coords" and "throat.all".
    x, y, z : int or None
        Number of layers in each direction. If None or <= 0, returns empty mask.

    Returns:
    --------
    layer : dict
        layer['x'], layer['y'], layer['z']: each is a boolean array of shape (n_i, num_throats)
    """
    axis_dict = {"x": x, "y": y, "z": z}
    ax_to_idx = {"x": 0, "y": 1, "z": 2}

    coords = mpn["throat.coords"]  # shape: (num_throats, 3)
    throat_all = mpn["throat.all"]  # shape: (num_throats,)
    num_throat = throat_all.size

    layer = {}
    for ax, n_layers in axis_dict.items():
        if n_layers is None or n_layers <= 0:
            layer[ax] = np.zeros((0, num_throat), dtype=bool)
            continue

        coord = coords[:, ax_to_idx[ax]]
        min_val = coord.min()
        max_val = coord.max()

        # Initialize mask
        mask = np.zeros((n_layers, num_throat), dtype=bool)

        # Degenerate case: all throats at same coordinate
        if min_val == max_val:
            warnings.warning(
                f"length of {ax} coordinate is zero, all throats in all layers",
                category=UserWarning,
                stacklevel=1,
            )
            # All throats in all layers
            layer[ax] = ~mask
            continue

        # Create bins: n_layers → n_layers + 1 edges
        bins = np.linspace(min_val, max_val, n_layers + 1)

        # Digitize to get layer indices
        indices = np.digitize(coord, bins, right=False) - 1
        # Assign max-valued throats to the last layer
        indices[indices > (n_layers - 1)] = n_layers - 1

        # Assign throats to layers
        mask[indices, np.arange(num_throat)] = True

        layer[ax] = mask

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


def trim_pore(
    mpn: type_mpn,
    pores: Union[list, tuple, np.ndarray] = None,
    throats: Union[list, tuple, np.ndarray] = None,
    remove_iso_pores: bool = False,
    remove_neg_throats: bool = False,
    inplace: bool = False,
) -> type_mpn:
    mpn = is_inplace(mpn, inplace)
    if pores is None:
        pores = np.array([], dtype=np.int32)
    if throats is None:
        throats = np.array([], dtype=np.int32)

    pores = np.reshape(pores, -1)
    throats = np.reshape(throats, -1)
    isolated_throats = (
        np.isin(mpn["throat.conns"], pores, kind="table").any(axis=1).nonzero()[0]
    )
    if remove_neg_throats:
        neg_throats = (mpn["throat.conns"] < 0).any(axis=1).nonzero()[0]
        throats = np.concatenate(
            (
                throats,
                isolated_throats,
                neg_throats,
            )
        )
    else:
        throats = np.concatenate((throats, isolated_throats))
    remain_pores_bool = np.isin(mpn["pore._id"], pores, kind="table", invert=True)
    remain_throats_bool = np.isin(mpn["throat._id"], throats, kind="table", invert=True)
    for key in mpn:
        if key.startswith("pore."):
            mpn[key] = mpn[key][remain_pores_bool]
        elif key.startswith("throat."):
            mpn[key] = mpn[key][remain_throats_bool]
        else:
            pass

    if remove_iso_pores:
        remain_pores_bool = np.isin(mpn["pore._id"], mpn["throat.conns"], kind="table")
        num_iso_pores = mpn["pore._id"].size - remain_pores_bool.sum()
        if num_iso_pores > 0:
            print(f"Find {num_iso_pores} isolated pores")
            for key in mpn:
                if key.startswith("pore."):
                    mpn[key] = mpn[key][remain_pores_bool]
                else:
                    pass

    pore__id_continuous = np.arange(mpn["pore._id"].size, dtype=mpn["pore._id"].dtype)
    mpn["throat.conns"] = remap(
        arr=mpn["throat.conns"],
        keys=mpn["pore._id"],
        values=pore__id_continuous,
        inplace=True,
    )

    mpn["pore._id"] = pore__id_continuous
    mpn["pore.label"] = np.arange(mpn["pore.label"].size, dtype=mpn["pore.label"].dtype)

    mpn["throat._id"] = np.arange(mpn["throat._id"].size, dtype=mpn["throat._id"].dtype)
    mpn["throat.label"] = np.arange(
        mpn["throat.label"].size, dtype=mpn["throat.label"].dtype
    )
    # """
    # We can also use np.digitize do the thing.
    # https://stackoverflow.com/questions/13572448/replace-values-of-a-numpy-index-array-with-values-of-a-list
    # """

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


def func_g(
    mpn: type_mpn,
    throat_flux: NDArray,
    Re_throat: NDArray,
    n: float = 0.296,
    m: float = 1.0,
    C: float = 26.0,
    E: float = 27.0,
    gamma: float = 1.0,
):
    """
    Compute throat hydraulic conductance with inlet/outlet inertial corrections.

    References:
        -https://www.sciencedirect.com/science/article/pii/S0017931024014571?via%3Dihub#b11
        -https://doi.org/10.1016/j.ijheatmasstransfer.2024.126630
        -https://doi.org/10.1016/j.cej.2013.11.077

    """
    # === 1. Extract basic geometric and physical properties ===
    pore_radius = mpn["pore.radius"]
    throat_radius = mpn["throat.radius"]
    throat_length = mpn["throat.total_length"]
    viscosity = mpn["throat.viscosity"]
    density = mpn["throat.density"]

    conn_pore0 = mpn["throat.conns"][:, 0]
    conn_pore1 = mpn["throat.conns"][:, 1]

    r_throat_sq = throat_radius**2
    r_throat_4 = throat_radius**4

    # === 2. Determine flow direction ===
    is_positive_flux = throat_flux > 0
    abs_flux = np.fabs(throat_flux)

    # Identify upstream (inlet) and downstream (outlet) pore radii based on flow direction
    r_inlet = np.where(
        is_positive_flux, pore_radius[conn_pore1], pore_radius[conn_pore0]
    )
    r_outlet = np.where(
        is_positive_flux, pore_radius[conn_pore0], pore_radius[conn_pore1]
    )

    # === 3. Pure viscous resistance (Hagen-Poiseuille) ===
    viscous_resistance = 8.0 / np.pi * viscosity * throat_length / r_throat_4

    # === 4. Prefactor for inertial / kinetic energy correction ===
    kinetic_prefactor = density * abs_flux / (2.0 * np.pi**2 * r_throat_4)

    has_flow = Re_throat > 0

    # === 5. Inlet contraction loss coefficient (C-term) ===
    # NOTE: K_contraction_Re and K_contraction_geom are raised to power n independently, then summed
    beta_inlet_sq = r_throat_sq / r_inlet**2
    K_contraction_geom_raw = 0.5 * (1.0 - beta_inlet_sq)

    mask_C = has_flow & (K_contraction_geom_raw > 0)

    # Reynolds-dependent term: (C / Re)^n
    K_contraction_Re = np.zeros(Re_throat.size, dtype=np.float64)
    np.divide(C, Re_throat, where=mask_C, out=K_contraction_Re)
    K_contraction_Re **= n

    # Geometric term: C_geom^n
    K_contraction_geom = np.zeros(Re_throat.size, dtype=np.float64)
    np.power(K_contraction_geom_raw, n, where=mask_C, out=K_contraction_geom)

    K_contraction_total = K_contraction_Re + K_contraction_geom

    # === 6. Outlet expansion loss coefficient (E-term) ===
    # NOTE: K_expansion_Re and K_expansion_geom are raised to power m independently, then summed
    beta_outlet_sq = r_throat_sq / r_outlet**2
    K_expansion_geom_raw = 1.0 - beta_outlet_sq

    mask_E = has_flow & (K_expansion_geom_raw > 0)

    # Reynolds-dependent term: (E / Re)^m
    K_expansion_Re = np.zeros(Re_throat.size, dtype=np.float64)
    np.divide(E, Re_throat, where=mask_E, out=K_expansion_Re)
    K_expansion_Re **= m

    # Geometric term: E_geom^m
    K_expansion_geom = np.zeros(Re_throat.size, dtype=np.float64)
    np.power(K_expansion_geom_raw, m, where=mask_E, out=K_expansion_geom)

    K_expansion_total = K_expansion_Re + K_expansion_geom

    # === 7. Total inertial resistance correction ===
    inertial_correction = kinetic_prefactor * (K_contraction_total + K_expansion_total)

    # === 8. Momentum flux change due to cross-section variation ===
    mask_all = mask_C & mask_E
    momentum_change = (
        gamma
        / (2.0 * np.pi**2)
        * density
        * abs_flux
        * (1.0 / r_inlet**4 - 1.0 / r_outlet**4)
    )
    momentum_change[~mask_all] = 0.0

    # === 9. Assemble total conductance ===
    net_inertial_resistance = inertial_correction - momentum_change
    mask_invalid = np.isfinite(net_inertial_resistance)
    np.bitwise_not(mask_invalid, out=mask_invalid)
    mask_invalid |= net_inertial_resistance < 0
    net_inertial_resistance[mask_invalid] = 0.0
    total_resistance = viscous_resistance + net_inertial_resistance
    conductance = 1.0 / total_resistance

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


def calculate_heat_flow(
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
    output = {}
    grouped = boundary_conditions.groupby("names", as_index=False, sort=False)
    for name, group in grouped:
        ids = np.asarray(group["ids"])
        result = mass_balance_conv(mpn, g_ij, P_profile, ids)
        output[name] = result.sum()
    output["total"] = sum(output.values())
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
    total_information = np.empty((2 * num_throat, 4), dtype=np.int64)
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
