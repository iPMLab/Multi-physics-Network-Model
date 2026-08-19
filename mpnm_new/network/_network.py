#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:00:02 2022

@author: htmt
"""

import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET
import warnings

from typing import Union, Literal, Optional, Any, List
import copy
import itertools
from pathlib import Path
from collections import defaultdict
import re
from functools import partial
from ..topotool import trim_pore
from ..util import is_inplace, to_list
from ._classical_pn_keys import Pore1_names, Pore2_names, Throat1_names, Throat2_names


type_mpn = dict
type_str = Union[str, Path]


def coordination_number(mpn: type_mpn):
    pore_conns = np.bincount(
        mpn["throat.conns"].reshape(-1), minlength=mpn["pore.all"].size
    )
    return pore_conns


def load_Statoil(path: type_str, name: type_str = None, prefix: type_str = None):
    name = prefix if name is None else name
    Path_mpnm = Path(path)
    Path_node1 = Path_mpnm / f"{name}_node1.dat"
    Path_node2 = Path_mpnm / f"{name}_node2.dat"
    Path_link1 = Path_mpnm / f"{name}_link1.dat"
    Path_link2 = Path_mpnm / f"{name}_link2.dat"

    Pores1 = pd.read_csv(
        Path_node1,
        skiprows=1,
        usecols=[0, 1, 2, 3, 4],
        sep=r"\s+",
        names=Pore1_names,
    )
    Pores2 = pd.read_csv(Path_node2, names=Pore2_names, sep=r"\s+")
    Pores = pd.merge(Pores1, Pores2, on=["pore._id"], how="inner")
    Throats1 = pd.read_csv(Path_link1, skiprows=1, names=Throat1_names, sep=r"\s+")
    Throats2 = pd.read_csv(Path_link2, names=Throat2_names, sep=r"\s+")
    Throats = pd.merge(
        Throats1,
        Throats2,
        on=["throat._id", "throat.pore_1_index", "throat.pore_2_index"],
        how="inner",
    )
    Pores = Pores.astype({"pore._id": np.int64, "pore.connection_number": np.int64})
    Throats = Throats.astype(
        {
            "throat._id": np.int64,
            "throat.pore_1_index": np.int64,
            "throat.pore_2_index": np.int64,
        }
    )
    return Pores, Throats


def dfs2pn(
    Pores, Throats, calculate_real_shape_factor=True, remove_in_out_throats=True
):
    Pores = Pores[Pores.loc[:, "pore.connection_number"] >= 0]
    pn_o = {col: Pores[col].to_numpy() for col in Pores.columns}
    pn_o.update({col: Throats[col].to_numpy() for col in Throats.columns})

    num_pore = pn_o["pore._id"].size
    num_throat = pn_o["throat._id"].size
    mpn = {}
    mpn["pore._id"] = np.arange(num_pore)
    mpn["pore.label"] = mpn["pore._id"].copy()
    mpn["pore.connection_number"] = pn_o["pore.connection_number"]
    mpn["pore.all"] = np.ones(num_pore, dtype=bool)
    mpn["pore.volume"] = pn_o["pore.volume"]
    mpn["pore.radius"] = pn_o["pore.radius"]
    mpn["pore.shape_factor"] = pn_o["pore.shape_factor"]
    mpn["pore.coords"] = np.column_stack(
        (pn_o["pore.x"], pn_o["pore.y"], pn_o["pore.z"])
    )
    mpn["throat.conns"] = np.column_stack(
        (pn_o["throat.pore_1_index"], pn_o["throat.pore_2_index"])
    )
    # print(mpn["throat.conns"])
    mpn["throat.conns"] = np.sort(
        (mpn["throat.conns"] - 1).astype(np.int32, copy=False), axis=1
    )
    throat_conns_pore_0 = mpn["throat.conns"][:, 0]
    throat_conns_pore_1 = mpn["throat.conns"][:, 1]
    inThroats = throat_conns_pore_0 == -2
    outThroats = throat_conns_pore_0 == -1
    throat_inlets = inThroats.nonzero()[0]
    throat_outlets = outThroats.nonzero()[0]
    mpn["throat._id"] = np.arange(num_throat)
    mpn["throat.label"] = mpn["throat._id"].copy()
    mpn["throat.all"] = np.ones(num_throat, dtype=bool)
    mpn["throat.inside"] = ~(inThroats | outThroats)
    pore_inlets = throat_conns_pore_1[inThroats]
    # print(inlets_index)
    mpn["pore.inlets"] = ~mpn["pore.all"]
    mpn["pore.inlets"][pore_inlets] = True
    pore_outlets = throat_conns_pore_1[outThroats]
    mpn["pore.outlets"] = ~mpn["pore.all"]
    mpn["pore.outlets"][pore_outlets] = True
    mpn["throat.radius"] = pn_o["throat.radius"]
    mpn["throat.shape_factor"] = pn_o["throat.shape_factor"]
    mpn["throat.length"] = pn_o["throat.length"]
    mpn["throat.total_length"] = pn_o["throat.total_length"]
    mpn["throat.conduit_lengths_pore1"] = pn_o["throat.conduit_lengths_pore1"]
    mpn["throat.conduit_lengths_pore2"] = pn_o["throat.conduit_lengths_pore2"]
    mpn["throat.conduit_lengths_throat"] = pn_o["throat.length"]
    mpn["throat.area"] = (mpn["throat.radius"] ** 2) / (
        4.0 * mpn["throat.shape_factor"]
    )
    mpn["pore.area"] = (mpn["pore.radius"] ** 2) / (4.0 * mpn["pore.shape_factor"])
    mpn["pore.solid"] = np.zeros(num_pore, dtype=bool)

    if calculate_real_shape_factor:
        BndG1 = np.sqrt(3) / 36 + 0.00001
        BndG2 = 0.07
        mpn["throat.real_shape_factor"] = mpn["throat.shape_factor"].copy()
        mpn["throat.real_shape_factor"][
            (mpn["throat.shape_factor"] > BndG1) & (mpn["throat.shape_factor"] <= BndG2)
        ] = 1 / 16
        mpn["throat.real_shape_factor"][(mpn["throat.shape_factor"] > BndG2)] = (
            1 / 4 / np.pi
        )
        mpn["pore.real_shape_factor"] = mpn["pore.shape_factor"].copy()
        mpn["pore.real_shape_factor"][
            (mpn["pore.shape_factor"] > BndG1) & (mpn["pore.shape_factor"] <= BndG2)
        ] = 1 / 16
        mpn["pore.real_shape_factor"][(mpn["pore.shape_factor"] > BndG2)] = (
            1 / 4 / np.pi
        )
        mpn["throat.real_k"] = mpn["throat.all"] * 0.6
        mpn["throat.real_k"][
            (mpn["throat.shape_factor"] > BndG1) & (mpn["throat.shape_factor"] <= BndG2)
        ] = 0.5623
        mpn["throat.real_k"][(mpn["throat.shape_factor"] > BndG2)] = 0.5
        mpn["pore.real_k"] = mpn["pore.all"] * 0.6
        mpn["pore.real_k"][
            (mpn["pore.shape_factor"] > BndG1) & (mpn["pore.shape_factor"] <= BndG2)
        ] = 0.5623
        mpn["pore.real_k"][(mpn["pore.shape_factor"] > BndG2)] = 0.5
    mpn["throat.area"] = (mpn["throat.radius"] ** 2) / (
        4.0 * mpn["throat.shape_factor"]
    )
    mpn["pore.area"] = (mpn["pore.radius"] ** 2) / (4.0 * mpn["pore.shape_factor"])
    mpn["pore.solid"] = np.zeros(num_pore, dtype=bool)
    if remove_in_out_throats:
        throats_trim = np.concatenate((throat_inlets, throat_outlets))
        trim_pore(mpn, throats=throats_trim, inplace=True)
    return mpn


def read_network(
    path: type_str,
    name: type_str = None,
    prefix: type_str = None,
    calculate_real_shape_factor: bool = True,
    remove_in_out_throats: bool = True,
) -> type_mpn:
    """
    Read the network from the given path and return a dictionary containing the network information.


    Parameters
    ----------
    path : type_str
        The path of the network files.
    name : type_str, optional
        The name of the network, by default None.
    prefix : type_str, optional
        The prefix of the network files, by default None.
    calculate_shape_factor : bool, optional
        Whether to calculate the shape factor or not, by default True.
    remove_in_out_throats : bool, optional
        Whether to remove the inlet and outlet throats or not, by default True.
    Returns
    -------
    dict
        A dictionary containing the network information.
    """

    """
    Throats1 = np.loadtxt("test1D_link1.dat", skiprows=1)
    Throats2 = np.loadtxt("test1D_link2.dat")
    Pores   = np.loadtxt("test1D_node2.dat")
    Pores1   = np.loadtxt('test1D_node1.dat',skiprows=1, usecols=(0,1,2,3,4))
    #"""
    Pores, Throats = load_Statoil(path, name, prefix)
    mpn = dfs2pn(
        Pores=Pores,
        Throats=Throats,
        calculate_real_shape_factor=calculate_real_shape_factor,
        remove_in_out_throats=remove_in_out_throats,
    )

    return mpn


def read_pypne(pn_pypne, calculate_real_shape_factor=True, remove_in_out_throats=True):
    pn_o = pn_pypne

    Pores = pd.DataFrame({k: v for k, v in pn_o.items() if k.startswith("pore.")})
    Throats = pd.DataFrame({k: v for k, v in pn_o.items() if k.startswith("throat.")})
    mpn = dfs2pn(
        Pores=Pores,
        Throats=Throats,
        calculate_real_shape_factor=calculate_real_shape_factor,
        remove_in_out_throats=remove_in_out_throats,
    )

    return mpn


def mpn2pyg(
    mpn: type_mpn, exclude_key: list = None, require_coords: bool = True, norm=True
):
    import torch
    from torch_geometric.data import Data

    if exclude_key is None:
        exclude_key = []

    exclude_key.extend(["pore.coords", "throat.conns", "throat.all", "pore.all"])
    pore_data = []
    throat_data = []
    if require_coords:
        pore_data.append(mpn["pore.coords"][:, 0])
        pore_data.append(mpn["pore.coords"][:, 1])
        pore_data.append(mpn["pore.coords"][:, 2])
    for key in mpn.keys():
        if key not in exclude_key:
            if key.startswith("pore."):
                if np.std(mpn[key]) == 0:
                    continue
                pore_data.append(mpn[key])
            elif key.startswith("throat."):
                if np.std(mpn[key]) == 0:
                    continue
                throat_data.append(mpn[key])
    x_np = np.asarray(pore_data).T
    edge_attr_np = np.asarray(throat_data).T
    if norm:
        mean_x = np.mean(x_np, axis=0)
        std_x = np.std(x_np, axis=0)
        x_np = (x_np - mean_x) / std_x
        edg_mean = np.mean(edge_attr_np, axis=0)
        edg_std = np.std(edge_attr_np, axis=0)
        edge_attr_np = (edge_attr_np - edg_mean) / edg_std
    edge_index = torch.tensor(mpn["throat.conns"].T, dtype=torch.long)
    x = torch.tensor(x_np, dtype=torch.float32)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if norm:
        return data, mean_x, std_x, edg_mean, edg_std
    else:
        return data


def read_surface_area(
    mpn: type_mpn,
    path: type_str,
    index_col=0,
    resolution: float = 1.0,
    labels: Union[list, tuple] = ("x-", "x+", "y-", "y+", "z-", "z+"),
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
    Parameters
    ----------
    mpn : type_mpn
        The network.
    path : type_str
        The path of the surface area file.
    index_col : int, optional
        The index column of the surface area file, by default 0.
    resolution : float, optional
        The resolution of the network, by default 1.
    labels : Union[list, tuple], optional
        The labels of the surface area, by default ('x-', 'x+', 'y-', 'y+', 'z-', 'z+').
    inplace : bool, optional
        Whether to modify the network in place or return a new one, by default True.
    Returns
    -------
    type_mpn
        The modified network.
    """
    mpn = is_inplace(mpn, inplace)
    if "pore.all_surface" not in mpn.keys():
        mpn["pore.all_surface"] = np.zeros_like(mpn["pore.all"], dtype=bool)
    else:
        pass
    if labels is None:
        labels = ("x-", "x+", "y-", "y+", "z-", "z+")
    axis_tuple = ("x-", "x+", "y-", "y+", "z-", "z+")
    Boundaries_areas = pd.read_csv(path, index_col=index_col).to_numpy()
    Boundaries_areas[:, 0] = Boundaries_areas[:, 0] - 1

    max_label = mpn["pore.all"].size - 1

    Boundaries_areas = Boundaries_areas[Boundaries_areas[:, 0] <= max_label]
    Boundaries_labels = Boundaries_areas[:, 0]
    colm_bools_list = []  # each column > 0
    colm_labels_list = []  # each column labels
    colm_areas_list = []  # each column areas
    mpn_pore_which_surface_area_list = []  # each mpn[pore.which_surface_area]
    mpn_pore_which_surface_list = []  # each mpn[pore.which_surface]
    for i in range(len(axis_tuple)):
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

    for i in range(len(labels)):
        mpn["pore." + labels[i]] = mpn_pore_which_surface_list[i]
        mpn["pore.all_surface"][colm_labels_list[i]] = True
        mpn["pore." + labels[i] + "_area"] = (
            mpn_pore_which_surface_area_list[i] * resolution**2
        )

    return mpn


def getting_zoom_value(
    mpn: type_mpn,
    sides: Union[
        Literal["x-", "x+", "y-", "y+", "z-", "z+"],
        List[Literal["x-", "x+", "y-", "y+", "z-", "z+"]],
    ],
    boundary_surfaces: Union[List[str], str],
    imsize: tuple,
    resolution: float,
) -> np.ndarray:
    """
    Calculate the zoom value(s) for the given side(s).

    Parameters
    ----------
    mpn : dict-like
        The network data containing at least 'pore.radius'.
    sides : str or list of str
        Side(s) of the network. One or more of: 'x-', 'x+', 'y-', 'y+', 'z-', 'z+'.
    boundary_surfaces : str or list of str
        Corresponding boundary surface label(s) in `mpn`.
    imsize : tuple of int
        Image size as (depth, height, width) i.e., (z, y, x).
    resolution : float
        Physical resolution (e.g., meters per voxel).

    Returns
    -------
    np.ndarray of float
        Zoom value(s) for the given side(s).
    """

    imsize_arr = np.asarray(imsize)
    assert imsize_arr.size == 3 and np.issubdtype(imsize_arr.dtype, np.integer), (
        "imsize must be a 3-tuple of integers"
    )

    sides_list = to_list(sides)
    boundary_surfaces_list = to_list(boundary_surfaces)

    assert len(boundary_surfaces_list) == len(sides_list), (
        f"Length of boundary_surfaces ({len(boundary_surfaces_list)}) "
        f"must match length of sides ({len(sides_list)})"
    )

    imsize_dict = {"z": imsize_arr[0], "y": imsize_arr[1], "x": imsize_arr[2]}
    physical_size = {k: v * resolution for k, v in imsize_dict.items()}

    axis_from_side = {side: side[0] for side in sides_list}  # first char is axis

    zoom_values = []
    for side, bnd in zip(sides_list, boundary_surfaces_list):
        axis = axis_from_side[side]

        area_real = np.prod([physical_size[k] for k in physical_size if k != axis])

        radii = mpn["pore.radius"][mpn[bnd]]
        total_pore_area = np.sum(radii**2 * np.pi)

        zoom = area_real / total_pore_area
        zoom_values.append(zoom)

    return np.array(zoom_values, dtype=np.float64)


def get_dtype(
    dtype,
    target: Literal["numpy", "vtk"],
    _cache={},
):
    # _np_vtk = {
    #     numpy.uint8: vtkConstants.VTK_UNSIGNED_CHAR,
    #     numpy.uint16: vtkConstants.VTK_UNSIGNED_SHORT,
    #     numpy.uint32: vtkConstants.VTK_UNSIGNED_INT,
    #     numpy.uint64: vtkConstants.VTK_UNSIGNED_LONG_LONG,
    #     numpy.int8: vtkConstants.VTK_SIGNED_CHAR,
    #     numpy.int16: vtkConstants.VTK_SHORT,
    #     numpy.int32: vtkConstants.VTK_INT,
    #     numpy.int64: vtkConstants.VTK_LONG_LONG,
    #     numpy.float32: vtkConstants.VTK_FLOAT,
    #     numpy.float64: vtkConstants.VTK_DOUBLE,
    #     numpy.complex64: vtkConstants.VTK_FLOAT,
    #     numpy.complex128: vtkConstants.VTK_DOUBLE,
    # }
    if not _cache:
        dtype_map_vtk = {
            np.dtype("bool_"): "Int8",
            # np.dtype("u1"): "UInt8",
            np.dtype("uint8"): "UInt8",
            np.dtype("uint16"): "UInt16",
            np.dtype("uint32"): "UInt32",
            np.dtype("uint64"): "UInt64",
            np.dtype("int16"): "Int16",
            np.dtype("int32"): "Int32",
            np.dtype("int64"): "Int64",
            np.dtype("float32"): "Float32",
            np.dtype("float64"): "Float64",
        }
        dtype_map_numpy = {v: k for k, v in dtype_map_vtk.items()}
        _cache.update(vtk=dtype_map_vtk, numpy=dtype_map_numpy)

    if target.lower() == "vtk":
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        result = _cache["vtk"].get(dtype)
        assert result is not None, (
            f"Unsupported NumPy dtype: {dtype}. Supported: {list(_cache['vtk'].keys())}"
        )
        return result

    elif target.lower() == "numpy":
        result = _cache["numpy"].get(dtype)
        assert result is not None, (
            f"Unsupported VTK dtype: {dtype}. Supported: {list(_cache['numpy'].keys())}"
        )
        return result

    else:
        raise ValueError(f"Unsupported target: {target}")


def vtk2network(path: str, keep_label: bool = False) -> dict:

    root = ET.parse(Path(path)).getroot()
    mpn = {}

    def extract_data_arrays(da):
        """Extract all DataArray under PointData or CellData."""
        name = da.get("Name", "Unknown")
        dtype = get_dtype(da.get("type"), target="numpy")
        arr = np.fromstring(da.text, dtype=dtype, sep=" ")
        return name, arr

    node_map = {
        p.tag: c.tag
        for p, c in zip(root.findall(".//DataArray/.."), root.findall(".//DataArray"))
    }
    node_map = {"PointData": [], "CellData": [], "Points": [], "Cells": [], "Lines": []}

    da_parents = root.findall(".//DataArray/..")
    da_names_counts = defaultdict(int)
    for da_parent in da_parents:
        for da in da_parent.findall("DataArray"):
            da_name = da.get("Name", "Unknown")
            if da_names_counts[da_name] > 0:
                da.set("Name", f"{da_name}_{da_names_counts[da_name]}")
            da_names_counts[da_name] += 1
            node_map[da_parent.tag].append(da)

    node_map["Lines"].extend(node_map.pop("Cells"))

    # --- Process PointData (pore properties) ---
    for da in node_map["PointData"]:
        name, arr = extract_data_arrays(da)
        if name not in mpn:
            mpn[name] = arr
        else:
            warnings.warn(
                f"Multiple {name} found, keeping the first one. Node tag: {da.tag}",
                category=UserWarning,
                stacklevel=1,
            )

    # --- Process CellData (throat properties) ---
    for da in node_map["CellData"]:
        name, arr = extract_data_arrays(da)
        if name not in mpn:
            mpn[name] = arr
        else:
            warnings.warn(
                f"Multiple {name} found, keeping the first one. Node tag: {da.tag}",
                category=UserWarning,
                stacklevel=1,
            )
            mpn[name] = arr

    # --- Process Points (coordinates) ---
    for da in node_map["Points"]:
        name, arr = extract_data_arrays(da)
        arr = np.ascontiguousarray(arr.reshape((-1, 3)))
        if "pore.coords" not in mpn:
            mpn["pore.coords"] = arr
        else:
            warnings.warn(
                f"Multiple pore.coords found, keeping the first one. Node tag: {da.tag}",
                category=UserWarning,
                stacklevel=1,
            )

    # --- Process Lines (connections) ---
    for da in node_map["Lines"]:
        name = da.get("Name", "Unknown")
        if name == "connectivity":
            name, arr = extract_data_arrays(da)
            if arr.size % 2 == 0:
                mpn["throat.conns"] = np.ascontiguousarray(arr.reshape((-1, 2)))
            else:
                warnings.warn(
                    "Connectivity array has odd length; skip reshaping",
                    category=UserWarning,
                    stacklevel=1,
                )
                mpn["throat.conns"] = arr

    # --- Rename keys and process labels ---
    LABEL_PATTERN = re.compile(r"^(.*?)(pore\.|throat\.)(.*)$")
    new_mpn = {}
    mpn_names_counts = defaultdict(int)
    for key, value in mpn.items():
        match = LABEL_PATTERN.match(key)
        if not match:
            print(
                f"Warning: Cannot parse key '{key}', make sure the name of DataArray "
                "is in the format 'pore.prop_name' or 'throat.prop_name', skipping"
            )
            continue

        prefix, label_type, prop_name = match.groups()
        new_key = f"{label_type}{prop_name}" if not keep_label else key

        # Convert label arrays to boolean
        # if "label" in prefix.lower():
        #     value = value.astype(bool)

        # Handle duplicate keys
        if mpn_names_counts[new_key] > 0:
            new_key = f"{new_key}_{mpn_names_counts[new_key]}"
        mpn_names_counts[new_key] += 1

        new_mpn[new_key] = value
    return new_mpn


def array_to_element(name: str, array: np.ndarray, n: int = 1) -> ET.Element:
    """
    Convert numpy array to VTK XML element.

    Parameters
    ----------
    name : str
        The name of the array in the VTK file.
    array : np.ndarray
        The numpy array to convert.
    n : int, optional
        Number of components per value (default: 1).

    Returns
    -------
    ET.Element
        The constructed VTK DataArray element.
    """
    array = np.ascontiguousarray(array)
    assert array.dtype != np.int8, (
        f"Array '{name}' has dtype int8, which is reserved for boolean data (True/False). "
        "Please use another integer dtype (e.g., uint8, int16 ...) for non-boolean integers."
    )
    dtype_vtk = get_dtype(array.dtype, target="vtk")
    if dtype_vtk == "Int8":
        array = array.view(np.int8)

    element = ET.Element(
        "DataArray", {"Name": name, "NumberOfComponents": str(n), "type": dtype_vtk}
    )

    # Optimized array to string conversion
    element.text = "\t".join(array.ravel().astype(str))
    return element


def network2vtk(
    mpn: type_mpn[str, Any],
    filename: str = "test.vtp",
    network_name: str = "mpnm",
    nan: Optional[float] = None,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
    keys_ignore: Optional[List[str]] = None,
) -> None:
    """
    Convert network dictionary to VTK PolyData XML file.

    Parameters
    ----------
    mpn : Dict[str, Any]
        Network dictionary containing 'pore.coords', 'throat.conns' and other arrays.
    filename : str, optional
        Output filename (default: "test.vtp").
    network_name : str, optional
        Prefix for network properties (default: "mpnm").
    fill_nans : float, optional
        Value to replace NaNs with (default: None, skip arrays with NaNs).
    fill_infs : float, optional
        Value to replace infinities with (default: None, skip arrays with infinities).
    """
    # Setup output file path
    _keys_ignore = ["properties", "inner_info", "inner_start2end"]
    if keys_ignore is not None:
        _keys_ignore.extend(keys_ignore)
    keys_ignore = set(_keys_ignore)

    filename = Path(filename)
    if filename.suffix != ".vtp":
        filename = filename.with_name(filename.name + ".vtp")
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Create a working copy of the network
    mpn = mpn.copy()

    # Extract required network components
    points = mpn.pop("pore.coords")
    pairs = mpn.pop("throat.conns")
    num_points = points.shape[0]
    num_throats = pairs.shape[0]

    # Filter out special keys
    filtered_keys = [k for k in mpn.keys() if k not in keys_ignore]

    # Create VTK document structure
    root = ET.Element(
        "VTKFile", {"byte_order": "LittleEndian", "type": "PolyData", "version": "0.1"}
    )

    polydata = ET.SubElement(root, "PolyData")
    piece = ET.SubElement(
        polydata,
        "Piece",
        {"NumberOfPoints": str(num_points), "NumberOfLines": str(num_throats)},
    )

    # Add points data
    points_node = ET.SubElement(piece, "Points")
    points_node.append(array_to_element("coords", points.T.reshape(-1, order="F"), n=3))

    # Add lines/connections data
    lines_node = ET.SubElement(piece, "Lines")
    lines_node.append(array_to_element("connectivity", pairs))
    lines_node.append(
        array_to_element(
            "offsets",
            np.arange(start=2, stop=num_throats * 2 + 2, step=2, dtype=pairs.dtype),
        )
    )

    # Prepare data containers
    point_data = ET.SubElement(piece, "PointData")
    cell_data = ET.SubElement(piece, "CellData")

    # Sort keys
    sorted_items = []
    for key in filtered_keys:
        prop_type = (
            "label" if np.isdtype(mpn[key].dtype, np.dtype("bool")) else "properties"
        )
        vtk_key = f"{network_name} | network | {prop_type} || {key}"
        sorted_items.append((vtk_key, key))

    sorted_items.sort()

    # Process remaining network properties
    for vtk_key, key in sorted_items:
        array = mpn[key]

        # Skip object arrays
        if array.dtype == object:
            warnings.warning(
                f"{key} has dtype object and will not be written to file",
                category=UserWarning,
                stacklevel=1,
            )
            continue

        # Determine property type and format key
        if nan is not None:
            array[np.isnan(array)] = nan
        if posinf is not None:
            array[np.isposinf(array)] = posinf
        if neginf is not None:
            array[np.isneginf(array)] = neginf

        # Create element and add to appropriate section
        element = array_to_element(vtk_key, array)

        if key.startswith("pore."):
            if array.size == num_points:
                point_data.append(element)
            else:
                warnings.warning(
                    f"{key} has incorrect size ({array.size}), expected num_pores = {num_points}, will not be written to file",
                    category=UserWarning,
                    stacklevel=1,
                )
        elif key.startswith("throat."):
            if array.size == num_throats:
                cell_data.append(element)
            else:
                warnings.warning(
                    f"{key} has incorrect size ({array.size}), expected num_throats = {num_throats}, will not be written to file",
                    category=UserWarning,
                    stacklevel=1,
                )
        else:
            warnings.warning(
                f"{key} is not started with 'pore.' or 'throat.' and will not be written to file",
                category=UserWarning,
                stacklevel=1,
            )

    # Write to file with proper formatting
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
