#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:00:02 2022

@author: htmt
"""
import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET
import logging
from typing import Union, Literal, Optional, Any
import copy
import itertools
from pathlib import Path
from collections import defaultdict
import re
from functools import partial
from ..topotool import trim_pore
from ..util import is_inplace
from ._classical_pn_keys import Pore1_names, Pore2_names, Throat1_names, Throat2_names

logger = logging.getLogger(__name__)

type_mpn = dict
type_str = Union[str, Path]


def load_Statoil(path: type_str, name: type_str = None, prefix: type_str = None):
    name = name if name != None else prefix
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
    mpn["throat._id"] = np.arange(num_throat)
    mpn["throat.label"] = mpn["throat._id"].copy()
    mpn["throat.all"] = np.ones(num_throat, dtype=bool)
    mpn["throat.inside"] = ~((inThroats | outThroats).astype(bool))
    inlets_index = throat_conns_pore_1[inThroats]
    # print(inlets_index)
    mpn["pore.inlets"] = ~mpn["pore.all"]
    mpn["pore.inlets"][inlets_index] = True
    outlets_index = throat_conns_pore_1[outThroats]
    mpn["pore.outlets"] = ~mpn["pore.all"]
    mpn["pore.outlets"][outlets_index] = True
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
        mpn["throat.real_shape_factor"] = mpn["throat.shape_factor"]
        mpn["throat.real_shape_factor"][
            (mpn["throat.shape_factor"] > BndG1) & (mpn["throat.shape_factor"] <= BndG2)
        ] = (1 / 16)
        mpn["throat.real_shape_factor"][(mpn["throat.shape_factor"] > BndG2)] = (
            1 / 4 / np.pi
        )
        mpn["pore.real_shape_factor"] = mpn["pore.shape_factor"]
        mpn["pore.real_shape_factor"][
            (mpn["pore.shape_factor"] > BndG1) & (mpn["pore.shape_factor"] <= BndG2)
        ] = (1 / 16)
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
        throats_trim = np.concatenate((inlets_index, outlets_index), axis=0)
        mpn = trim_pore(mpn, throats=throats_trim, remove_iso_pore=False)
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

    max_label = len(mpn["pore.all"]) - 1

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
    side: Literal["x-", "x+", "y-", "y+", "z-", "z+"],
    imsize,
    resolution,
) -> float:
    """
    calculate the zoom value for the given side
    Parameters
    ----------
    mpn : type_mpn
        The network.
    side : Literal['x-', 'x+', 'y-', 'y+', 'z-', 'z+']
        The side of the network.
    imsize : tuple
        The size of the image.
    resolution : float
        The resolution of the network.
    Returns
    -------
    float
        The zoom value for the given side.
    """
    if side in ["x-", "x+"]:
        value = (
            imsize[0]
            * imsize[1]
            * resolution**2
            / np.sum(
                mpn["pore.radius"][mpn["pore.boundary_" + side + "_surface"]] ** 2
                * np.pi
            )
        )
    elif side in ["y-", "y+"]:
        value = (
            imsize[0]
            * imsize[2]
            * resolution**2
            / np.sum(
                mpn["pore.radius"][mpn["pore.boundary_" + side + "_surface"]] ** 2
                * np.pi
            )
        )
    elif side in ["z-", "z+"]:
        value = (
            imsize[1]
            * imsize[2]
            * resolution**2
            / np.sum(
                mpn["pore.radius"][mpn["pore.boundary_" + side + "_surface"]] ** 2
                * np.pi
            )
        )
    else:
        raise ValueError("Wrong side name")
    return value


def get_program_parameters(path: type_str):
    import argparse

    description = "Read a VTK XML PolyData file."
    epilogue = """"""
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilogue,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--filename", default=path)
    args = parser.parse_args()
    return args.filename


def vtk2network(path: str, keep_label: bool = False) -> dict:
    """
    Read a VTK XML PolyData file and convert it to a network dictionary.

    Parameters
    ----------
    path : str
        The path of the VTK XML PolyData file.
    keep_label : bool, optional
        Whether to keep the original label prefixes, by default False.
        If False, labels will be converted to boolean arrays.

    Returns
    -------
    dict
        A dictionary containing network data with keys like 'pore.coords',
        'throat.conns', and other extracted properties.
    """
    get_dtype_numpy = partial(get_dtype, target="numpy")
    path_vtk = Path(path)
    root = ET.parse(path_vtk)
    mpn = {}

    # Precompile regex patterns for better performance
    LABEL_PATTERN = re.compile(r"^(.*?)(pore\.|throat\.)(.*)$")

    def find_nodes_by_tags(root, tags):
        """Find all nodes with specified tags and return as a dictionary."""
        result = defaultdict(list)
        for node in root.iter():
            if node.tag in tags:
                result[node.tag].append(node)
        return result

    # Find relevant nodes in the VTK file
    nodes_dict = find_nodes_by_tags(
        root, {"Points", "Lines", "PointData", "CellData", "Cells"}
    )

    # Process data arrays (PointData and CellData)
    for data_node in itertools.chain(nodes_dict["PointData"], nodes_dict["CellData"]):
        for data in data_node:
            attrib = data.attrib
            name = attrib.get("Name", "Unknown_Name")
            dtype = get_dtype_numpy(attrib["type"])
            array = np.fromstring(data.text, dtype=dtype, sep=" ")

            # Handle duplicate names by adding suffix
            if name in mpn:
                suffix = 1
                while f"{name}_{suffix}" in mpn:
                    suffix += 1
                name = f"{name}_{suffix}"

            mpn[name] = array

    # Process points (coordinates)
    if "Points" in nodes_dict:
        points = nodes_dict["Points"][0][0]  # Assuming single Points node
        attrib = points.attrib
        coords = np.fromstring(
            points.text, dtype=get_dtype_numpy(attrib["type"]), sep=" "
        )
        mpn["pore.coords"] = coords.reshape((3, -1), order="F").T

    # Process connections (Lines or Cells)
    for conn_node in itertools.chain(nodes_dict["Lines"], nodes_dict["Cells"]):
        for conn in conn_node:
            if conn.attrib.get("Name") == "connectivity":
                conns = np.fromstring(
                    conn.text, dtype=get_dtype_numpy(conn.attrib["type"]), sep=" "
                )
                if len(conns) % 2 == 0:
                    mpn["throat.conns"] = conns.reshape((-1, 2))
                else:
                    mpn["throat.conns"] = conns
                    print("Warning: connectivity array has odd length, skip reshaping")

    # Process labels and key names
    for key in list(mpn.keys()):
        match = LABEL_PATTERN.match(key)
        if not match:
            print(f"Warning: Cannot find pore or throat prefix for {key}, skipping")
            continue

        prefix, label_type, prop_name = match.groups()

        # Convert label arrays to boolean
        if "label" in prefix.lower():
            mpn[key] = mpn[key].astype(bool)

        # Rename keys if not keeping original labels
        if not keep_label:
            new_key = f"{label_type}{prop_name}"
            mpn[new_key] = mpn.pop(key)

    return mpn


def get_dtype(dtype, target: Literal["numpy", "VTK"]):
    # Create a mapping between numpy and VTK dtypes
    if not hasattr(get_dtype, "dtype_map_VTK") or not hasattr(get_dtype, "dtype_numpy"):
        get_dtype.dtype_map_VTK = {
            np.dtype("bool_"): "Bit",
            # np.dtype("bool_"): "In8",
            np.dtype("i1"): "Int8",
            np.dtype("i2"): "Int16",
            np.dtype("i4"): "Int32",
            np.dtype("i8"): "Int64",
            np.dtype("u1"): "UInt8",
            np.dtype("u2"): "UInt16",
            np.dtype("u4"): "UInt32",
            np.dtype("u8"): "UInt64",
            np.dtype("f4"): "Float32",
            np.dtype("f8"): "Float64",
        }
        get_dtype.dtype_numpy = {
            value: key for key, value in get_dtype.dtype_map_VTK.items()
        }

    if target == "VTK":
        dtype_map_VTK = get_dtype.dtype_map_VTK
        dtype_VTK = dtype_map_VTK.get(dtype, None)
        if dtype_VTK is None:
            raise ValueError(
                f"Unsupported dtype: {dtype} , dtype should be one of {list(dtype_map_VTK.keys())}"
            )
        return dtype_VTK
    elif target == "numpy":
        dtype_map_numpy = get_dtype.dtype_numpy
        dtype_numpy = dtype_map_numpy.get(dtype, None)
        if dtype_numpy is None:
            raise ValueError(
                f"Unsupported dtype: {dtype} , dtype should be one of {list(dtype_map_numpy.values())}"
            )
        return dtype_numpy
    else:
        raise ValueError(f"Unsupported target: {target}")


logger = logging.getLogger(__name__)


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
    dtype_vtk = get_dtype(array.dtype, target="VTK")

    # Handle boolean arrays (convert to uint8)
    if dtype_vtk == "Bit":
        array = array.astype(np.uint8)
        dtype_vtk = "Int8"

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
    fill_nans: Optional[float] = None,
    fill_infs: Optional[float] = None,
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
    filtered_keys = [
        k
        for k in mpn.keys()
        if k not in {"properties", "inner_info", "inner_start2end"}
    ]

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
    points_node.append(array_to_element("coords", points.T.ravel("F"), n=3))

    # Add lines/connections data
    lines_node = ET.SubElement(piece, "Lines")
    lines_node.append(array_to_element("connectivity", pairs))
    lines_node.append(
        array_to_element(
            "offsets", np.arange(start=2, stop=num_throats * 2 + 2, step=2)
        )
    )

    # Prepare data containers
    point_data = ET.SubElement(piece, "PointData")
    cell_data = ET.SubElement(piece, "CellData")

    # Process remaining network properties
    for key in filtered_keys:
        array = mpn[key]

        # Skip object arrays
        if array.dtype == object:
            logger.warning(f"{key} has dtype object and will not be written to file")
            continue

        # Determine property type and format key
        prop_type = "label" if array.dtype == bool else "properties"
        vtk_key = f"{network_name} | network | {prop_type} || {key}"

        # Handle NaN and Inf values
        if np.isnan(array.sum()):
            if fill_nans is None:
                logger.warning(f"{key} contains NaNs and will not be written to file")
                continue
            array[np.isnan(array)] = fill_nans

        if np.isinf(array.sum()):
            if fill_infs is None:
                logger.warning(
                    f"{key} contains infinities and will not be written to file"
                )
                continue
            array[np.isinf(array)] = fill_infs

        # Create element and add to appropriate section
        element = array_to_element(vtk_key, array)

        if array.size == num_points:
            point_data.append(element)
        elif array.size == num_throats:
            cell_data.append(element)
        else:
            logger.warning(
                f"{key} has incorrect size ({array.size}) and will not be written to file"
            )

    # Write to file with proper formatting
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
