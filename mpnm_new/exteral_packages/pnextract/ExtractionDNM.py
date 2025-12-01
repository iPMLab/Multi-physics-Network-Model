#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:58:31 2024

@author: yjp
"""

import numpy as np
from skimage.measure import regionprops_table
import time
import pandas as pd
import sys
from joblib import Parallel, delayed
import edt
from tqdm import tqdm
from skimage import measure
from scipy import ndimage
import sys

sys.path.append(r"../")
from mpnm_new import network as net
from mpnm_new import util
from functools import partial
from mpnm_new import extraction


def calculate_surface_area(image):
    return (
        extraction.mk_functionals_3d(
            image,
            return_curvature=False,
            return_surface=True,
            return_volume=False,
            return_euler26=False,
        )
        * 8
    )


def unique_uint_nonzero(arr, return_counts=False):
    res = np.unique(arr, return_counts=return_counts)
    return res


# def check_index(index):
#     if np.any(index[1]-index[0]) > 0:
#         return True
#     else:
#         return False


def Mix_image(Path_dict, array_void_VElems, array_solid_VElems):
    """
    mix pore image and solid image, zero of pore and solid is ignored,
    solid label is equal to maximum of pore label + solid image
    """
    Images_dict = Path_dict["Images"]
    pore_label_max = array_void_VElems.max()
    array_solid_VElems = np.where(
        array_solid_VElems > 0, pore_label_max + array_solid_VElems, 0
    )
    array_mix = array_void_VElems + array_solid_VElems
    array_mix = array_mix.astype(np.int32)
    if Images_dict.get("mix") is not None:
        array_mix.tofile(Images_dict["mix"])
    values = unique_uint_nonzero(array_mix)
    solid_labels = unique_uint_nonzero(array_solid_VElems)
    pore_labels = unique_uint_nonzero(array_void_VElems)
    labels = np.concatenate((pore_labels, solid_labels))
    mix_image = np.where(array_mix > 0, array_mix, 0)
    solid_image = np.where(array_solid_VElems > 0, array_solid_VElems, 0)
    pore_image = np.where(array_void_VElems > 0, array_void_VElems, 0)
    if len(values) != len(labels):
        print("Error")
        sys.exit()
    else:
        print("No Error")
    if np.all(values == np.arange(1, len(values) + 1)):
        print("Labels are continuous")
    else:
        print("Warning! Labels are not continuous")
        labels_lost = list(set(values) - set(np.arange(1, len(values) + 1)))
        print(labels_lost)
    return mix_image, labels, solid_image, solid_labels, pore_image, pore_labels


def get_phase_props_table(labeled_image):
    """
    Conclude the properties of different labels in labeled image
    """
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
    phase_props_table.rename(columns=name_map, inplace=True)
    phase_props_table = phase_props_table.to_numpy(dtype=np.int32)
    return phase_props_table


def find(i, phase_props_table, offset=2):
    """
    Index of label is label-1
    """
    pd_index = i - 1
    index = np.empty((2, 3), dtype=np.int32)
    index[0] = phase_props_table[pd_index, 1:4] - offset - 1
    index[1] = phase_props_table[pd_index, 4:7] + offset
    index[0] = np.clip(index[0], 0, None)
    return index


def find_center(i, phase_props_table, image, n_workers_dt=10):
    index = find(i, phase_props_table)
    zyx_min = index[0]
    zyx_max = index[1]
    region = image[
        zyx_min[0] : zyx_max[0], zyx_min[1] : zyx_max[1], zyx_min[2] : zyx_max[2]
    ]
    region_bool = region == i
    distance_map = edt.edt(region_bool, parallel=n_workers_dt)
    # centerindex=np.round(np.mean(np.argwhere(distance_map==distance_map.max()),axis=0)).astype(int)
    centerindex = np.unravel_index(np.argmax(distance_map), distance_map.shape)
    radius = distance_map[centerindex[0], centerindex[1], centerindex[2]]
    if radius < 1:
        radius = 1

    centerindex += index[0]
    result = np.empty(5)
    result[0] = i
    result[1:4] = centerindex
    result[4] = radius
    return result


def calc_volum(i, phase_props_table, image):
    pd_index = i - 1  # index is 1 smaller than i
    volume = phase_props_table[pd_index, 7]
    index = find(i, phase_props_table)
    zyx_min = index[0]
    zyx_max = index[1]
    region = image[
        zyx_min[0] : zyx_max[0], zyx_min[1] : zyx_max[1], zyx_min[2] : zyx_max[2]
    ]
    region_bool = region == i
    region_bool = np.pad(
        region_bool,
        ((5, 5), (5, 5), (5, 5)),
        "constant",
        constant_values=(False, False),
    )
    # verts, faces, normals, values = measure.marching_cubes(region_bool)
    # area = measure.mesh_surface_area(verts, faces)
    area = calculate_surface_area(region_bool)
    return np.array([i, volume, area])


def calc_surf(i, phase_props_table, image):
    index = find(i, phase_props_table)
    zyx_min = index[0]
    zyx_max = index[1]
    region = image[
        zyx_min[0] : zyx_max[0], zyx_min[1] : zyx_max[1], zyx_min[2] : zyx_max[2]
    ]

    region = np.pad(
        region, ((5, 5), (5, 5), (5, 5)), "constant", constant_values=(False, False)
    )
    region_mask = region == i

    # verts, faces, normals, values = measure.marching_cubes(region_mask)
    # area_i = measure.mesh_surface_area(verts, faces)
    area_i = calculate_surface_area(region_mask)
    structure1 = np.ones((3, 3, 3), dtype=bool)
    region_dilated_mask = ndimage.binary_dilation(region_mask, structure=structure1)
    values = region[region_dilated_mask & (~region_mask)]
    values = unique_uint_nonzero(values)
    if len(values) == 0:
        table = np.array([i, -1, -1])

    else:
        table = np.empty((len(values), 3))
        for j in range(len(values)):
            value_j = values[j]
            # verts_j, faces_j, normals_j, values_j = measure.marching_cubes(
            #     region == value_j
            # )
            # area_j = measure.mesh_surface_area(verts_j, faces_j)

            # verts_total, faces_total, normals_total, values_total = measure.marching_cubes(
            #     np.isin(region, (i, value_j))
            # )
            # area_total = measure.mesh_surface_area(verts_total, faces_total)
            area_j = calculate_surface_area(region == value_j)
            area_total = calculate_surface_area(
                np.isin(region, (i, value_j), kind="table")
            )
            table[j, 0] = i
            table[j, 1] = values[j]
            table[j, 2] = (area_i + area_j - area_total) / 2
        table = table[table[:, 2] > 0]

    return table


def find_node_center(
    csv_dir,
    phase_props_table,
    image,
    labels=None,
    resolution=1e-5,
    n_workers=32,
    n_workers_dt=10,
    backend="loky",
):
    t0 = time.time()
    labels = unique_uint_nonzero(image) if labels is None else labels
    # arrays = Parallel(n_jobs=n_workers,prefer='threads')(delayed(find_center)(i,phase_props_table,image,n_workers_dt) for i in tqdm(labels))
    arrays = Parallel(n_jobs=n_workers, backend=backend)(
        delayed(find_center)(i, phase_props_table, image, n_workers_dt)
        for i in tqdm(labels)
    )
    tend = time.time()
    cen_table = pd.DataFrame(
        np.vstack(arrays), columns=["index", "z", "y", "x", "radius"]
    )
    cen_table = cen_table.loc[:, ["index", "x", "y", "z", "radius"]]
    cen_table = cen_table.astype(
        {
            "index": "int32",
            "x": "int32",
            "y": "int32",
            "z": "int32",
            "radius": "float16",
        }
    )
    cen_table.to_csv(csv_dir)
    print("running time of finding center: %.6fs" % (tend - t0))


def find_node_volume(
    csv_dir,
    phase_props_table,
    image,
    labels=None,
    resolution=1e-5,
    n_workers=32,
    backend="loky",
):
    t0 = time.time()
    labels = unique_uint_nonzero(image) if labels is None else labels
    labels = labels[labels > 0]
    result_v = Parallel(n_jobs=n_workers, backend=backend)(
        delayed(calc_volum)(i, phase_props_table, image) for i in tqdm(labels)
    )
    result_v = np.vstack(result_v).astype(np.int32)

    result_v = pd.DataFrame(result_v, columns=["index", "volume", "area"])

    result_v.to_csv(csv_dir)
    tend = time.time()
    print("running time of calculation volume: %.6fs" % (tend - t0))


def find_interface(
    csv_dir,
    phase_props_table,
    image,
    labels=None,
    resolution=1e-5,
    n_workers=32,
    backend="loky",
):
    t0 = time.time()
    labels = unique_uint_nonzero(image) if labels is None else labels
    arrays = Parallel(n_jobs=n_workers, backend=backend)(
        delayed(calc_surf)(i, phase_props_table, image) for i in tqdm(labels)
    )
    table = np.vstack(arrays).astype(np.float32)
    table = table[(table[:, 1] > 0) & (table[:, 2] > 0)]
    # table = np.vstack((table, table[:, [1, 0, 2]]))
    # table[table[:, 1] < table[:, 0]] = table[table[:, 1]
    #                                          < table[:, 0]][:, [1, 0, 2]]
    table[:, :2] = np.sort(table[:, :2], axis=1)
    """
    count each pair and the average contact_area
    """
    connections_unique, connections_indices, connections_count = np.unique(
        table[:, :2], axis=0, return_inverse=True, return_counts=True
    )
    contact_area_average = (
        np.bincount(connections_indices, weights=table[:, 2]) / connections_count
    )

    table = np.column_stack((connections_unique, contact_area_average))
    table = table[np.lexsort((table[:, 1], table[:, 0]))]

    table = pd.DataFrame(table, columns=["pore_1", "pore_2", "contact_area"])
    table = table.astype({"pore_1": "int32", "pore_2": "int32"})
    table.to_csv(csv_dir)
    tend = time.time()
    print("running time of calculation interface：%.6fs" % (tend - t0))

    """
    'solid_center':Path_dict['solid_network'] /f'{stem_input}_solid_center.csv',
    'pore_center': Path_dict['solid_network'] /f'{stem_input}_pore_center.csv',
    'dual_volume':Path_dict['solid_network'] /f'{stem_input}_dual_network_volume.csv',
    'dual_interface':Path_dict['solid_network'] /f'{stem_input}_dual_network_interface.csv'
    """


def summary_csvs(Path_dict, resolution):
    Csvs_dict = Path_dict["Csvs"]

    pore_center = pd.read_csv(Csvs_dict["pore_center"], index_col=0).to_numpy()
    solid_center = pd.read_csv(Csvs_dict["solid_center"], index_col=0).to_numpy()
    pore_center[:, 1:] *= resolution
    solid_center[:, 1:] *= resolution
    volume = (
        pd.read_csv(Csvs_dict["dual_volume"], index_col=0).to_numpy().astype(np.float64)
    )
    volume[:, 1] *= resolution**3
    volume[:, 2] *= resolution**2
    summary = np.concatenate(
        (pore_center, solid_center),
    )
    summary = np.concatenate((summary, volume[:, 1:]), axis=1)
    num_pore = len(pore_center)
    num_solid = len(solid_center)
    num_all = num_pore + num_solid
    void = np.zeros((num_all, 1), dtype=bool)
    void[:num_pore, 0] = True
    summary = np.concatenate((summary, void), axis=1)
    pd.DataFrame(
        summary, columns=["index", "x", "y", "z", "radius", "volume", "area", "void"]
    ).to_csv(Csvs_dict["summary"])
    if Csvs_dict.get("boundaries_areas") is not None:
        boundaries_areas = (
            pd.read_csv(Csvs_dict["boundaries_areas"], index_col=0)
            .loc[:, ["x-", "x+", "y-", "y+", "z-", "z+"]]
            .to_numpy()
            * resolution**2
        )
        boundaries_array = np.concatenate(
            (boundaries_areas, boundaries_areas.astype(bool)), axis=1
        )
        summary = np.concatenate((summary, boundaries_array), axis=1)
        pd.DataFrame(
            summary,
            columns=[
                "index",
                "x",
                "y",
                "z",
                "radius",
                "volume",
                "area",
                "void",
                "x-_surface_area",
                "x+_surface_area",
                "y-_surface_area",
                "y+_surface_area",
                "z-_surface_area",
                "z+_surface_area",
                "x-_surface",
                "x+_surface",
                "y-_surface",
                "y+_surface",
                "z-_surface",
                "z+_surface",
            ],
        ).to_csv(Csvs_dict["summary"])


def dualn_extraction(Path_dict, resolution, use_pne=False):
    Images_dict = Path_dict["Images"]
    Csvs_dict = Path_dict["Csvs"]
    Vtps_dict = Path_dict["Vtps"]
    Txts_dict = Path_dict["Txts"]

    summary = pd.read_csv(Csvs_dict["summary"], index_col=0)
    interface = pd.read_csv(
        Csvs_dict["dual_interface"], index_col=0
    ).to_numpy() - np.array([[1, 1, 0]])
    interface[:, 2] *= resolution**2
    num_all = len(summary)
    num_pore = np.count_nonzero(summary.loc[:, "void"])
    num_solid = num_all - num_pore
    pore_coords = summary.loc[:, ["x", "y", "z"]].to_numpy()
    pore_radius = summary.loc[:, "radius"].to_numpy()
    pore_volume = summary.loc[:, "volume"].to_numpy()
    pore_surface = summary.loc[:, "area"].to_numpy()
    pore_void = summary.loc[:, "void"].to_numpy().astype(bool)
    throat_conns = interface[:, 0:2].astype(np.int32)
    throat_area = interface[:, 2]
    throat_radius = (interface[:, 2] / np.pi) ** 0.5
    throat_length = np.linalg.norm(
        pore_coords[throat_conns[:, 0]] - pore_coords[throat_conns[:, 1]], axis=1
    )
    slice_pore = slice(None, num_pore)
    slice_solid = slice(num_pore, None)
    slice_all = slice(None, None)
    net_pore = net.read_network(
        path=Path_dict["pore_network"], name=f"{Images_dict['pore'].stem}"
    )
    if use_pne:
        pass
    else:
        net_pore["pore.coords"] = pore_coords[slice_pore]
        net_pore["pore.radius"] = pore_radius[slice_pore]
        net_pore["pore.volume"] = pore_volume[slice_pore]
        net_pore["pore.surface"] = pore_surface[slice_pore]
        net_pore["pore.void"] = pore_void[slice_pore]
        net_pore["pore.solid"] = np.zeros(num_pore, dtype=bool)
        throat_conns_pore = net_pore["throat.conns"]
        throat_conns_map = util.find_throat_conns_map(throat_conns_pore, throat_conns)
        throat_conns_pore_index = throat_conns_map[:, 0]
        throat_conns_index = throat_conns_map[:, 1]
        net_pore["pore._id"] = np.arange(len(net_pore["pore.all"]))
        net_pore["pore.label"] = net_pore["pore._id"].copy()
        net_pore["throat._id"] = np.arange(len(net_pore["throat.all"]))
        net_pore["throat.label"] = net_pore["throat._id"].copy()
        net_pore["throat.area"][throat_conns_pore_index] = throat_area[
            throat_conns_index
        ]
        # net_pore['throat.radius'][~throat_conns_pore_bool] = 0
        net_pore["throat.radius"][throat_conns_pore_index] = throat_radius[
            throat_conns_index
        ]
        net_pore["throat.length"][throat_conns_pore_index] = throat_length[
            throat_conns_index
        ]
        net_pore["throat.total_length"] = net_pore["throat.length"].copy()
        net_pore["throat.void"] = net_pore["throat.all"].copy()
        net_pore["throat.solid"] = ~net_pore["throat.all"].copy()
        missing_throats = np.setdiff1d(net_pore["throat._id"], throat_conns_pore_index)
        np.savetxt(Txts_dict["missing_throats"], missing_throats, fmt="%d")
        np.savetxt(
            Txts_dict["missing_throats"].with_name("throat_conns_pore.txt"),
            throat_conns_pore,
            fmt="%d",
        )
        np.savetxt(
            Txts_dict["missing_throats"].with_name("throat_conns.txt"),
            throat_conns,
            fmt="%d",
        )
        print("total throats num:", len(net_pore["throat._id"]))
        print("missing throats num:", len(missing_throats))
        print(
            f"missing throats percentage:{len(missing_throats) / len(net_pore['throat._id']) * 100:.1f}%"
        )

    """
    solid network
    """
    throat_solid = np.all(throat_conns >= num_pore, axis=1)
    throat_solid_num = np.count_nonzero(throat_solid)
    net_solid = {}
    net_solid["pore.all"] = np.ones(num_solid, dtype=bool)
    net_solid["pore._id"] = np.arange(num_solid)
    net_solid["pore.label"] = net_solid["pore._id"].copy()
    net_solid["pore.coords"] = pore_coords[slice_solid]
    net_solid["pore.radius"] = pore_radius[slice_solid]
    net_solid["pore.volume"] = pore_volume[slice_solid]
    net_solid["pore.surface"] = pore_surface[slice_solid]
    net_solid["pore.void"] = ~net_solid["pore.all"].copy()
    net_solid["pore.solid"] = net_solid["pore.all"].copy()
    net_solid["pore.shape_factor"] = np.ones(num_solid, dtype=np.float32)
    net_solid["throat.all"] = np.ones(throat_solid_num, dtype=bool)
    net_solid["throat._id"] = np.arange(throat_solid_num)
    net_solid["throat.label"] = net_solid["throat._id"].copy()
    net_solid["throat.conns"] = throat_conns[throat_solid] - num_pore
    net_solid["throat.area"] = throat_area[throat_solid]
    net_solid["throat.radius"] = throat_radius[throat_solid]
    net_solid["throat.solid"] = net_solid["throat.all"].copy()
    net_solid["throat.void"] = ~net_solid["throat.all"].copy()
    net_solid["throat.length"] = np.linalg.norm(
        pore_coords[throat_conns[throat_solid][:, 0]]
        - pore_coords[throat_conns[throat_solid][:, 1]],
        axis=1,
    )
    net_solid["throat.total_length"] = net_solid["throat.length"].copy()
    net_solid["throat.shape_factor"] = np.ones(throat_solid_num, dtype=np.float32)

    """
    dual network
    """
    throat_connect = np.any(throat_conns >= num_pore, axis=1) & np.any(
        throat_conns < num_pore, axis=1
    )
    net_dual = {}
    net_dual["pore.all"] = np.ones(num_all, dtype=bool)
    net_dual["pore._id"] = np.arange(num_all)
    net_dual["pore.label"] = net_dual["pore._id"].copy()
    net_dual["pore.coords"] = pore_coords
    net_dual["pore.radius"] = pore_radius
    net_dual["pore.volume"] = pore_volume
    net_dual["pore.surface"] = pore_surface
    net_dual["pore.void"] = np.concatenate(
        (net_pore["pore.void"], net_solid["pore.void"])
    )
    net_dual["pore.solid"] = np.concatenate(
        (net_pore["pore.solid"], net_solid["pore.solid"])
    )
    net_dual["pore.shape_factor"] = np.concatenate(
        (net_pore["pore.shape_factor"], net_solid["pore.shape_factor"])
    )
    net_dual["throat.conns"] = np.concatenate(
        (
            net_pore["throat.conns"],
            net_solid["throat.conns"] + num_pore,
            throat_conns[throat_connect],
        )
    )
    net_dual["throat.all"] = np.ones(len(net_dual["throat.conns"]), dtype=bool)
    net_dual["throat._id"] = np.arange(len(net_dual["throat.all"]))
    net_dual["throat.label"] = net_dual["throat._id"].copy()
    net_dual["throat.area"] = np.concatenate(
        (net_pore["throat.area"], net_solid["throat.area"], throat_area[throat_connect])
    )
    net_dual["throat.solid"] = np.all(net_dual["throat.conns"] >= num_pore, axis=1)
    net_dual["throat.void"] = np.all(net_dual["throat.conns"] < num_pore, axis=1)
    net_dual["throat.connect"] = np.any(
        net_dual["throat.conns"] >= num_pore, axis=1
    ) & np.any(net_dual["throat.conns"] < num_pore, axis=1)
    net_dual["throat.length"] = np.linalg.norm(
        pore_coords[net_dual["throat.conns"][:, 0]]
        - pore_coords[net_dual["throat.conns"][:, 1]],
        axis=1,
    )
    net_dual["throat.total_length"] = net_dual["throat.length"].copy()
    net_dual["throat.radius"] = np.concatenate(
        (
            net_pore["throat.radius"],
            net_solid["throat.radius"],
            throat_radius[throat_connect],
        )
    )
    # print(np.all((net_dual['throat.solid']+net_dual['throat.void'] +
    #       net_dual['throat.connect']).astype(bool) == net_dual['throat.all']))
    net_dual["throat.shape_factor"] = np.concatenate(
        (
            net_pore["throat.shape_factor"],
            net_solid["throat.shape_factor"],
            np.ones(np.count_nonzero(throat_connect)),
        )
    )

    if Csvs_dict.get("boundaries_areas") is not None:

        def read_boundaries_areas(summary, net, slice_):
            for key in (
                "x-_surface_area",
                "x+_surface_area",
                "y-_surface_area",
                "y+_surface_area",
                "z-_surface_area",
                "z+_surface_area",
            ):
                net[f"pore.{key}"] = summary.loc[:, key].to_numpy()[slice_]

            for key in (
                "x-_surface",
                "x+_surface",
                "y-_surface",
                "y+_surface",
                "z-_surface",
                "z+_surface",
            ):
                net[f"pore.{key}"] = summary.loc[:, key].to_numpy()[slice_].astype(bool)

        read_boundaries_areas(summary, net_pore, slice_pore)
        read_boundaries_areas(summary, net_solid, slice_solid)
        read_boundaries_areas(summary, net_dual, slice_all)

    net.network2vtk(net_pore, Vtps_dict["pore_network"])
    net.network2vtk(net_solid, Vtps_dict["solid_network"])
    net.network2vtk(net_dual, Vtps_dict["dual_network"])


if __name__ == "__main__":
    path = "./"
    file_name = "sphere_stacking_500_500_2000"
    path_pore = path + "/" + file_name + "_pore_pore.raw"
    path_solid = path + "/" + file_name + "_solid_solid.raw"
    size = [2000, 500, 500]
    # mix_image, labels, solid_image, solid_labels, pore_image, pore_labels = Mix_image(
    #     path_solid, path_pore, size)
    # phase_props_table = get_phase_props_table(mix_image)
    # name = file_name
    # find_node_center('solid',name,path,phase_props_table,solid_image,solid_labels)
    # find_node_center('pore',name,path,phase_props_table,pore_image,pore_labels)
    # find_node_volume('dual',name,path,phase_props_table,mix_image,labels)
    # find_interface('dual', name, path, phase_props_table, mix_image, labels)
    Path_dict = {}
    Path_dict
