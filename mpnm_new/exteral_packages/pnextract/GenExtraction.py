#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 01:11:41 2024

@author: htmt
"""

from multiprocessing import freeze_support
import sys
from pathlib import Path
import numpy as np


def get_input_params(**kwargs):
    # 定义参数默认值字典
    defaults = {
        "path_input": "./1.raw",
        "size": "100,100,100",
        "resolution": "1e-5",
        # "sigma": "0.4",
        # "r_max": "5.",
        "if_calculate_boundary_area": "1",
        "parallel": "1",
        "n_workers": "10",
        "n_workers_dt": "1",
    }
    help_params = {
        "path_input": "Path of input file",
        "size": "Size of image, format: x,y,z",
        "resolution": "Resolution of image, unit: m",
        # "sigma": "Sigma /of Gaussian filter",
        # "r_max": "The radius of the spherical element used in the maximum filter in solid network extraction, unit: m",
        "if_calculate_boundary_area": "Whether to calculate boundary area, 1 for yes, 0 for no",
        "parallel": "Whether to use parallel processing, 1 for yes, 0 for no",
        "n_workers": "Number of workers for parallel postprocessing",
        "n_workers_dt": "Number of workers for parallel processing of distance map",
    }

    # 动态合并输入值与默认值
    params = {
        key: kwargs.get(key)
        or input(f"{key}, Default is {defaults[key]}, {help_params[key]}:\n")
        or defaults[key]
        for key in defaults
    }

    # 类型转换处理
    params["path_input"] = Path(
        params["path_input"].strip("'\"")
        if isinstance(params["path_input"], str)
        else params["path_input"]
    )
    params["size"] = np.array(
        [int(i) for i in params["size"].split(",")], dtype=np.int64
    )
    params["resolution"] = float(eval(params["resolution"]))
    params["sigma"] = None
    params["r_max"] = None
    params["if_calculate_boundary_area"] = bool(
        int(params["if_calculate_boundary_area"])
    )
    params["parallel"] = bool(int(params["parallel"]))
    params["n_workers"] = int(params["n_workers"])
    params["n_workers_dt"] = int(params["n_workers_dt"])
    return params


def gen_extraction(input_params=None):
    from multiprocessing import Pool
    import warnings
    import sys
    import os
    from Pnm import pore_network_extraction as PNE
    from Snm import solid_network_extraction as SNE
    from ExtractionDNM import (
        Mix_image,
        find_node_volume,
        find_interface,
        find_node_center,
        get_phase_props_table,
        summary_csvs,
        dualn_extraction,
    )
    from Calculate_boundary_area import calculate_boundary_area

    # from merge_solid import merge_solid

    import platform

    pf = platform.system()
    from joblib import Parallel, delayed

    def source_path(relative_path):
        if getattr(sys, "frozen", False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    cd = source_path("")
    os.chdir(cd)

    warnings.filterwarnings("ignore", category=UserWarning)
    if input_params is None:
        input_params = {}
    params = get_input_params(**input_params)

    print(params)
    Path_input = params["path_input"]
    size = params["size"]
    resolution = params["resolution"]
    sigma = params["sigma"]
    r_max = params["r_max"]
    if_calculate_boundary_area = params["if_calculate_boundary_area"]
    parallel = params["parallel"]
    n_workers = params["n_workers"]
    n_workers_dt = params["n_workers_dt"]

    # Root paths
    Path_dict = {
        "Path_input": Path_input,
        "name_input": Path_input.name,
        "stem_input": Path_input.stem,
        "parent_input": Path_input.parent,
    }

    # Sub paths
    Path_dict["pore_network"] = Path_dict["parent_input"] / "pore_network"
    Path_dict["solid_network"] = Path_dict["parent_input"] / "solid_network"
    Path_dict["images"] = Path_dict["parent_input"] / "images"
    Path_dict["vtps"] = Path_dict["parent_input"] / "vtps"
    Path_dict["txts"] = Path_dict["parent_input"] / "txts"

    # images
    stem_input = Path_dict["stem_input"]
    Path_images = Path_dict["images"]
    Path_dict["Images"] = {
        "pore": Path_images / f"{stem_input}_pore.raw",
        # 'pore_pore': Path_image/f'{stem_input}_pore_pore.raw',
        "pore_VElems": Path_images / f"{stem_input}_pore_VElems.raw",
        "pore_VElems_origin": Path_images
        / f"{stem_input}_pore_VElems_origin_{size[0] + 2}_{size[1] + 2}_{size[2] + 2}.raw",
        # 'solid': Path_image/f'{stem_input}_solid.raw',
        # 'solid_solid': Path_image / f'{stem_input}_solid_solid.raw',
        "solid_VElems": Path_images / f"{stem_input}_solid_VElems.raw",
        "mix": Path_images / f"{stem_input}_mix.raw",
    }

    Path_dict["Csvs"] = {
        "solid_center": Path_dict["solid_network"] / f"{stem_input}_solid_center.csv",
        "pore_center": Path_dict["solid_network"] / f"{stem_input}_pore_center.csv",
        "dual_volume": Path_dict["solid_network"]
        / f"{stem_input}_dual_network_volume.csv",
        "dual_interface": Path_dict["solid_network"]
        / f"{stem_input}_dual_network_interface.csv",
        "summary": Path_dict["solid_network"] / f"{stem_input}_summary.csv",
    }
    Path_dict["Vtps"] = {
        "pore_network": Path_dict["vtps"] / f"{stem_input}_pore_network.vtp",
        "solid_network": Path_dict["vtps"] / f"{stem_input}_solid_network.vtp",
        "dual_network": Path_dict["vtps"] / f"{stem_input}_dual_network.vtp",
    }
    Path_dict["Txts"] = {
        "missing_throats": Path_dict["txts"] / f"{stem_input}_missing_throats.txt",
    }
    Images_dict = Path_dict["Images"]
    Csvs_dict = Path_dict["Csvs"]
    Path_dict["pnextract"] = {}
    if pf == "Windows":
        Path_dict["pnextract"]["origin"] = Path("./res/pnextract.exe")
        backend = "loky"
    else:
        backend = "multiprocessing"
        Path_dict["pnextract"]["origin"] = Path("./res/pnextract")
    Path_dict["pnextract"]["target"] = (
        Path_dict["pore_network"] / Path_dict["pnextract"]["origin"].name
    )
    Path_dict["mhd"] = {}
    Path_dict["mhd"]["origin"] = Path("./res/Sample.mhd")
    Path_dict["mhd"]["target"] = (
        Path_dict["pore_network"].resolve() / f"{Images_dict['pore'].stem}.mhd"
    )
    if if_calculate_boundary_area:
        Path_dict["Csvs"]["boundaries_areas"] = (
            Path_dict["solid_network"] / "Boundaries_areas.csv"
        )
    Path_dict["pore_network"].mkdir(parents=True, exist_ok=True)
    Path_dict["solid_network"].mkdir(parents=True, exist_ok=True)
    Path_dict["images"].mkdir(parents=True, exist_ok=True)
    Path_dict["vtps"].mkdir(parents=True, exist_ok=True)
    Path_dict["txts"].mkdir(parents=True, exist_ok=True)

    PNE_args = (Path_dict, resolution, size)
    SNE_args = (Path_dict, 1, resolution, size, sigma, r_max, n_workers_dt)

    funcs = [PNE, SNE]
    args = [PNE_args, SNE_args]
    res = Parallel(n_jobs=2 if parallel else 1, backend=backend)(
        delayed(func)(*arg) for func, arg in zip(funcs, args)
    )
    res_PNE, res_SNE = res[0], res[1]
    # if parallel:
    #     pool = Pool(processes=2)
    #     process_PNE = pool.apply_async(PNE, args=PNE_args)
    #     process_SNE = pool.apply_async(SNE, args=SNE_args)
    #     pool.close()
    #     pool.join()
    #     res_PNE = process_PNE.get()
    #     res_SNE = process_SNE.get()
    # else:
    #     res_PNE = PNE(*PNE_args)
    #     res_SNE = SNE(*SNE_args)

    mix_image, labels, solid_image, solid_labels, pore_image, pore_labels = Mix_image(
        Path_dict, res_PNE, res_SNE
    )
    phase_props_table = get_phase_props_table(mix_image)
    find_node_center(
        Csvs_dict["solid_center"],
        phase_props_table,
        solid_image,
        solid_labels,
        resolution=resolution,
        n_workers=n_workers,
        n_workers_dt=n_workers_dt,
        backend=backend,
    )
    find_node_center(
        Csvs_dict["pore_center"],
        phase_props_table,
        pore_image,
        pore_labels,
        resolution=resolution,
        n_workers=n_workers,
        backend=backend,
    )
    find_node_volume(
        Csvs_dict["dual_volume"],
        phase_props_table,
        mix_image,
        labels,
        resolution=resolution,
        n_workers=n_workers,
        backend=backend,
    )
    find_interface(
        Csvs_dict["dual_interface"],
        phase_props_table,
        mix_image,
        labels,
        resolution=resolution,
        n_workers=n_workers,
        backend=backend,
    )

    if Csvs_dict.get("boundaries_areas") is not None:
        calculate_boundary_area(mix_image, Path_dict)
    summary_csvs(Path_dict, resolution)
    dualn_extraction(Path_dict, resolution)


if __name__ == "__main__":
    freeze_support()
    gen_extraction()
# try:
#     if __name__ == '__main__':
#         freeze_support()
#         main()
# except:
#     pass
