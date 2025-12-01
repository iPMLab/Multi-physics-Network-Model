from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_1_8_0 as PARAM
import h5py
import pyvista as pv

PARAM = PARAM()
u_inlets = PARAM.u_inlets
num_params = PARAM.num_comsol_params
Path_comsol = PARAM.Path_comsol
prefix = PARAM.prefix
shape = PARAM.raw_shape
Path_mesh_vtu = PARAM.Path_mesh_vtu
resolution = PARAM.resolution

heat_flux = 50000
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

# def write_h5_single_txt():
#     Path_txt = Path(f"{Path_comsol}/{prefix}_{heat_flux}W.txt")
#     data = np.loadtxt(Path_txt)
#     # 创建一个 HDF5 文件（可复用文件名或新建）
#     with h5py.File(f"{Path_comsol}/{prefix}.h5", "w") as hf:
#         for i, u_inlet in tqdm(enumerate(u_inlets)):
#             data_i = data[:, num_params * i : num_params * (i + 1)]

#             data_i = data_i.T.reshape(num_params, *shape)

#             # 在 HDF5 文件中为每个 u_inlet 创建独立数据集
#             dataset_name = f"_u{float(u_inlet):.5f}_hf{int(heat_flux)}"
#             hf.create_dataset(
#                 dataset_name,
#                 data=data_i,
#                 compression="gzip",  # 启用压缩
#             )


# def write_h5_multi_txt():
#     with h5py.File(f"{Path_comsol}/{prefix}.h5", "w") as hf:
#         for i, u_inlet in tqdm(enumerate(u_inlets)):
#             txt_name_i = f"{prefix}_u{float(u_inlet):.5f}_hf{heat_flux}W.txt"
#             Path_txt_i = Path(f"{Path_comsol}/{txt_name_i}")
#             data_i = np.loadtxt(Path_txt_i)
# data_i = data_i.T.reshape(num_params, *shape)
#             dataset_name = f"_u{float(u_inlet):.5f}_hf{int(heat_flux)}"
#             hf.create_dataset(
#                 dataset_name,
#                 data=data_i,
#                 compression="gzip",  # 启用压缩
#             )


def write_h5_data(multi=False):
    with h5py.File(PARAM.Path_data_h5, "a") as f:
        mesh_group = f["mesh"]
        cell2voxel = mesh_group["cell2voxel"][:]
        cells = mesh_group["cells"][:]
        condition_group = f.require_group("condition")
        if not multi:
            data = np.loadtxt(f"{Path_comsol}/{prefix}_{heat_flux}W.txt")
        for i, u_inlet in tqdm(enumerate(u_inlets)):
            if multi:
                txt_name_i = f"{prefix}_u{float(u_inlet):.5f}_hf{heat_flux}W.txt"
                Path_txt_i = Path(f"{Path_comsol}/{txt_name_i}")
                data_i = np.loadtxt(Path_txt_i)
            else:
                data_i = data[:, num_params * i : num_params * (i + 1)]

            group_name = f"_u{float(u_inlet):.5f}_hf{int(heat_flux)}"
            print(group_name)
            group = condition_group.require_group(group_name)
            for param_name, param_index in comsol_params_map.items():
                point_data_ = data_i[:, param_index]
                group.create_dataset(
                    name=f"point.{param_name}", data=point_data_, compression="gzip"
                )
                cell_data_ = np.mean(point_data_[cells], axis=1, keepdims=False)
                voxel_data_ = cell_data_[cell2voxel.reshape(shape)]
                group.create_dataset(
                    name=f"voxel.{param_name}", data=voxel_data_, compression="gzip"
                )


multi = bool(1)
write_h5_data(multi=multi)
