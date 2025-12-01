import mph
from pathlib import Path
import numpy as np
import pyvista as pv

import h5py
import pandas as pd
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_1_8_0 as PARAM

import sys

sys.path.append("../../")
from mpnm_new.util import unique_rows, find_throat_conns_map

PARAM = PARAM()
shape = PARAM.raw_shape
resolution = PARAM.resolution
Path_h5 = PARAM.Path_data_h5
Path_comsol = PARAM.Path_comsol
Path_binary_raw = PARAM.Path_binary_raw
client = mph.start()
model = client.load(
    r"D:\yjp\Workdir\Code\ZJU\Study\Python\heat_tranfer\3D_Study\COMSOL\big\3D_Finney_results\Finney_1_8_0_50000W_lowRe.mph"
)

hf = 10000


Path_T_s_vtu = Path_comsol / "T_s.vtu"
Path_F_s_vtu = Path_comsol / "F_s.vtu"
T_par = ["T", "ht.dfluxx", "ht.dfluxy", "ht.dfluxz"]
F_par = ["p", "u", "v", "w", "ht.cfluxx", "ht.cfluxy", "ht.cfluxz"]
T_par_map = {k: v for k, v in zip(T_par, range(len(T_par)))}
F_par_map = {k: v for k, v in zip(F_par, range(len(F_par)))}


model_java = model.java
export_tags = list(model_java.result().export().tags())
result_node = model_java.result()

# if "datatemp" in export_tags:
# T_node.set("fullprec", "off")

###### T sturcture export ######
result_node.export().remove("T_s")
result_node.export().create("T_s", "Data")
T_s_node = result_node.export("T_s")
T_s_node.set("exporttype", "vtu")
T_s_node.set("expr", ["T"])
T_s_node.set("innerinput", "manual")
T_s_node.set("solnum", "1")
T_s_node.set("smooth", "everywhere")  # none, material, internal, everywhere, expression
T_s_node.set("filename", str(Path_T_s_vtu.resolve()))
T_s_node.run()
###### F structure export ######
result_node.export().remove("F_s")
result_node.export().create("F_s", "Data")
F_s_node = result_node.export("F_s")
F_s_node.set("exporttype", "vtu")
F_s_node.set("expr", ["p"])
F_s_node.set("innerinput", "manual")
F_s_node.set("solnum", "1")
F_s_node.set("smooth", "material")
F_s_node.set("includenan", "off")
F_s_node.set("recover", "ppr")
F_s_node.set("filename", str(Path_F_s_vtu.resolve()))
F_s_node.run()


# u_inlets = np.asarray(
#     model_java.study("std1").feature("param").getStringArray("plistarr")).reshape(-1)
u_inlets = np.array([0.00006, 0.00056, 0.00562, 0.05617, 0.28087])
# u_inlets = u_inlets[:1]

for i, u_inlet in enumerate(u_inlets):
    ##### T export ######
    result_node.export().remove("T")
    result_node.export().create("T", "Data")
    T_node = result_node.export("T")
    T_node.set("exporttype", "text")
    T_node.set("expr", T_par)
    T_node.set("innerinput", "manual")
    T_node.set("solnum", [str(i + 1)])
    T_node.set("header", "off")
    T_node.set("includecoords", "off")
    T_node.set(
        "smooth", "everywhere"
    )  # none, material, internal, everywhere, expression
    T_node.set("recover", "ppr")
    T_node.set("separator", ",")
    # T_node.set("threshold", "manual")
    # T_node.set("thresholdvalue", "0.01")
    # T_node.set("sort", "on")
    T_node.set(
        "filename", str((Path_comsol / f"T_u{u_inlet:.5f}_hf{hf}.txt").resolve())
    )
    T_node.run()

    ##### F export ######
    result_node.export().remove("F")
    result_node.export().create("F", "Data")
    F_node = result_node.export("F")
    F_node.set("exporttype", "text")
    F_node.set("expr", F_par)
    F_node.set("innerinput", "manual")
    F_node.set("solnum", [str(i + 1)])
    F_node.set("header", "off")
    F_node.set("includecoords", "off")
    F_node.set("includenan", "off")
    F_node.set("smooth", "material")  # none, material, internal, everywhere, expression
    F_node.set("recover", "ppr")
    # F_node.set("threshold", "manual")
    # F_node.set("thresholdvalue", "0.05")
    F_node.set("separator", ",")
    F_node.set(
        "filename", str((Path_comsol / f"F_u{u_inlet:.5f}_hf{hf}.txt").resolve())
    )
    F_node.run()

# client.remove(model)
# del client

mesh_T_s = pv.read(Path_T_s_vtu, progress_bar=True)
cells_T_s = mesh_T_s.cells
points_T_s = mesh_T_s.points
# cells_T_s = cells_T_s.reshape(-1, 5)[:, 1:]
cells_T_s = cells_T_s.reshape(-1, 5)[:, 1:]
mesh_F_s = pv.read(Path_F_s_vtu, progress_bar=True)
cells_F_s = mesh_F_s.cells
points_F_s = mesh_F_s.points
# cells_F_s = cells_F_s.reshape(-1, 5)[:, 1:]

points_map = find_throat_conns_map(points_T_s, points_F_s)
assert np.all(
    points_map[:, 0] == points_map[:, 1]
), " np.all(points_map[:, 0] == points_map[:, 1]) is False"

num_points = points_T_s.shape[0]


Path_h5.parent.mkdir(exist_ok=True, parents=True)
with h5py.File(Path_h5, "a") as f:
    f.compression = "gzip"

    group_mesh = f.require_group("mesh")
    group_voxel = f.require_group("voxel")

    ####### mesh data #######
    if "points" not in group_mesh:
        group_mesh.create_dataset("points", data=points_T_s)
    if "cells" not in group_mesh:
        group_mesh.create_dataset("cells", data=cells_T_s)
    if "cell.center" not in group_mesh:
        group_mesh.create_dataset("cell.center", data=mesh_T_s.cell_centers().points)
    for i, u_inlet in enumerate(u_inlets):
        group_mesh_i = group_mesh.require_group(f"_u{u_inlet:.5f}_hf{hf}")
        point_data_T = pd.read_csv(
            Path_comsol / f"T_u{u_inlet:.5f}_hf{hf}.txt",
            sep=",",
            header=None,
            engine="pyarrow",
        ).to_numpy(dtype=np.float32)
        point_data_F = pd.read_csv(
            Path_comsol / f"F_u{u_inlet:.5f}_hf{hf}.txt",
            sep=",",
            header=None,
            engine="pyarrow",
        ).to_numpy(dtype=np.float32)
        for k, v in T_par_map.items():
            k_ = f"point.{k}"
            if k_ in group_mesh_i:
                del group_mesh_i[k_]
            group_mesh_i.create_dataset(k_, data=point_data_T[:, v])

        point_data_F_template = np.full(num_points, np.nan, dtype=np.float32)
        for k, v in F_par_map.items():
            k_ = f"point.{k}"
            if k_ in group_mesh_i:
                del group_mesh_i[k_]
            point_data_F_template_i = point_data_F_template.copy()
            point_data_F_template_i[points_map[:, 0]] = point_data_F[:, v]
            group_mesh_i.create_dataset(k_, data=point_data_F_template_i)
        if "point.U" in group_mesh_i:
            del group_mesh_i["point.U"]
        point_data_F_template_i = point_data_F_template.copy()
        point_data_F_template_i[points_map[:, 0]] = np.sqrt(
            point_data_F[:, F_par_map["u"]] ** 2
            + point_data_F[:, F_par_map["v"]] ** 2
            + point_data_F[:, F_par_map["w"]] ** 2
        )
        group_mesh_i.create_dataset("point.U", data=point_data_F_template_i)
    if "cell.solid" not in group_mesh:
        cell_solid = group_mesh[f"_u{u_inlet:.5f}_hf{hf}"]["point.p"][:][cells_T_s]
        cell_solid = np.any(np.isnan(cell_solid), axis=1)
        group_mesh.create_dataset("cell.solid", data=cell_solid)

    ######## voxel data #######
    if "voxel.cell" not in group_voxel:
        z_img, y_img, x_img = np.indices(shape)
        z_img, y_img, x_img = z_img.ravel(), y_img.ravel(), x_img.ravel()
        coords_img = np.column_stack((x_img, y_img, z_img)).astype(np.float32)
        coords_img += 0.5
        coords_img *= resolution
        voxel_cell = mesh_T_s.find_containing_cell(coords_img).reshape(shape)
    else:
        voxel_cell = group_voxel["voxel.cell"][:]
    assert np.all(voxel_cell != -1), "some points are outside the mesh"
    voxel_keys = ["p", "T", "U"]
    if "voxel.cell" not in group_voxel:
        group_voxel.create_dataset("voxel.cell", data=voxel_cell)
    for i, u_inlet in enumerate(u_inlets):
        group_voxel_i = group_voxel.require_group(f"_u{u_inlet:.5f}_hf{hf}")
        for k in voxel_keys:
            k_ = f"voxel.{k}"
            if k_ in group_voxel_i:
                del group_voxel_i[k_]
            group_voxel_i.create_dataset(
                k_,
                data=group_mesh[f"_u{u_inlet:.5f}_hf{hf}"][f"point.{k}"][:][
                    cells_T_s
                ].mean(axis=1)[voxel_cell],
            )
    if "voxel.solid" not in group_voxel:
        group_voxel.create_dataset(
            "voxel.solid",
            data=group_mesh["cell.solid"][:][voxel_cell],
        )
        Path_binary_raw.parent.mkdir(exist_ok=True, parents=True)
        group_mesh["cell.solid"][:][voxel_cell].astype(np.uint8).tofile(Path_binary_raw)
print("done")

# mesh = pv.UnstructuredGrid(
#     np.insert(cells_T_s, 0, 4, axis=1).reshape(-1),
#     np.full(cells_T_s.shape[0], pv.CellType.TETRA),
#     points_T_s,
# )

# mesh.point_data["point.cfluxx"] = point_data_T[:, T_par_map["ht.dfluxx"]]
# mesh.plot()
