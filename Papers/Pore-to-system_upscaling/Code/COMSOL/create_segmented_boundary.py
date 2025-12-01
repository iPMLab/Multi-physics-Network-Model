import sys

sys.path.append("../../")
from mpnm_new import extraction, network as net
import pyvista as pv
import numpy as np
from pathlib import Path
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_1_8_0 as PARAM
from scipy.spatial import cKDTree

PARAM = PARAM()
Path_binary_raw = PARAM.Path_binary_raw
raw_shape = PARAM.raw_shape
resolution = PARAM.resolution
Path_mix_raw = PARAM.Path_mix_raw
Path_net_dual = PARAM.Path_net_dual
Path_net_pore = PARAM.Path_net_pore
Path_net_solid = PARAM.Path_net_solid

Path_mesh = Path(r"C:\Users\yjp\Desktop\test\mesh_o.vtu")
mesh = pv.read(Path_mesh, progress_bar=True)
cells = mesh.cells.reshape(-1, mesh.cells[0] + 1)[:, 1:]
point_p = mesh.point_data["Pressure"]
cell_center = mesh.cell_centers().points
cell_p = point_p[cells]
cell_solid = np.any(np.isnan(cell_p), axis=1)
mesh.cell_data["solid"] = cell_solid
# mesh.save(Path_mesh)
z_coord, y_coord, x_coord = np.indices(raw_shape)
z_coord, y_coord, x_coord = z_coord.ravel(), y_coord.ravel(), x_coord.ravel()
z_coord = (z_coord + 0.5) * resolution
y_coord = (y_coord + 0.5) * resolution
x_coord = (x_coord + 0.5) * resolution

coords = np.vstack((x_coord, y_coord, z_coord)).T
voxel_cell = mesh.find_containing_cell(coords)
assert np.all(voxel_cell != -1), "Some points are outside the mesh"
image_binary = mesh.cell_data["solid"][voxel_cell.reshape(raw_shape)].astype(np.uint8)
image_binary.tofile("image_binary.raw")

# image = np.fromfile(
#     Path_binary_raw,
#     dtype=np.uint8,
# ).reshape(raw_shape)


config_0 = {
    "method": "pne",
    "target_value": 0,
    "resolution": resolution,
    "mode": "image",
}
# config_0 = {"method": "snow", "target_value": 0,"r_max":5}
# config_1 = {"method": "snow", "target_value": 1}
config_1 = {
    "method": "pne",
    "target_value": 1,
    "resolution": resolution,
    "mode": "image",
}

config_list = [config_0, config_1]


image_mix, nets, seps = extraction.dualn_phase_extraction(
    fill_unlabeled=False,
    image=image_binary,
    resolution=resolution,
    config_list=config_list,
    n_workers_segmentation=len(config_list),
    n_workers_extraction=32,
    backend="loky",
)
Path_net_dual.parent.mkdir(exist_ok=True, parents=True)
Path_mix_raw.parent.mkdir(exist_ok=True, parents=True)

image_mix.tofile(Path_mix_raw)
dualn, pn, sn = nets
net.network2vtk(dualn, Path_net_dual)
net.network2vtk(pn, Path_net_pore)
net.network2vtk(sn, Path_net_solid)
print("pore_num_dual:", dualn["pore.all"].size)
print("pore_num_pn:", pn["pore.all"].size)
print("pore_num_sn:", sn["pore.all"].size)
max_seg = np.max(image_mix)
unseg_void_z, unseg_void_y, unseg_void_x = np.where(
    (image_binary == 0) & (image_mix == 0)
)
unseg_solid_z, unseg_solid_y, unseg_solid_x = np.where(
    (image_binary == 1) & (image_mix == 0)
)
image_mix[unseg_void_z, unseg_void_y, unseg_void_x] = max_seg + 1
image_mix[unseg_solid_z, unseg_solid_y, unseg_solid_x] = max_seg + 2

image_mix_void_bool = image_binary == 0
image_mix_solid_bool = ~image_mix_void_bool
z_coords_void, y_coords_void, x_coords_void = np.where(image_mix_void_bool)
voxel_ravel_void = image_mix[image_mix_void_bool].ravel()
coords_void = np.column_stack((x_coords_void, y_coords_void, z_coords_void)).astype(
    np.float32
)
coords_void += 0.5
coords_void *= resolution

z_coords_solid, y_coords_solid, x_coords_solid = np.where(image_mix_solid_bool)
voxel_ravel_solid = image_mix[image_mix_solid_bool].ravel()
coords_solid = np.column_stack((x_coords_solid, y_coords_solid, z_coords_solid)).astype(
    np.float32
)
coords_solid += 0.5
coords_solid *= resolution


Tree_void = cKDTree(coords_void)
Tree_solid = cKDTree(coords_solid)

cell_center_void = cell_center[~cell_solid]
cell_center_solid = cell_center[cell_solid]

dist_void, ind_void = Tree_void.query(cell_center_void)
dist_solid, ind_solid = Tree_solid.query(cell_center_solid)

cell_voxel = np.empty(cells.shape[0], dtype=np.int32)
cell_voxel[~cell_solid] = voxel_ravel_void[ind_void]
cell_voxel[cell_solid] = voxel_ravel_solid[ind_solid]


mesh.cell_data["Material"] = cell_voxel.astype(np.int32)

mesh.plot(scalars="Material", cmap="tab10", show_edges=True)
# with open("output.bdf", "w") as f:
#     # 写入文件头
#     f.write("$ NASTRAN BLD File Created from PolyData\n")
#     f.write("BEGIN BULK\n")

#     # 写入节点(GRID)
#     points = mesh.points
#     for i, (x, y, z) in enumerate(points, 1):
#         f.write(f"GRID,{i},,{x:.6f},{y:.6f},{z:.6f}\n")

#     # 写入单元
#     if mesh.n_cells > 0:
#         cells = mesh.cells.reshape(-1, mesh.cells[0] + 1)
#         for i, cell in enumerate(cells[1:], 1):
#             cell_type = cell[0]
#             nodes = cell[1:]
#             f.write(
#                 f"CTETRA,{i},{mesh.cell_data['Material'][i-1]},{nodes[0] + 1},{nodes[1] + 1},{nodes[2] + 1},{nodes[3] + 1}\n"
#             )

#     # 写入材料属性
#     # f.write("MAT1,1,1.0E7,,0.3,2700.\n")  # 示例材料
#     # f.write("PSHELL,1,1,0.1\n")  # 示例属性

#     f.write("ENDDATA\n")
points = mesh.points

cells = mesh.cells.reshape(-1, mesh.cells[0] + 1)[:, 1:5]
node_ids = cells + 1  # 所有节点的索引+1

with open(r"C:\Users\yjp\Desktop\test\output.bdf", "w") as f:
    f.write("$ NASTRAN BLD File Created from PolyData\n")
    f.write("BEGIN BULK\n")
    # GRID 行
    np.savetxt(
        f,
        np.column_stack([np.arange(1, len(points) + 1), points]),
        fmt="GRID,%d,,%.6f,%.6f,%.6f",
        delimiter="",
    )
    # CTETRA 行
    np.savetxt(
        f,
        np.column_stack(
            [np.arange(1, len(cells) + 1), mesh.cell_data["Material"], node_ids]
        ),
        fmt="CTETRA,%d,%d,%d,%d,%d,%d",
        delimiter="",
    )
    f.write("ENDDATA\n")
