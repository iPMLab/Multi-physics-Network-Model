import numpy as np
import numba as nb
import pyvista as pv
import sys
import h5py

sys.path.append("../../")
from mpnm_new.util import unique_rows
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_1_8_0 as PARAM
from pathlib import Path
from scipy.spatial import cKDTree

PARAM = PARAM()
u_inlets = PARAM.u_inlets
num_params = PARAM.num_comsol_params
Path_comsol = PARAM.Path_comsol
prefix = PARAM.prefix
shape = PARAM.raw_shape
Path_mesh_vtu = PARAM.Path_mesh_vtu
resolution = PARAM.resolution

Path_mesh_vtu = Path(
    r"D:\yjp\Workdir\Code\ZJU\Study\Python\multi-physic-network-model\sample_data_1_8_0\Finney\comsol_data\mesh_all.vtu"
)
mesh = pv.read(Path_mesh_vtu, progress_bar=True)
points_unique, index, inverse, counts = unique_rows(
    mesh.points,
    return_index=True,
    return_inverse=True,
    return_counts=True,
    keepdims=False,
)

# inverse = np.sort(inverse,axis=0)
# mesh.save(Path_mesh_vtu, binary=True)
# mesh = mesh.clean()
z_img, y_img, x_img = np.indices(PARAM.raw_shape)
z_img, y_img, x_img = z_img.ravel(), y_img.ravel(), x_img.ravel()
coords_img = np.column_stack((x_img, y_img, z_img)).astype(np.float32)
coords_img += 0.5
coords_img *= resolution
cell2voxel = mesh.find_containing_cell(coords_img)
if np.any(cell2voxel == -1):
    print("some points are outside the mesh")
    not_in_mesh = np.where(cell2voxel == -1)[0]

    Tree = cKDTree(mesh.cell_data["center"])
    closest_cell_dist, closest_cell = Tree.query(coords_img[not_in_mesh], k=1)
    cell2voxel[not_in_mesh] = closest_cell

cell2voxel = cell2voxel.reshape(shape)
cells = mesh.cells.reshape(-1, 5)[:, 1:]
P = mesh.point_data["P"]
cell_solid = np.any(np.isnan(P[cells]), axis=1)
binary_raw = cell_solid[cell2voxel].astype(np.uint8)

binary_raw.tofile(PARAM.Path_binary_raw)
cell_center = mesh.cell_centers().points
###################
# Overide h5 data #
###################

with h5py.File(PARAM.Path_data_h5, "w") as f:
    group = f.require_group("mesh")
    group.create_dataset("cells", data=cells, compression="gzip")
    group.create_dataset("points", data=mesh.points, compression="gzip")
    group.create_dataset("cell.solid", data=cell_solid, compression="gzip")
    group.create_dataset("cell.center", data=cell_center, compression="gzip")
    group.create_dataset("cell2voxel", data=cell2voxel, compression="gzip")
