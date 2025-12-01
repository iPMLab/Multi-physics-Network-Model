import sys

sys.path.append("../../")
import h5py
from mpnm_new import extraction, network as net, util
import numpy as np
import numba as nb
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_1_8_0 as PARAM
from skimage.measure import regionprops, regionprops_table
from scipy.spatial import cKDTree

PARAM = PARAM()
Path_binary_raw = PARAM.Path_binary_raw
raw_shape = PARAM.raw_shape
resolution = PARAM.resolution
Path_mix_raw = PARAM.Path_mix_raw
Path_net_dual = PARAM.Path_net_dual
Path_net_pore = PARAM.Path_net_pore
Path_net_solid = PARAM.Path_net_solid
image = np.fromfile(
    Path_binary_raw,
    dtype=np.uint8,
).reshape(raw_shape)


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


# res = extraction.dualn_phase_extraction(
#     fill_unlabeled=False,
#     image=image,
#     resolution=resolution,
#     config_list=config_list,
#     n_workers_segmentation=len(config_list),
#     n_workers_extraction=32,
#     backend="loky",
# )


@nb.njit(fastmath=True)
def nb_find_bbox(labeled_image, min_value=1):
    shape_max = max(labeled_image.shape)
    labeled_image_max = labeled_image.max()
    zs_min = np.full(labeled_image_max + 1, shape_max + 1)
    ys_min = zs_min.copy()
    xs_min = zs_min.copy()
    zs_max = np.full(labeled_image_max + 1, -1)
    ys_max = zs_max.copy()
    xs_max = zs_max.copy()

    for z in range(labeled_image.shape[0]):
        for y in range(labeled_image.shape[1]):
            for x in range(labeled_image.shape[2]):
                value = labeled_image[z, y, x]
                if value >= min_value:
                    zs_min[value] = min(zs_min[value], z)
                    ys_min[value] = min(ys_min[value], y)
                    xs_min[value] = min(xs_min[value], x)
                    zs_max[value] = max(zs_max[value], z)
                    ys_max[value] = max(ys_max[value], y)
                    xs_max[value] = max(xs_max[value], x)

    labels = np.where(zs_max != -1)[0]
    zs_min = zs_min[labels]
    ys_min = ys_min[labels]
    xs_min = xs_min[labels]
    zs_max = zs_max[labels] + 1
    ys_max = ys_max[labels] + 1
    xs_max = xs_max[labels] + 1
    return labels, zs_min, ys_min, xs_min, zs_max, ys_max, xs_max


@nb.njit(fastmath=True)
def nb_relabel(
    cell2voxel,
    image_labeled,
    labels,
    zs_min,
    ys_min,
    xs_min,
    zs_max,
    ys_max,
    xs_max,
):
    for i in range(labels.size):
        z_min, y_min, x_min = zs_min[i], ys_min[i], xs_min[i]
        z_max, y_max, x_max = zs_max[i], ys_max[i], xs_max[i]
        label = labels[i]
        region_small = cell2voxel[z_min:z_max, y_min:y_max, x_min:x_max].flatten()
        region_small_bool = region_small == label
        region_big = image_labeled[z_min:z_max, y_min:y_max, x_min:x_max].flatten()
        region_big_label = np.where(region_small_bool, region_big, 0)
        labels_big_counts = np.bincount(region_big_label)
        if labels_big_counts.size > 1:
            labels_big_counts = labels_big_counts[1:]
            label_big = np.argmax(labels_big_counts) + 1
            label_big = np.max(region_big_label)
            region_big[region_small_bool] = label_big
            region_big = region_big.reshape(z_max - z_min, y_max - y_min, x_max - x_min)
            image_labeled[z_min:z_max, y_min:y_max, x_min:x_max] = region_big

    return image_labeled


### a cell should belong to only one label ###
def relabel_segments2cell(cell2voxel, image_labeled):
    segmented_bool = image_labeled > 0
    cell2voxel = cell2voxel.copy()
    image_labeled = image_labeled.copy()
    cell2voxel = np.where(segmented_bool, cell2voxel, -1)
    image_labeled = np.where(segmented_bool, image_labeled, 0)

    cell2voxel_possible = np.isin(
        cell2voxel,
        (
            np.bincount(cell2voxel.reshape(-1)[cell2voxel.reshape(-1) >= 0]) > 1
        ).nonzero()[0],
        kind="table",
    )
    cell2voxel_possible = np.where(cell2voxel_possible, cell2voxel, -1)
    labels, zs_min, ys_min, xs_min, zs_max, ys_max, xs_max = nb_find_bbox(
        cell2voxel, min_value=0
    )
    image_labeled = nb_relabel(
        cell2voxel,
        image_labeled,
        labels,
        zs_min,
        ys_min,
        xs_min,
        zs_max,
        ys_max,
        xs_max,
    )

    return image_labeled


def get_cell_label(cell2voxel, image_labeled, num_cells):
    cell2voxel = cell2voxel.flatten()
    image_labeled = image_labeled.flatten()
    assert (
        cell2voxel.size == image_labeled.size
    ), "cell2voxel and image_labeled should have the same size"
    cell_voxel = np.empty(num_cells, dtype=np.int32)
    cell_voxel[:] = -1
    for i in range(cell2voxel.size):
        cell_ = cell2voxel[i]
        label_ = image_labeled[i]
        label_old = cell_voxel[cell_]
        if label_ > 0:
            # if label_old != -1:
            #     if label_ != label_old:
            #         raise ValueError("cell", cell_, "has two labels", label_old, label_)

            cell_voxel[cell_] = label_

    return cell_voxel


with h5py.File(
    PARAM.Path_data_h5,
    "r",
) as f:
    mesh_group = f["mesh"]
    voxel_group = f["voxel"]
    voxel_cell = voxel_group["voxel.cell"][:]
    cell_center = mesh_group["cell.center"][:]
    cells = mesh_group["cells"][:]
    cell_solid = mesh_group["cell.solid"][:]


image_mix, nets, seps = extraction.multi_phase_segmentation(
    image, config_list, fill_unlabeled=False, n_workers=len(config_list), backend="loky"
)
print(seps)
# image_mix = relabel_segments2cell(voxel_cell, image_mix)
### check contiuity of labels ###
image_labels = util.unique_uint_nonzero(image_mix)
if ~np.all(np.arange(1, image_labels.max() + 1) == image_labels):
    raise ValueError("labels are not continuous")

sep_void = seps[1]

image_mix_void_bool = image == 0
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

# cell_voxel_seg = get_cell_label(voxel_cell, image_mix, num_cells=cells.shape[0])
# cell_voxel_seg_indices = np.where(cell_voxel_seg > 0)[0]
# cell_voxel_seg_map = np.column_stack(
#     (cell_voxel_seg_indices, cell_voxel_seg[cell_voxel_seg_indices])
# )
# cell_voxel[cell_voxel_seg_map[:, 0]] = cell_voxel_seg_map[:, 1]

with h5py.File(PARAM.Path_data_h5, "a") as f:
    mesh_group = f["mesh"]
    if "cell.voxel" in mesh_group:
        del mesh_group["cell.voxel"]
    mesh_group.create_dataset(
        "cell.voxel",
        data=cell_voxel,
        compression="gzip",
    )

nets = extraction.extract_from_image(
    image_mix, resolution, n_workers=16, backend="loky", seps=seps
)
dualn = nets[0]
pn = nets[1]
sn = nets[2]
num_pn_pore = pn["pore._id"].size
num_solid_pore = sn["pore._id"].size

## Void Solid Properties
pn["pore.void"] = pn["pore.all"].copy()
pn["pore.solid"] = ~pn["pore.all"]
pn["throat.void"] = pn["throat.all"].copy()
pn["throat.solid"] = ~pn["throat.all"]

sn["pore.solid"] = sn["pore.all"].copy()
sn["pore.void"] = ~sn["pore.all"]
sn["throat.solid"] = sn["throat.all"].copy()
sn["throat.void"] = ~sn["throat.all"]

## Dual Properties
dualn["pore.void"][:num_pn_pore] = True
dualn["pore.solid"][num_pn_pore:] = True
dualn["throat.void"] = np.all(dualn["throat.conns"] < num_pn_pore, axis=1)
dualn["throat.solid"] = np.all(dualn["throat.conns"] >= num_pn_pore, axis=1)
dualn["throat.connect"] = ~(dualn["throat.void"] | dualn["throat.solid"])


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
