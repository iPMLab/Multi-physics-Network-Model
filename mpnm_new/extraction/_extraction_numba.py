import numba as nb
import numpy as np
from scipy import sparse
from typing import Tuple
from scipy import ndimage as ndi
from ._minkowski_numba import nb_functionals_2d, nb_functionals_3d
from ._edt_numba import nb_edt

# def _build_variable_indices(band: np.ndarray) -> np.ndarray:
#     num_variables = np.count_nonzero(band)
#     variable_indices = np.full(band.shape, -1, dtype=np.int32)
#     variable_indices[band] = np.arange(num_variables)
#     return variable_indices


# def signed_distance_function(
#     levelset: np.ndarray, band_radius: int
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Return the distance to the 0.5 levelset of a function, the mask of the
#     border (i.e., the nearest cells to the 0.5 level-set) and the mask of the
#     band (i.e., the cells of the function whose distance to the 0.5 level-set
#     is less of equal to `band_radius`).
#     """

#     binary_array = np.where(levelset > 0, True, False)

#     # Compute the band and the border.
#     dist_func = ndi.distance_transform_edt
#     distance = np.where(
#         binary_array, dist_func(binary_array) - 0.5, -dist_func(~binary_array) + 0.5
#     )
#     border = np.abs(distance) < 1
#     band = np.abs(distance) <= band_radius

#     return distance, border, band


# def _buildq2d(variable_indices: np.ndarray):
#     """
#     Builds the filterq matrix for the given variables.

#     Version for 2 dimensions.
#     """

#     num_variables = variable_indices.max() + 1
#     filterq = sparse.lil_matrix((3 * num_variables, num_variables))

#     # Pad variable_indices to simplify out-of-bounds accesses
#     variable_indices = np.pad(
#         variable_indices, [(1, 1), (1, 1)], mode="constant", constant_values=-1
#     )

#     coords = np.nonzero(variable_indices >= 0)
#     for count, (i, j) in enumerate(zip(*coords)):

#         assert variable_indices[i, j] == count

#         filterq[2 * count, count] = -2
#         neighbor = variable_indices[i - 1, j]
#         if neighbor >= 0:
#             filterq[2 * count, neighbor] = 1
#         else:
#             filterq[2 * count, count] += 1

#         neighbor = variable_indices[i + 1, j]
#         if neighbor >= 0:
#             filterq[2 * count, neighbor] = 1
#         else:
#             filterq[2 * count, count] += 1

#         filterq[2 * count + 1, count] = -2
#         neighbor = variable_indices[i, j - 1]
#         if neighbor >= 0:
#             filterq[2 * count + 1, neighbor] = 1
#         else:
#             filterq[2 * count + 1, count] += 1

#         neighbor = variable_indices[i, j + 1]
#         if neighbor >= 0:
#             filterq[2 * count + 1, neighbor] = 1
#         else:
#             filterq[2 * count + 1, count] += 1

#     filterq = filterq.tocsr()
#     return filterq  # .T.dot(filterq)


# def _buildq3d(variable_indices: np.ndarray):
#     """
#     Builds the filterq matrix for the given variables.
#     """

#     num_variables = variable_indices.max() + 1
#     filterq = sparse.lil_matrix((3 * num_variables, num_variables))

#     # Pad variable_indices to simplify out-of-bounds accesses
#     variable_indices = np.pad(
#         variable_indices, [(0, 1), (0, 1), (0, 1)], mode="constant", constant_values=-1
#     )

#     coords = np.nonzero(variable_indices >= 0)
#     for count, (i, j, k) in enumerate(zip(*coords)):

#         assert variable_indices[i, j, k] == count

#         filterq[3 * count, count] = -2
#         neighbor = variable_indices[i - 1, j, k]
#         if neighbor >= 0:
#             filterq[3 * count, neighbor] = 1
#         else:
#             filterq[3 * count, count] += 1

#         neighbor = variable_indices[i + 1, j, k]
#         if neighbor >= 0:
#             filterq[3 * count, neighbor] = 1
#         else:
#             filterq[3 * count, count] += 1

#         filterq[3 * count + 1, count] = -2
#         neighbor = variable_indices[i, j - 1, k]
#         if neighbor >= 0:
#             filterq[3 * count + 1, neighbor] = 1
#         else:
#             filterq[3 * count + 1, count] += 1

#         neighbor = variable_indices[i, j + 1, k]
#         if neighbor >= 0:
#             filterq[3 * count + 1, neighbor] = 1
#         else:
#             filterq[3 * count + 1, count] += 1

#         filterq[3 * count + 2, count] = -2
#         neighbor = variable_indices[i, j, k - 1]
#         if neighbor >= 0:
#             filterq[3 * count + 2, neighbor] = 1
#         else:
#             filterq[3 * count + 2, count] += 1

#         neighbor = variable_indices[i, j, k + 1]
#         if neighbor >= 0:
#             filterq[3 * count + 2, neighbor] = 1
#         else:
#             filterq[3 * count + 2, count] += 1

#     filterq = filterq.tocsr()
#     return filterq


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_buildq(variable_indices):
    ndim = variable_indices.ndim
    direction_num = 4 if ndim == 2 else 6
    coords = np.column_stack(np.where(variable_indices >= 0))
    num_indices = coords.shape[0]
    map_neighbors = np.full((direction_num, num_indices), -1, dtype=np.int64)
    neighbors_num_false = np.zeros((ndim, num_indices), dtype=np.int64)

    if ndim == 2:
        structure = np.array(((-1, 0), (1, 0), (0, -1), (0, 1)))
        for i in nb.prange(num_indices):
            x = coords[i, 0]
            y = coords[i, 1]
            for j in range(direction_num):
                dx = structure[j, 0]
                dy = structure[j, 1]
                x_ = x + dx
                y_ = y + dy
                val = variable_indices[x_, y_]
                if val >= 0:
                    map_neighbors[j, i] = val
                else:
                    neighbors_num_false[j // 2, i] += 1
    else:
        structure = np.array(
            ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
        )
        for i in nb.prange(num_indices):
            x = coords[i, 0]
            y = coords[i, 1]
            z = coords[i, 2]

            for j in range(direction_num):
                dx = structure[j, 0]
                dy = structure[j, 1]
                dz = structure[j, 2]
                x_ = x + dx
                y_ = y + dy
                z_ = z + dz
                val = variable_indices[x_, y_, z_]
                if val >= 0:
                    map_neighbors[j, i] = val
                else:
                    neighbors_num_false[j // 2, i] += 1

    valid_mask = map_neighbors >= 0
    cols_true, rows_true = np.where(valid_mask)
    neighbors = map_neighbors.reshape(-1)[valid_mask.reshape(-1)]
    neighbors_num = len(neighbors)

    num_nondig = neighbors_num
    num_diag = num_indices * ndim
    num_total = num_diag + num_nondig
    rows_cols_data = np.empty((3, num_total), dtype=np.int64)
    arange = np.arange(num_indices)
    for d in range(ndim):
        start = d * num_indices
        end = (d + 1) * num_indices
        rows_cols_data[0, start:end] = arange * ndim + d
        rows_cols_data[1, start:end] = arange
        rows_cols_data[2, start:end] = -2 + neighbors_num_false[d]

    # 填充非对角线部分
    rows_cols_data[0, num_diag:] = rows_true * ndim + cols_true // 2
    rows_cols_data[1, num_diag:] = neighbors
    rows_cols_data[2, num_diag:] = 1
    return rows_cols_data


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_binary_dilation(
    image: np.ndarray, structure: np.ndarray, iterations: int = 1
) -> np.ndarray:
    assert image.ndim == structure.ndim
    new_image = np.zeros(image.shape, dtype=np.bool_)
    ndim = image.ndim

    if ndim == 1:
        x_img = image.shape[0]
        x_struct = structure.shape[0]
        x_offset = x_struct // 2
        x_struct_offset = np.where(structure)[0] - x_offset
        dir_num = len(x_struct_offset)
        for it in range(iterations):
            for x in nb.prange(x_img):
                if image[x]:  # 仅检查结构元素中为True的位置
                    for i in range(dir_num):
                        img_x = x + x_struct_offset[i]
                        if 0 <= img_x < x_img:
                            new_image[img_x] = True
    elif ndim == 2:
        y_img, x_img = image.shape
        y_struct, x_struct = structure.shape
        y_offset, x_offset = y_struct // 2, x_struct // 2  # 结构元素中心偏移
        y_struct_offset, x_struct_offset = np.where(structure)
        y_struct_offset -= y_offset
        x_struct_offset -= x_offset
        dir_num = len(x_struct_offset)
        for it in range(iterations):
            for y, x in nb.pndindex(image.shape):
                if image[y, x]:  # 仅检查结构元素中为True的位置
                    for i in range(dir_num):
                        img_y = y + y_struct_offset[i]
                        img_x = x + x_struct_offset[i]
                        if 0 <= img_y < y_img and 0 <= img_x < x_img:
                            new_image[img_y, img_x] = True
    else:
        z_img, y_img, x_img = image.shape
        z_struct, y_struct, x_struct = structure.shape
        z_offset, y_offset, x_offset = (
            z_struct // 2,
            y_struct // 2,
            x_struct // 2,
        )
        z_struct_offset, y_struct_offset, x_struct_offset = np.where(structure)
        z_struct_offset -= z_offset
        y_struct_offset -= y_offset
        x_struct_offset -= x_offset
        dir_num = len(x_struct_offset)
        for it in range(iterations):
            for z, y, x in nb.pndindex(image.shape):
                if image[z, y, x]:  # 仅检查结构元素中为True的位置
                    for i in range(dir_num):
                        img_z = z + z_struct_offset[i]
                        img_y = y + y_struct_offset[i]
                        img_x = x + x_struct_offset[i]
                        if (
                            0 <= img_z < z_img
                            and 0 <= img_y < y_img
                            and 0 <= img_x < x_img
                        ):
                            new_image[img_z, img_y, img_x] = True

    return new_image


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_binary_erosion(
    image: np.ndarray, structure: np.ndarray, iterations: int = 1
) -> np.ndarray:
    new_image = np.zeros(image.shape, dtype=np.bool_)
    ndim = image.ndim
    if ndim == 1:
        x_img = image.shape[0]
        x_struct = structure.shape[0]
        x_offset = x_struct // 2
        x_struct_offset = np.where(structure)[0] - x_offset
        for it in range(iterations):
            for x in nb.prange(x_img):
                valid = True
                for dx in x_struct_offset:
                    img_x = x + dx
                    if img_x < 0 or img_x >= x_img:
                        valid = False
                    elif image[img_x] == False:
                        valid = False
                new_image[x] = valid

    elif ndim == 2:
        y_img, x_img = image.shape
        y_struct, x_struct = structure.shape
        y_offset, x_offset = y_struct // 2, x_struct // 2  # 结构元素中心偏移
        y_struct_offset, x_struct_offset = np.where(structure)
        y_struct_offset -= y_offset
        x_struct_offset -= x_offset
        for it in range(iterations):
            for y, x in nb.pndindex(image.shape):
                valid = True
                for dy, dx in zip(y_struct_offset, x_struct_offset):
                    img_y = y + dy
                    img_x = x + dx
                    if img_y < 0 or img_y >= y_img or img_x < 0 or img_x >= x_img:
                        valid = False
                    elif image[img_y, img_x] == False:
                        valid = False
                new_image[y, x] = valid
    else:
        z_img, y_img, x_img = image.shape
        z_struct, y_struct, x_struct = structure.shape
        z_offset, y_offset, x_offset = (
            z_struct // 2,
            y_struct // 2,
            x_struct // 2,
        )
        z_struct_offset, y_struct_offset, x_struct_offset = np.where(structure)
        z_struct_offset -= z_offset
        y_struct_offset -= y_offset
        x_struct_offset -= x_offset
        for it in range(iterations):
            for z, y, x in nb.pndindex(image.shape):
                valid = True
                for dz, dy, dx in zip(
                    z_struct_offset, y_struct_offset, x_struct_offset
                ):
                    img_z = z + dz
                    img_y = y + dy
                    img_x = x + dx
                    if (
                        img_z < 0
                        or img_z >= z_img
                        or img_y < 0
                        or img_y >= y_img
                        or img_x < 0
                        or img_x >= x_img
                    ):
                        valid = False
                    elif not image[img_z, img_y, img_x]:
                        valid = False
                new_image[z, y, x] = valid

    return new_image


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_max_filter_non_padding(image_padded, struct):
    """
    image_padded is original image to be padded with np.pad(((struct.shape[0] // 2, struct.shape[0] // 2),)*ndim)
    """
    # 计算填充量
    ndim = image_padded.ndim
    new_image_shape = [
        image_padded.shape[i] - struct.shape[i] // 2 * 2 for i in range(ndim)
    ]
    if ndim == 1:
        x_img = new_image_shape[0]
        new_image = np.zeros(
            x_img,
            dtype=image_padded.dtype,
        )
        x_struct_offset = np.where(struct)[0]
        for x in nb.prange(new_image.shape[0]):
            max_val = -np.inf  # 初始化当前像素的最大值
            for dx in x_struct_offset:
                img_x = x + dx
                if image_padded[img_x] > max_val:
                    max_val = image_padded[img_x]
            new_image[x] = max_val

    elif ndim == 2:
        y_img, x_img = new_image_shape
        new_image = np.zeros(
            (y_img, x_img),
            dtype=image_padded.dtype,
        )
        y_struct_offset, x_struct_offset = np.where(struct)
        for y, x in nb.pndindex(new_image.shape):
            max_val = -np.inf  # 初始化当前像素的最大值
            for dy, dx in zip(y_struct_offset, x_struct_offset):
                img_y = y + dy
                img_x = x + dx
                if image_padded[img_y, img_x] > max_val:
                    max_val = image_padded[img_y, img_x]
            new_image[y, x] = max_val
    else:
        z_img, y_img, x_img = new_image_shape
        new_image = np.zeros(
            (z_img, y_img, x_img),
            dtype=image_padded.dtype,
        )
        z_struct_offset, y_struct_offset, x_struct_offset = np.where(struct)
        for z, y, x in nb.pndindex(new_image.shape):
            max_val = -np.inf  # 初始化当前像素的最大值
            for dz, dy, dx in zip(z_struct_offset, y_struct_offset, x_struct_offset):
                img_z = z + dz
                img_y = y + dy
                img_x = x + dx
                if image_padded[img_z, img_y, img_x] > max_val:
                    max_val = image_padded[img_z, img_y, img_x]
            new_image[z, y, x] = max_val

    return new_image


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_compute_surface_area_voxel(grid):
    area = 0
    nx, ny, nz = grid.shape
    for x in nb.prange(nx):
        for y in nb.prange(ny):
            for z in nb.prange(nz):
                if grid[x, y, z] == 0:
                    continue
                # 检查6个方向
                if x == 0 or grid[x - 1, y, z] == 0:
                    area += 1
                if x == nx - 1 or grid[x + 1, y, z] == 0:
                    area += 1
                if y == 0 or grid[x, y - 1, z] == 0:
                    area += 1
                if y == ny - 1 or grid[x, y + 1, z] == 0:
                    area += 1
                if z == 0 or grid[x, y, z - 1] == 0:
                    area += 1
                if z == nz - 1 or grid[x, y, z + 1] == 0:
                    area += 1
    return area


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_get_objects_volume(labeled_image, label_max):
    label_num = label_max + 1
    nz, ny, nx = labeled_image.shape
    volume = np.zeros(label_num, dtype=np.int64)
    z_min = np.full(label_num, nz + 1, dtype=np.int64)
    z_max = volume.copy()
    y_min = np.full(label_num, ny + 1, dtype=np.int64)
    y_max = volume.copy()
    x_min = np.full(label_num, nx + 1, dtype=np.int64)
    x_max = volume.copy()

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                value = labeled_image[z, y, x]
                if value < 0:
                    continue
                z_min[value] = min(z_min[value], z)
                z_max[value] = max(z_max[value], z + 1)
                y_min[value] = min(y_min[value], y)
                y_max[value] = max(y_max[value], y + 1)
                x_min[value] = min(x_min[value], x)
                x_max[value] = max(x_max[value], x + 1)
                volume[value] += 1
    return z_min, z_max, y_min, y_max, x_min, x_max, volume
