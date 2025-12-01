import time
import numba as nb
import numpy as np
from .numpy_atomic import (
    atomic_add,
    atomic_sub,
    atomic_max,
    atomic_min,
)


# @nb.njit(fastmath=True, cache=True, nogil=True, parallel=True)
# def nb_unique_uint(arr, zero=True):
#     counts = np.zeros(arr.max() + 1, dtype=np.int64)
#     if zero:
#         for i in nb.prange(arr.size):
#             if arr[i] >= 0:
#                 atomic_add(counts, arr[i], 1)
#     else:
#         for i in nb.prange(arr.size):
#             if arr[i] > 0:
#                 atomic_add(counts, arr[i], 1)
#     unique_elements = np.nonzero(counts)[0]
#     return unique_elements, counts[unique_elements]


@nb.njit(fastmath=True, cache=True, nogil=True, parallel=False, error_model="numpy")
def nb_unique_uint(arr, zero=True):
    counts = np.zeros(arr.max() + 1, dtype=np.int64)
    if zero:
        for i in range(arr.size):
            arr_i = arr[i]
            counts[arr_i] += arr_i >= 0
    else:
        for i in range(arr.size):
            arr_i = arr[i]
            counts[arr_i] += arr_i > 0
    unique_elements = np.nonzero(counts)[0]
    return unique_elements, counts[unique_elements]


@nb.njit(fastmath=True, cache=True, nogil=True, parallel=True, error_model="numpy")
def nb_remap(arr, keys, values):
    Typed_Dict = {}
    for i in range(keys.size):
        Typed_Dict[keys[i]] = values[i]
    for i in nb.prange(arr.size):
        if arr[i] in Typed_Dict:
            arr[i] = Typed_Dict[arr[i]]
    return arr


@nb.njit(fastmath=True, cache=True, nogil=True, parallel=False, error_model="numpy")
def nb_isin(elements, test_elements):
    res = np.empty(elements.size, dtype=np.bool_)
    test_elements = set(test_elements)
    for i in range(elements.size):
        res[i] = elements[i] in test_elements
    return res


@nb.njit(fastmath=True, cache=True, nogil=True, parallel=True, error_model="numpy")
def nb_unravel_index(indices, shape):
    assert np.max(indices) < shape.prod(), "max flat index out of range"
    len_shape = len(shape)
    indices_size = indices.size
    multi_indices = np.empty((indices_size, len_shape), dtype=np.int64)
    for i in nb.prange(indices_size):
        index = indices[i]
        for j in range(len_shape - 1, -1, -1):
            multi_indices[i, j] = index % shape[j]
            index //= shape[j]
    return multi_indices


@nb.njit(parallel=True, fastmath=True, nogil=True, cache=True, error_model="numpy")
def nb_get_image_from_coords(x, y, z, r, nx, ny, nz):
    offset = 1.1
    num_ball = x.shape[0]
    image = np.zeros((nz, ny, nx), dtype=np.bool_)
    for ipar in nb.prange(0, num_ball):
        x_i = x[ipar]
        y_i = y[ipar]
        z_i = z[ipar]
        r_i = r[ipar]
        x_min = max(0, int(np.floor(x_i - offset * r_i)))
        x_max = min(nx, int(np.ceil(x_i + offset * r_i)))
        y_min = max(0, int(np.floor(y_i - offset * r_i)))
        y_max = min(ny, int(np.ceil(y_i + offset * r_i)))
        z_min = max(0, int(np.floor(z_i - offset * r_i)))
        z_max = min(nz, int(np.ceil(z_i + offset * r_i)))
        r2 = r_i**2
        for z_ in range(z_min, z_max):
            dz2 = (z_ - z_i) ** 2
            for y_ in range(y_min, y_max):
                dz2dy2 = dz2 + (y_ - y_i) ** 2
                for x_ in range(x_min, x_max):
                    dz2dy2dx2 = dz2dy2 + (x_ - x_i) ** 2
                    if dz2dy2dx2 < r2:
                        image[z_, y_, x_] = True

    return image

@nb.njit(fastmath=True, cache=True, nogil=True, parallel=False, error_model="numpy")
def nb_add_at(arr, indices, values):
    for i in range(indices.size):
        arr[indices[i]] += values[i]
    return arr


# @nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
# def nb_add_at(arr, indices, values):
#      for i in nb.prange(len(indices)):
#          atomic_add(arr, indices[i], values[i])

# @nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
# def nb_sub_at(arr, indices, values):
#      for i in nb.prange(len(indices)):
#          atomic_sub(arr, indices[i], values[i])
#
# @nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
# def nb_max_at(arr, indices, values):
#      for i in nb.prange(len(indices)):
#          atomic_max(arr, indices[i], values[i])
#
# @nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
# def nb_min_at(arr, indices, values):
#      for i in nb.prange(len(indices)):
#          atomic_min(arr, indices[i], values[i])


if __name__ == "__main__":
    data = np.random.randint(0, 200, 200000)
    res_nb = nb_unique_uint(data)
    print(res_nb[1])
    res_np = np.unique(data, return_counts=True)
    print(res_np[1])
    assert np.all(res_nb[0] == res_np[0])
    assert np.all(res_nb[1] == res_np[1])
