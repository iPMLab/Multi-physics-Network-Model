import time
import numba as nb
import numpy as np
from .numba_atomic import (
    nb_atomic_add,
    nb_atomic_sub,
    nb_atomic_max,
    nb_atomic_min,
)
from .numba_prefetch import _prefetch_ptr_write
from ._PARAM_JIT import PARAM_JIT
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


@nb.njit(parallel=False, **PARAM_JIT)
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


@nb.njit(parallel=True, **PARAM_JIT)
def nb_remap(arr, keys, values):
    Typed_Dict = {}
    for i in range(keys.size):
        Typed_Dict[keys[i]] = values[i]
    for i in nb.prange(arr.size):
        if arr[i] in Typed_Dict:
            arr[i] = Typed_Dict[arr[i]]
    return arr


@nb.njit(parallel=False, **PARAM_JIT)
def nb_isin(elements, test_elements):
    res = np.empty(elements.size, dtype=np.bool_)
    test_elements = set(test_elements)
    for i in range(elements.size):
        res[i] = elements[i] in test_elements
    return res


@nb.njit(parallel=True, **PARAM_JIT)
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


@nb.njit(parallel=True, **PARAM_JIT)
def nb_voxelise_cylinders(p0, p1, r, resolution, cube_x, cube_y, cube_z, verbose=False):
    offset = 1.1
    N = p0.shape[0]
    nz, ny, nx = (
        int(cube_z / resolution),
        int(cube_y / resolution),
        int(cube_x / resolution),
    )
    if verbose:
        print(f"nz: {nz}, ny: {ny}, nx: {nx}")
    image = np.zeros((nz, ny, nx), dtype=np.bool_)
    p0 = p0 / resolution
    p1 = p1 / resolution
    r = r / resolution

    # 预计算轴向向量和长度平方
    v = p1 - p0  # (N, 3)
    v_dot_v = (v**2).sum(axis=1)  # (N,)
    for ipar in nb.prange(N):
        p0_i = p0[ipar]  # (3,)
        p1_i = p1[ipar]  # (3,)
        v_i = v[ipar]  # (3,)
        vv = v_dot_v[ipar]
        r_i = r[ipar]
        r2 = r_i * r_i

        # 计算包围盒（保守估计）
        # 圆柱的包围盒 = 两个球（p1, r）和（p2, r）的并集的包围盒
        inv_len = 1.0 / np.sqrt(vv)
        ux = v_i[0] * inv_len
        uy = v_i[1] * inv_len
        uz = v_i[2] * inv_len

        # 计算各轴向的最大偏移（垂直于轴线的分量）
        dx = r_i * np.sqrt(max(0.0, 1.0 - ux * ux)) * offset
        dy = r_i * np.sqrt(max(0.0, 1.0 - uy * uy)) * offset
        dz = r_i * np.sqrt(max(0.0, 1.0 - uz * uz)) * offset

        x_min = max(0, int(np.floor(min(p0_i[0], p1_i[0]) - dx)))
        x_max = min(nx, int(np.ceil(max(p0_i[0], p1_i[0]) + dx)))
        y_min = max(0, int(np.floor(min(p0_i[1], p1_i[1]) - dy)))
        y_max = min(ny, int(np.ceil(max(p0_i[1], p1_i[1]) + dy)))
        z_min = max(0, int(np.floor(min(p0_i[2], p1_i[2]) - dz)))
        z_max = min(nz, int(np.ceil(max(p0_i[2], p1_i[2]) + dz)))

        for z_ in range(z_min, z_max):
            for y_ in range(y_min, y_max):
                for x_ in range(x_min, x_max):
                    P = np.array(
                        [
                            x_ + np.float32(0.5),
                            y_ + np.float32(0.5),
                            z_ + np.float32(0.5),
                        ],
                        dtype=np.float32,
                    )
                    w = P - p0_i  # (3,)
                    proj = (w * v_i).sum()  # dot(w, v_i)
                    t = proj / vv

                    if t <= 0.0 or t >= 1.0:
                        continue

                    # 最近点 Q = p1_i + t * v_i
                    qx = p0_i[0] + t * v_i[0]
                    qy = p0_i[1] + t * v_i[1]
                    qz = p0_i[2] + t * v_i[2]

                    dx = x_ - qx
                    dy = y_ - qy
                    dz = z_ - qz
                    dist2 = dx * dx + dy * dy + dz * dz

                    if dist2 < r2:
                        image[z_, y_, x_] = True

    return image


@nb.njit(parallel=True, **PARAM_JIT)
def nb_voxelise_spheres(x, y, z, r, nx, ny, nz):
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


@nb.njit(parallel=True, **PARAM_JIT)
def nb_voxelise(grid_out, tri_verts, positions, rotations):
    EPS_min = 1e-14
    # x direction oriented
    nz, ny, nx = grid_out.shape
    mask_odd = np.zeros((nz, ny), dtype=np.bool_)  # Mark odd intersection points

    num_particles = positions.shape[0]
    num_tri = tri_verts.shape[0]

    for i_P in nb.prange(num_particles):
        position_P = positions[i_P]
        rotation_P = rotations[i_P]
        if rotation_P.size == 1:
            tri_verts_P = tri_verts + position_P  # Only translate
        else:
            tri_verts_P = np.empty_like(tri_verts)
            # 手动展开 tri_verts @ rotation_P.T
            for i in range(num_tri):
                for j in range(3):  # 三个顶点
                    v = tri_verts[i, j, :]
                    # v_rotated[k] = sum_{l=0}^{2} v[l] * rotation_P.T[l, k] = sum_l v[l] * rotation_P[k, l]
                    for k in range(3):
                        tri_verts_P[i, j, k] = (
                            v[0] * rotation_P[k, 0]
                            + v[1] * rotation_P[k, 1]
                            + v[2] * rotation_P[k, 2]
                        )
            tri_verts_P = tri_verts_P + position_P
        # Precompute triangle AABBs

        tri_min = np.empty((num_tri, 3))  # 注意：应为 num_tri
        tri_max = np.empty((num_tri, 3))
        for i in range(num_tri):
            for k in range(3):  # x, y, z
                coords = tri_verts_P[i, :, k]
                tri_min[i, k] = min(coords)
                tri_max[i, k] = max(coords)

        tri_z_min, tri_z_max = tri_min[:, 2].min(), tri_max[:, 2].max()
        tri_y_min, tri_y_max = tri_min[:, 1].min(), tri_max[:, 1].max()
        # tri_x_min, tri_x_max = tri_min[:, 0].min(), tri_max[:, 0].max()

        grid_z_min, grid_z_max, grid_y_min, grid_y_max = (
            max(np.int64(tri_z_min), 0),
            min(np.int64(tri_z_max), nz - 1),
            max(np.int64(tri_y_min), 0),
            min(np.int64(tri_y_max), ny - 1),
        )

        for iz in nb.prange(grid_z_min, grid_z_max + 1):
            gz = np.float32(iz) + np.float32(0.5)
            for iy in nb.prange(grid_y_min, grid_y_max + 1):
                gy = np.float32(iy) + np.float32(0.5)
                x_crosses = np.empty(num_tri, dtype=np.float32)
                count = 0
                for t in range(num_tri):
                    # Fast AABB culling in Y and Z (since ray is along X)
                    if (
                        gy < tri_min[t, 1]
                        or gy > tri_max[t, 1]
                        or gz < tri_min[t, 2]
                        or gz > tri_max[t, 2]
                    ):
                        continue

                    # Extract triangle vertices
                    v0x, v0y, v0z = tri_verts_P[t, 0]  # shape (3,) -> [x, y, z]
                    v1x, v1y, v1z = tri_verts_P[t, 1]
                    v2x, v2y, v2z = tri_verts_P[t, 2]

                    # Edge v1 -> v2 in ZY plane
                    cross1 = (v2z - v1z) * (gy - v1y) - (v2y - v1y) * (gz - v1z)
                    cross0 = (v2z - v1z) * (v0y - v1y) - (v2y - v1y) * (v0z - v1z)
                    if cross1 * cross0 < 0:
                        continue  # outside

                    # Edge v2 -> v0 in ZY plane
                    cross2 = (v0z - v2z) * (gy - v2y) - (v0y - v2y) * (gz - v2z)
                    cross1b = (v0z - v2z) * (v1y - v2y) - (v0y - v2y) * (v1z - v2z)
                    if cross2 * cross1b < 0:
                        continue

                    # Edge v0 -> v1 in ZY plane
                    cross3 = (v1z - v0z) * (gy - v0y) - (v1y - v0y) * (gz - v0z)
                    cross2b = (v1z - v0z) * (v2y - v0y) - (v1y - v0y) * (v2z - v0z)
                    if cross3 * cross2b < 0:
                        continue

                    # Vector e1 = v1 - v0, e2 = v2 - v0
                    e1x = v1x - v0x
                    e1y = v1y - v0y
                    e1z = v1z - v0z

                    e2x = v2x - v0x
                    e2y = v2y - v0y
                    e2z = v2z - v0z

                    # Normal vector N = e1 × e2
                    A = e1y * e2z - e1z * e2y  # Nx
                    B = e1z * e2x - e1x * e2z  # Ny
                    C = e1x * e2y - e1y * e2x  # Nz
                    if abs(A) < EPS_min:
                        continue  # plane is parallel to X-axis; x not uniquely defined
                    D = -(A * v0x + B * v0y + C * v0z)

                    x_intersect = -(B * gy + C * gz + D) / A

                    # Optional: clamp to global mesh bounds (helps with floating errors)
                    # if tri_x_min - EPS_min <= x_intersect <= tri_x_max + EPS_min:
                    x_crosses[count] = x_intersect
                    count += 1

                if not count:
                    continue

                # Sort and deduplicate intersections
                x_crosses = np.sort(x_crosses[:count])
                unique_x = np.empty(count, dtype=np.float32)
                unique_x[0] = x_crosses[0]
                unique_x[1:] = x_crosses[1:][x_crosses[1:] - x_crosses[:-1] > 1e-6]

                n_cross = len(unique_x)
                if n_cross % 2 == 0:
                    # Fill even pairs
                    for k in range(0, n_cross, 2):
                        x0, x1 = unique_x[k], unique_x[k + 1]
                        ix_start = max(0, int(np.floor(x0)))
                        ix_end = min(nx, int(np.ceil(x1)))
                        for ix in range(ix_start, ix_end):
                            xc = np.float32(ix) + np.float32(0.5)
                            if x0 < xc < x1:
                                grid_out[iz, iy, ix] = True
                else:
                    # Mark for post-processing
                    mask_odd[iz, iy] = True

    # for iz, iy in nb.pndindex((nz, ny)):
    for iz in nb.prange(nz):
        for iy in nb.prange(ny):
            if not mask_odd[iz, iy]:
                continue

            # 动态收集有效邻居（不越界）
            neighbor_count = np.zeros(nx, dtype=np.uint8)
            dirs = (
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            )
            for dz, dy in dirs:
                zz, yy = iz + dz, iy + dy
                if 0 <= zz < nz and 0 <= yy < ny:
                    neighbor_count += grid_out[zz, yy, :]

            interp_mask = neighbor_count > 3
            grid_out[iz, iy, :] |= interp_mask

    return grid_out


@nb.njit(parallel=False, **PARAM_JIT)
def nb_add_at(arr, indices, values, inplace=True):
    if not inplace:
        arr = arr.copy()
    P = 64
    itemsize = arr.itemsize
    n = indices.size
    for i in range(n):
        addr = arr.ctypes.data + itemsize * indices[i + P]
        _prefetch_ptr_write(addr)
        arr[indices[i]] += values[i]
    return arr


@nb.njit(parallel=False, **PARAM_JIT)
def nb_sub_at(arr, indices, values, inplace=True):
    if not inplace:
        arr = arr.copy()
    P = 64
    itemsize = arr.itemsize
    n = indices.size
    for i in range(n):
        addr = arr.ctypes.data + itemsize * indices[i + P]
        _prefetch_ptr_write(addr)
        arr[indices[i]] -= values[i]
    return arr


@nb.njit(parallel=False, **PARAM_JIT)
def nb_max_at(arr, indices, values, inplace=True):
    if not inplace:
        arr = arr.copy()
    P = 64
    itemsize = arr.itemsize
    n = indices.size
    for i in range(n):
        addr = arr.ctypes.data + itemsize * indices[i + P]
        _prefetch_ptr_write(addr)
        arr[indices[i]] = max(arr[indices[i]], values[i])
    return arr


@nb.njit(parallel=False, **PARAM_JIT)
def nb_min_at(arr, indices, values, inplace=True):
    if not inplace:
        arr = arr.copy()
    P = 64
    itemsize = arr.itemsize
    n = indices.size
    for i in range(n):
        addr = arr.ctypes.data + itemsize * indices[i + P]
        _prefetch_ptr_write(addr)
        arr[indices[i]] = min(arr[indices[i]], values[i])
    return arr


if __name__ == "__main__":
    data = np.random.randint(0, 200, 200000)
    res_nb = nb_unique_uint(data)
    print(res_nb[1])
    res_np = np.unique(data, return_counts=True)
    print(res_np[1])
    assert np.array_equal(res_nb[0], res_np[0])
    assert np.array_equal(res_nb[1], res_np[1])
