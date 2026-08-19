import numpy as np
import pandas as pd
import os
import numba as nb
import copy
from contextlib import contextmanager
from sparse_dot_mkl import mkl_set_num_threads_local, mkl_set_num_threads
from ._utils_numba import (
    nb_remap,
    # nb_unique_uint,
    nb_isin,
    nb_unravel_index,
    nb_atomic_add,
    nb_atomic_sub,
    nb_atomic_max,
    nb_atomic_min,
    nb_voxelise_spheres,
    nb_voxelise,
    nb_add_at,
    nb_sub_at,
    nb_max_at,
    nb_min_at,
)


import xxhash


def check_mpn(mpn, keys_in_mpn=None):
    if keys_in_mpn is not None:
        keys_not_in_mpn = frozenset(keys_in_mpn) - mpn.keys()
        if len(keys_not_in_mpn) > 0:
            raise KeyError(f"Keys {keys_not_in_mpn} not in mpn")

    if not mpn["throat.conns"].flags["C_CONTIGUOUS"]:
        mpn["throat.conns"] = np.ascontiguousarray(mpn["throat.conns"])
        print(
            "Warning: 'throat.conns' array is not C contiguous, converting to C contiguous."
        )
    if "pore.void" not in mpn:
        mpn["pore.void"] = ~mpn["pore.all"].copy()
    if "pore.solid" not in mpn:
        mpn["pore.solid"] = ~mpn["pore.all"].copy()

    return mpn


def is_inplace(mpn, inplace: bool):
    """
    Check if the network is inplace or not.
    Parameters
    ----------
    mpn : dict
        The network dictionary.
    inplace : bool
        Whether to perform the operation inplace or not.
    Returns
    -------
    dict
        The network dictionary.
    """
    if inplace:
        pass
    else:
        mpn = copy.deepcopy(mpn)
    return mpn


def to_list(x):
    if isinstance(x, str):
        return [x]
    else:
        return list(x)


def set_num_threads(num_threads, numba=True, numexpr=True, mkl=True, openmp=True):
    num_threads = round(num_threads)
    if num_threads < 0:
        num_threads = os.cpu_count() + num_threads + 1
    if numba:
        nb.set_num_threads(num_threads)
    if numexpr:
        os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    if mkl:
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        # mkl_set_num_threads_local(num_threads)
        mkl_set_num_threads(num_threads)
    if openmp:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)


def pd_enumerate_itertuples(df: pd.DataFrame, index=False, name="mpn"):
    """
    This function is used to iterate over the rows of a pandas DataFrame.
    Parameters:
    df: pandas DataFrame
    index: boolean, default False, whether to include the index as the first column
    name: string, default'mpn', the name of the index column
    Returns:
    A generator that yields each row of the DataFrame as a tuple.
    """
    for index, row in enumerate(df.itertuples(index=index, name=name)):
        yield index, row


def pd_itertuples(df: pd.DataFrame, index=False, name="mpn"):
    """
    This function is used to iterate over the rows of a pandas DataFrame.
    Parameters:
    df: pandas DataFrame
    index: boolean, default False, whether to include the index as the first column
    name: string, default'mpn', the name of the index column
    Returns:
    A generator that yields each row of the DataFrame as a tuple.
    """
    for row in df.itertuples(index=index, name=name):
        yield row


def pd_col_loc(df: pd.DataFrame, col_name):
    """
    This function is used to get the index of a column in a pandas DataFrame.
    Parameters:
    df: pandas DataFrame
    col_name: string, the name of the column
    Returns:
    An integer that represents the index of the column in the DataFrame.
    """
    return df.columns.get_loc(col_name)


# def unique_uint(arr, zero=True, return_counts=False):
#     arr = np.reshape(arr, -1)

#     res = nb_unique_uint(arr, zero)
#     if return_counts:
#         return res
#     else:
#         return res[0]


def unique_uint(
    arr, return_index=False, return_counts=False, equal_nan=True, zero=True
):
    arr = np.reshape(arr, -1)
    arr = arr[arr >= 0] if zero else arr[arr > 0]
    res = np.unique(
        arr, return_index=return_index, return_counts=return_counts, equal_nan=equal_nan
    )
    return res


def unique_uint_nonzero(
    arr,
    return_index=False,
    return_counts=False,
    equal_nan=True,
):
    return unique_uint(
        arr,
        return_index=return_index,
        return_counts=return_counts,
        equal_nan=equal_nan,
        zero=False,
    )


def remap(arr, keys, values, inplace=False):
    shape = arr.shape
    arr = np.asarray(arr)
    arr = np.reshape(arr, -1)
    keys = np.reshape(keys, -1)
    values = np.reshape(values, -1)
    if not inplace:
        arr = arr.copy()
    if arr.dtype != values.dtype:
        arr = arr.astype(values.dtype)
    if keys.dtype != values.dtype:
        keys = keys.astype(values.dtype)
    arr = nb_remap(arr, keys, values)
    arr = arr.reshape(shape)
    return arr


def isin(element, test_elements):
    element = np.asarray(element)
    shape = element.shape
    element = ravel(element)
    test_elements = ravel(element)

    res = nb_isin(element, test_elements)
    res = res.reshape(shape)
    return res


def unique_rows(
    ar,
    unordered_row=False,
    keepdims=True,
    return_index=False,
    return_inverse=False,
    return_counts=False,
):
    """Remove repeated rows from a 2D array.

    In particular, if given an array of coordinates of shape
    (Npoints, Ndim), it will remove repeated points.

    Parameters
    ----------
    ar : ndarray, shape (M, N)
        The input array.

    Returns
    -------
    ar_out : ndarray, shape (P, N)
        A copy of the input array with repeated rows removed.

    Raises
    ------
    ValueError : if `ar` is not two-dimensional.

    Notes
    -----
    The function will generate a copy of `ar` if it is not
    C-contiguous, which will negatively affect performance for large
    input arrays.

    Examples
    --------
    # >>> ar = np.array([[1, 0, 1],
    # ...                [0, 1, 0],
    # ...                [1, 0, 1]], np.uint8)
    # >>> unique_rows(ar)
    array([[0, 1, 0],
            [1, 0, 1]], dtype=uint8)
    """
    if ar.ndim != 2:
        raise ValueError(f"unique_rows() only makes sense for 2D arrays, got {ar.ndim}")
    # sort the rows if unordered_row is True
    if unordered_row:
        ar = np.sort(ar, axis=1)
    # the view in the next line only works if the array is C-contiguous
    ar = np.ascontiguousarray(ar)
    # np.unique() finds identical items in a raveled array. To make it
    # see each row as a single item, we create a view of each row as a
    # byte string of length itemsize times number of columns in `ar`
    ar_row_view = ar.view(f"|S{ar.itemsize * ar.shape[1]}")
    res_origin = np.unique(
        ar_row_view,
        return_index=True,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    res = []
    unique_byte, index = res_origin[0], res_origin[1]
    ar_out = ar[index]
    idx = 2
    res.append(ar_out)
    if return_index:
        res.append(index)
    if return_inverse:
        inverse = res_origin[idx]
        if not keepdims:
            inverse = inverse.reshape(-1)
        res.append(inverse)
        idx += 1
    if return_counts:
        counts = res_origin[idx]
        res.append(counts)
        idx += 1

    return tuple(res) if len(res) != 1 else res[0]


# def find_throat_conns_map(mpn_0_throat_conns, mpn_1_throat_conns):
#     df1 = pd.DataFrame(
#         np.concatenate(
#             (mpn_0_throat_conns, np.arange(len(mpn_0_throat_conns)).reshape(-1, 1)),
#             axis=1,
#         ),
#         columns=["a", "b", "index_0"],
#     )
#     df2 = pd.DataFrame(
#         np.arange(len(mpn_1_throat_conns)),
#         columns=["index_1"],
#         index=pd.MultiIndex.from_arrays(mpn_1_throat_conns.T, names=["a", "b"]),
#     )
#     df1 = df1.join(df2, how="left", on=["a", "b"])
#     throat_conns_map = df1[["index_0", "index_1"]][df1["index_1"].notna()].to_numpy(
#         dtype=np.int64
#     )
#     return throat_conns_map


def find_throat_conns_map(mpn_0_throat_conns, mpn_1_throat_conns, unordered_row=False):
    mpn_0_throat_conns = np.ascontiguousarray(mpn_0_throat_conns)
    mpn_1_throat_conns = np.ascontiguousarray(mpn_1_throat_conns)

    if unordered_row:
        mpn_0_throat_conns = np.sort(mpn_0_throat_conns, axis=1)
        mpn_1_throat_conns = np.sort(mpn_1_throat_conns, axis=1)
    # C - contiguous array
    mpn_0_throat_conns = mpn_0_throat_conns.view(
        f"|S{mpn_0_throat_conns.itemsize * mpn_0_throat_conns.shape[1]}"
    )
    mpn_1_throat_conns = mpn_1_throat_conns.view(
        f"|S{mpn_1_throat_conns.itemsize * mpn_1_throat_conns.shape[1]}"
    )
    val, ind0, ind1 = np.intersect1d(
        mpn_0_throat_conns, mpn_1_throat_conns, return_indices=True
    )
    throat_conns_map = np.column_stack((ind0, ind1))

    return throat_conns_map


def voxelize_spheres(
    x,
    y,
    z,
    r,
    resolution,
    image_size_x,
    image_size_y,
    image_size_z,
    offset_x=0,
    offset_y=0,
    offset_z=0,
):
    """
    Get the image of a set of points with a given resolution and image size.

    Parameters
    ----------
    x : array_like
        The x-coordinates of the points.
    y : array_like
        The y-coordinates of the points.
    z : array_like
        The z-coordinates of the points.
    r : array_like
        The radius of the points.
    resolution : float
        The resolution of the image.
    image_size_x : int
        The size of the image in the x-direction.
    image_size_y : int
        The size of the image in the y-direction.
    image_size_z : int
        The size of the image in the z-direction.
    offset_x : float
        The offset of the image in the x-direction.
    offset_y : float
        The offset of the image in the y-direction.
    offset_z : float
        The offset of the image in the z-direction.
    Returns
    -------
    image : ndarray, shape (image_size_z, image_size_y, image_size_x)
        The image of the points.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    z = np.asarray(z).reshape(-1)
    r = np.asarray(r).reshape(-1)
    assert x.size == y.size == z.size, "x, y, z should have the same size"
    x = (x + offset_x) / resolution
    y = (y + offset_y) / resolution
    z = (z + offset_z) / resolution
    r = r / resolution
    image = nb_voxelise_spheres(
        x=x,
        y=y,
        z=z,
        r=r,
        nx=image_size_x,
        ny=image_size_y,
        nz=image_size_z,
    )
    return image


def voxelize_particles(
    cube_x,
    cube_y,
    cube_z,
    verts,
    faces,
    positions,
    rotations=None,
    resolution=1.0,
    ray_directions="x",
    verbose=False,
    dtype=np.float32,
):
    """
    Voxelizes multiple instances of the same mesh at different positions/rotations.

    Parameters:
        cube_x, cube_y, cube_z: physical size of global cube in x-direction, y-direction, z-direction
        verts: (V, 3) original mesh vertices
        faces: (F, 3) face indices
        positions:   (P, 3) list of particle centers
        rotations:   (P, 3, 3) or None
        x, y, z: size of global grid in each direction
        resolution: resolution of global voxel grid
        ray_directions: 'x', 'y', 'z', or 'xyz'
    Returns:
        global_voxel: (nx, ny, nz) boolean array
    """

    positions = np.asarray(positions)
    assert len(positions.shape) == 2, (
        f"positions must be (num_particles, 3), got {positions.shape}"
    )
    num_particles = positions.shape[0]
    if rotations is not None:
        rotations = np.asarray(rotations)
        if rotations.shape == (3, 3):
            rotations = np.broadcast_to(rotations, (num_particles, 3, 3))
        assert rotations.shape == (num_particles, 3, 3), (
            f"rotations shape must be (num_particles, 3, 3), got {rotations.shape}"
        )
    else:
        rotations = np.empty((1, 1, 1), dtype=dtype)
        rotations = np.broadcast_to(rotations, (num_particles, 1, 1))

    nz = int(cube_z / resolution)
    ny = int(cube_y / resolution)
    nx = int(cube_x / resolution)

    if verbose:
        print(f"resolution: {resolution}, nx: {nx}, ny: {ny}, nz: {nz}")

    # Build base particle vertices once
    tri_verts = verts[faces]
    tri_verts /= resolution
    # mesh = pv.PolyData(verts / resolution, np.insert(faces, 0, 3, axis=1).reshape(-1))
    # mesh.plot()
    positions /= resolution

    tri_verts = tri_verts.astype(dtype, copy=False)
    positions = positions.astype(dtype, copy=False)
    rotations = rotations.astype(dtype, copy=False)

    results = []
    for ray_direction in ray_directions:
        if ray_direction == "x":
            grid_x = np.zeros((nz, ny, nx), dtype=np.bool_)
            tri_verts = np.ascontiguousarray(tri_verts)
            grid_x = nb_voxelise(grid_x, tri_verts, positions, rotations)
            results.append(grid_x)
        elif ray_direction == "y":
            # Swap x and y in vertices: (x,y,z) -> (y,x,z)
            tri_verts_y = tri_verts.copy()
            tri_verts_y[:, :, [0, 1]] = tri_verts_y[:, :, [1, 0]]
            tri_verts_y = np.ascontiguousarray(tri_verts_y)

            positions_y = positions.copy()
            positions_y[:, [0, 1]] = positions_y[:, [1, 0]]  # swap x and y
            positions_y = np.ascontiguousarray(positions_y)

            if rotations[0].size > 1:  # (N, 3, 3)
                P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=rotations.dtype)
                rotations_y = P @ rotations @ P.T
            else:
                rotations_y = rotations  # scalar or (3,3)

            grid_y = np.zeros((nz, nx, ny), dtype=np.bool_)
            grid_y = nb_voxelise(grid_y, tri_verts_y, positions_y, rotations_y)
            grid_y = np.ascontiguousarray(grid_y.transpose(0, 2, 1))

            results.append(grid_y)
        elif ray_direction == "z":
            # Swap x and z: (x,y,z) -> (z,y,x)
            tri_verts_z = tri_verts.copy()
            tri_verts_z[:, :, [0, 2]] = tri_verts_z[:, :, [2, 0]]
            tri_verts_z = np.ascontiguousarray(tri_verts_z)

            positions_z = positions.copy()
            positions_z[:, [0, 2]] = positions_z[:, [2, 0]]  # swap x and z
            positions_z = np.ascontiguousarray(positions_z)

            if rotations[0].size > 1:
                P = np.array(
                    [[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=rotations.dtype
                )  # shape (3, 3)
                rotations_z = P @ rotations @ P.T
            else:
                rotations_z = rotations

            grid_z = np.zeros((nx, ny, nz), dtype=np.bool_)
            grid_z = nb_voxelise(grid_z, tri_verts_z, positions_z, rotations_z)
            grid_z = np.ascontiguousarray(grid_z.transpose(2, 1, 0))
            results.append(grid_z)

        else:
            raise ValueError(
                f"ray_directions must be 'x', 'y', or 'z', got {ray_directions}"
            )
    num_grid = len(results)
    if num_grid == 1:
        grid_out = results[0]
    else:
        grid_out = np.zeros((nz, ny, nx), dtype=np.uint8)
        for grid in results:
            grid_out += grid
        grid_out = grid_out > 0

    return grid_out


def ravel(arr):
    return np.reshape(arr, -1)


def unravel_index(indices, shape):
    indices = np.reshape(indices, -1)
    shape = np.reshape(shape, -1)
    res = nb_unravel_index(indices, shape)
    res = tuple(res.T)
    return res


def dicts2df(list_of_dicts, subset=None, keep="first"):
    df = pd.DataFrame(list_of_dicts)
    assert all(d.keys() == list_of_dicts[0].keys() for d in list_of_dicts)
    df = df.apply(lambda col: np.concatenate(col.values), axis=0)
    df.drop_duplicates(subset=subset, keep=keep, inplace=True, ignore_index=True)
    return df


def dfs2df(dfs, subset="ids", keep="first"):
    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(subset=subset, keep=keep, inplace=True, ignore_index=True)
    return df


def add_at(arr, indices, values, inplace=True, parallel=False):
    if arr.dtype is not values.dtype:
        values = values.astype(arr.dtype)
    if parallel:
        arr = nb_atomic_add(arr, indices, values, inplace=inplace)
    else:
        arr = nb_add_at(arr, indices, values, inplace=inplace)

    return arr


def sub_at(arr, indices, values, inplace=True, parallel=False):
    if arr.dtype is not values.dtype:
        values = values.astype(arr.dtype)
    if parallel:
        arr = nb_atomic_sub(arr, indices, values, inplace=inplace)
    else:
        arr = nb_sub_at(arr, indices, values, inplace=inplace)
    return arr


def max_at(arr, indices, values, inplace=True, parallel=False):
    if arr.dtype is not values.dtype:
        values = values.astype(arr.dtype)
    if parallel:
        arr = nb_atomic_max(arr, indices, values, inplace=inplace)
    else:
        arr = nb_max_at(arr, indices, values, inplace=inplace)
    return arr


def min_at(arr, indices, values, inplace=True, parallel=False):
    if arr.dtype is not values.dtype:
        values = values.astype(arr.dtype)
    if parallel:
        arr = nb_atomic_min(arr, indices, values, inplace=inplace)
    else:
        arr = nb_min_at(arr, indices, values, inplace=inplace)
    return arr


def keys_in_dict(mpn, keys):
    for key in keys:
        assert key in mpn, f"{key} not in {keys}"


def hash_array(arr):
    return xxhash.xxh3_64_intdigest(arr.data)


def randomize_colors(im, keep_vals=[0]):
    r"""
    Copied from porespy.tools.randomize_colors
    -----------
    Takes a greyscale image and randomly shuffles the greyscale values.

    In other words, all voxels labeled X will be labelled Y, and all voxels
    labeled Y will be labeled Z, and so on, where X, Y, Z are randomly selected
    from the values in the input image.

    Parameters
    ----------
    im : array_like
        An ND image of greyscale values.
    keep_vals : array_like
        Indicate which voxel values should NOT be altered.  The default is
        `[0]` which is useful for leaving the background of the image
        untouched.

    Returns
    -------
    image : ndarray
        An image the same size and type as ``im`` but with the greyscale values
        reassigned.  The unique values in both the input and output images will
        be identical.

    Notes
    -----
    If the greyscale values in the input image are not contiguous then the
    neither will they be in the output.

    This function is useful for improving the visibility of images with
    neighboring regions that are only incrementally different from each other,
    such as that returned by `scipy.ndimage.label`.

    Examples
    --------
    >>> import porespy as ps
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> im = np.random.randint(low=0, high=5, size=[4, 4])
    >>> print(im)
    [[4 0 3 3]
     [3 1 3 2]
     [4 0 0 4]
     [2 1 0 1]]
    >>> im_rand = ps.tools.randomize_colors(im)
    >>> print(im_rand)
    [[2 0 4 4]
     [4 1 4 3]
     [2 0 0 2]
     [3 1 0 1]]

    As can be seen, the 2's have become 3, 3's have become 4, and 4's have
    become 2.  1's remained 1 by random accident.  0's remain zeros by default,
    but this can be controlled using the `keep_vals` argument.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/randomize_colors.html>`__
    to view online example.

    """
    im_flat = im.flatten()
    keep_vals = np.array(keep_vals)
    swap_vals = ~np.isin(im_flat, keep_vals)
    im_vals = np.unique(im_flat[swap_vals])
    new_vals = np.random.permutation(im_vals)
    im_map = np.zeros(
        shape=[
            np.amax(im_vals) + 1,
        ],
        dtype=int,
    )
    im_map[im_vals] = new_vals
    im_new = im_map[im_flat]
    im_new = np.reshape(im_new, shape=np.shape(im))
    return im_new
