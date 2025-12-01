import numpy as np
import pandas as pd
import os
import numba as nb
import copy
from contextlib import contextmanager
from ._utils_numba import (
    nb_remap,
    # nb_unique_uint,
    nb_isin,
    nb_unravel_index,
    nb_add_at,
    nb_get_image_from_coords,
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
            "Warning: 'throat.conns' array is not C contiguous. Converted to C contiguous.\nTo avoid this warning, ensure 'throat.conns' is C contiguous before passing to the function."
        )
    if "pore.void" not in mpn:
        mpn["pore.void"] = ~mpn["pore.all"].copy()
    if "pore.solid" not in mpn:
        mpn["pore.solid"] = ~mpn["pore.all"].copy()

    return mpn


@contextmanager
def nb_threads(n_workers=None):
    # 保存初始线程数
    initial_threads = nb.get_num_threads()
    if n_workers is None:
        n_workers = initial_threads
    elif n_workers < 0:
        n_workers = os.cpu_count() + n_workers + 1
    nb.set_num_threads(n_workers)
    try:
        # 执行 with 语句体的代码
        yield
    finally:
        # 恢复初始线程数
        nb.set_num_threads(initial_threads)


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
    arr,
    return_index=False,
    return_counts=False,
    equal_nan=False,
    zero=True,
):
    arr = np.asarray(arr).reshape(-1)
    res = np.unique(
        arr,
        return_index=return_index,
        return_counts=return_counts,
        equal_nan=equal_nan,
    )

    # 处理返回结果
    if not return_index and not return_counts:
        unique_values = res
        mask = unique_values >= 0 if zero else unique_values > 0
        return unique_values[mask]

    unique_values = res[0]
    mask = unique_values >= 0 if zero else unique_values > 0
    filtered_unique = unique_values[mask]

    # 构建返回元组
    result = [filtered_unique]
    idx = 1
    if return_index:
        result.append(res[idx][mask])
        idx += 1
    if return_counts:
        result.append(res[idx][mask])

    return tuple(result) if len(result) > 1 else result[0]


def unique_uint_nonzero(
    arr,
    return_index=False,
    return_counts=False,
    equal_nan=False,
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


def get_image_from_coords(
    x,
    y,
    z,
    r,
    resoltion,
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
    resoltion : float
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
    x = (x + offset_x) / resoltion
    y = (y + offset_y) / resoltion
    z = (z + offset_z) / resoltion
    r = r / resoltion
    image = nb_get_image_from_coords(
        x, y, z, r, image_size_x, image_size_y, image_size_z
    )
    return image


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


def add_at(arr, indices, values, inplace=True):
    if not inplace:
        arr = arr.copy()
    nb_add_at(arr, indices, values)
    return arr


def keys_in_dict(mpn, keys):
    for key in keys:
        assert key in mpn, f"{key} not in {keys}"


def hash_array(arr):
    return xxhash.xxh3_64_intdigest(arr.data)
