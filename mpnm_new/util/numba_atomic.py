import numba as nb
import numpy as np
from typing import Literal
from numba.np.arrayobj import make_array, basic_indexing, normalize_indices

from numba import types


def _make_atomic_intrinsic(op_map, doc):
    """
    Factory to create atomic intrinsic functions for Numba.

    Parameters:
    - op_map: dict of {predicate(dtype) -> LLVM atomic operation name}
    - doc: docstring for the intrinsic
    """

    @nb.extending.intrinsic
    def intrinsic(typingctx, arr, idx, val):
        if not isinstance(arr, types.Array):
            raise TypeError("First argument must be an array")

        dtype = arr.dtype

        # Find matching op
        op = None
        for predicate, op_name in op_map.items():
            if predicate(dtype):
                op = op_name
                break

        if op is None:
            raise TypeError(f"Atomic operation not supported for dtype: {dtype}")

        result_type = dtype

        def codegen(context, builder, signature, args):
            arr_arg, idx_arg, val_arg = args
            arr_ty, idx_ty, val_ty = signature.args

            llary = make_array(arr_ty)(context, builder, arr_arg)
            index_types, indices = normalize_indices(
                context, builder, [idx_ty], [idx_arg]
            )
            view_data, _, _ = basic_indexing(
                context,
                builder,
                arr_ty,
                llary,
                index_types,
                indices,
                boundscheck=context.enable_boundscheck,
            )
            return builder.atomic_rmw(op, view_data, val_arg, "monotonic")

        return result_type(arr, idx, val), codegen

    intrinsic.__doc__ = doc
    return intrinsic


# Now define all atomics concisely:

_atomic_add = _make_atomic_intrinsic(
    op_map={
        lambda d: isinstance(d, types.Integer): "add",
        lambda d: isinstance(d, types.Float): "fadd",
    },
    doc="arr[idx] += val",
)

_atomic_sub = _make_atomic_intrinsic(
    op_map={
        lambda d: isinstance(d, types.Integer): "sub",
        lambda d: isinstance(d, types.Float): "fsub",
    },
    doc="arr[idx] -= val",
)

_atomic_and = _make_atomic_intrinsic(
    op_map={lambda d: isinstance(d, types.Integer): "and"}, doc="arr[idx] &= val"
)

_atomic_nand = _make_atomic_intrinsic(
    op_map={lambda d: isinstance(d, types.Integer): "nand"},
    doc="arr[idx] = ~(arr[idx] & val)",
)

_atomic_or = _make_atomic_intrinsic(
    op_map={lambda d: isinstance(d, types.Integer): "or"}, doc="arr[idx] |= val"
)

_atomic_xor = _make_atomic_intrinsic(
    op_map={lambda d: isinstance(d, types.Integer): "xor"}, doc="arr[idx] ^= val"
)

_atomic_max = _make_atomic_intrinsic(
    op_map={
        lambda d: isinstance(d, types.Integer) and d.signed: "max",
        lambda d: isinstance(d, types.Integer) and not d.signed: "umax",
        lambda d: isinstance(d, types.Float): "fmax",
    },
    doc="arr[idx] = max(arr[idx], val)",
)

_atomic_min = _make_atomic_intrinsic(
    op_map={
        lambda d: isinstance(d, types.Integer) and d.signed: "min",
        lambda d: isinstance(d, types.Integer) and not d.signed: "umin",
        lambda d: isinstance(d, types.Float): "fmin",
    },
    doc="arr[idx] = min(arr[idx], val)",
)


@nb.njit(parallel=False, fastmath=True, nogil=True, cache=True, error_model="numpy")
def _atomic_add_single(arr, idx, val):
    for i in range(idx.size):
        arr[idx[i]] += val[i]


@nb.njit(parallel=False, fastmath=True, nogil=True, cache=True, error_model="numpy")
def _atomic_sub_single(arr, idx, val):
    for i in range(idx.size):
        arr[idx[i]] -= val[i]


@nb.njit(parallel=False, fastmath=True, nogil=True, cache=True, error_model="numpy")
def _atomic_and_single(arr, idx, val):
    for i in range(idx.size):
        arr[idx[i]] &= val[i]


@nb.njit(parallel=False, fastmath=True, nogil=True, cache=True, error_model="numpy")
def _atomic_nand_single(arr, idx, val):
    for i in range(idx.size):
        arr[idx[i]] = ~(arr[idx[i]] & val[i])


@nb.njit(parallel=False, fastmath=True, nogil=True, cache=True, error_model="numpy")
def _atomic_or_single(arr, idx, val):
    for i in range(idx.size):
        arr[idx[i]] |= val[i]


@nb.njit(parallel=False, fastmath=True, nogil=True, cache=True, error_model="numpy")
def _atomic_xor_single(arr, idx, val):
    for i in range(idx.size):
        arr[idx[i]] ^= val[i]


@nb.njit(parallel=False, fastmath=True, nogil=True, cache=True, error_model="numpy")
def _atomic_max_single(arr, idx, val):
    for i in range(idx.size):
        arr[idx[i]] = max(arr[idx[i]], val[i])


@nb.njit(parallel=False, fastmath=True, nogil=True, cache=True, error_model="numpy")
def _atomic_min_single(arr, idx, val):
    for i in range(idx.size):
        arr[idx[i]] = min(arr[idx[i]], val[i])


@nb.njit(parallel=True, fastmath=True, nogil=True, cache=True, error_model="numpy")
def nb_atomic_add(
    arr,
    indices,
    values,
    inplace=False,
):
    assert values.dtype == arr.dtype, "arr and values must have the same dtype"
    if not inplace:
        arr = arr.copy()
    for i in nb.prange(indices.size):
        _atomic_add(arr, indices[i], values[i])
    return arr


@nb.njit(parallel=True, fastmath=True, nogil=True, cache=True, error_model="numpy")
def nb_atomic_sub(
    arr,
    indices,
    values,
    inplace=False,
):
    assert values.dtype == arr.dtype, "Atomic sub only supports same type values"
    if not inplace:
        arr = arr.copy()
    for i in nb.prange(indices.size):
        _atomic_sub(arr, indices[i], values[i])
    return arr


@nb.njit(parallel=True, fastmath=True, nogil=True, cache=True, error_model="numpy")
def nb_atomic_and(
    arr,
    indices,
    values,
    inplace=False,
):
    assert values.dtype == arr.dtype, "Atomic and only supports same type values"
    if not inplace:
        arr = arr.copy()
    for i in nb.prange(indices.size):
        _atomic_and(arr, indices[i], values[i])
    return arr


@nb.njit(parallel=True, fastmath=True, nogil=True, cache=True, error_model="numpy")
def nb_atomic_nand(
    arr,
    indices,
    values,
    inplace=False,
):
    assert values.dtype == arr.dtype, "Atomic nand only supports same type values"
    if not inplace:
        arr = arr.copy()
    for i in nb.prange(indices.size):
        _atomic_nand(arr, indices[i], values[i])
    return arr


@nb.njit(parallel=True, fastmath=True, nogil=True, cache=True, error_model="numpy")
def nb_atomic_or(
    arr,
    indices,
    values,
    inplace=False,
):
    assert values.dtype == arr.dtype, "Atomic or only supports same type values"
    if not inplace:
        arr = arr.copy()
    for i in nb.prange(indices.size):
        _atomic_or(arr, indices[i], values[i])
    return arr


@nb.njit(parallel=True, fastmath=True, nogil=True, cache=True, error_model="numpy")
def nb_atomic_xor(
    arr,
    indices,
    values,
    inplace=False,
):
    assert values.dtype == arr.dtype, "Atomic xor only supports same type values"
    if not inplace:
        arr = arr.copy()
    for i in nb.prange(indices.size):
        _atomic_xor(arr, indices[i], values[i])
    return arr


@nb.njit(parallel=True, fastmath=True, nogil=True, cache=True, error_model="numpy")
def nb_atomic_max(
    arr,
    indices,
    values,
    inplace=False,
):
    assert values.dtype == arr.dtype, "Atomic max only supports same type values"
    if not inplace:
        arr = arr.copy()
    for i in nb.prange(indices.size):
        _atomic_max(arr, indices[i], values[i])
    return arr


@nb.njit(parallel=True, fastmath=True, nogil=True, cache=True, error_model="numpy")
def nb_atomic_min(
    arr,
    indices,
    values,
    inplace=False,
):
    assert values.dtype == arr.dtype, "Atomic min only supports same type values"
    if not inplace:
        arr = arr.copy()
    for i in nb.prange(indices.size):
        _atomic_min(arr, indices[i], values[i])
    return arr


# 使用示例
if __name__ == "__main__":
    total_len = 1000000
    arr = np.zeros(total_len, dtype=np.float64)
    arr[0] = 0
    indices = np.random.randint(total_len, size=total_len, dtype=np.int64)
    values = np.random.rand(total_len).astype(np.float64)

    analytical = np.bincount(indices, weights=values, minlength=total_len)

    result = nb_atomic_add(arr, indices, values, inplace=True)
    print("结果数组:", result)
    print(np.allclose(result, analytical))

    result = nb_atomic_sub(arr, indices, values, inplace=True)
    print("结果数组:", result)

    result = nb_atomic_and(
        arr.astype(np.int64), indices, values.astype(np.int64), inplace=True
    )
    print("结果数组:", result)

    result = nb_atomic_nand(
        arr.astype(np.int64), indices, values.astype(np.int64), inplace=True
    )
    print("结果数组:", result)

    result = nb_atomic_or(
        arr.astype(np.int64), indices, values.astype(np.int64), inplace=True
    )
    print("结果数组:", result)

    result = nb_atomic_xor(
        arr.astype(np.int64), indices, values.astype(np.int64), inplace=True
    )
    print("结果数组:", result)

    result = nb_atomic_max(arr, indices, values, inplace=True)
    print("结果数组:", result)

    result = nb_atomic_min(arr, indices, values, inplace=True)
    print("结果数组:", result)

###### Reference ######
# import numpy as np

# from numba.core.extending import intrinsic
# from numba.core import types, cgutils
# from numba import njit, prange

# from numba.np.arrayobj import make_array, basic_indexing, normalize_indices

# from llvmlite import ir

# @intrinsic
# def atomic_array_add(tyctx, arr, idx, val):
#     def codegen(context, builder, signature, args):
#         [arr, idx, val] = args
#         [arr_ty, idx_ty, val_ty] = signature.args

#         llary = make_array(arr_ty)(context, builder, arr)

#         index_types, indices = normalize_indices(
#             context, builder, [idx_ty], [idx],
#         )
#         view_data, view_shapes, view_strides = basic_indexing(
#             context, builder, arr_ty, llary, index_types, indices,
#             boundscheck=context.enable_boundscheck,
#         )
#         cgutils.printf(builder, "data=%p\n", view_data)
#         out = builder.atomic_rmw("add", view_data, val, ordering="seq_cst")
#         return out

#     resty = arr.dtype
#     sig = resty(arr, idx, val)
#     return sig, codegen


# @njit(parallel=True)
# def foo(arr):
#     accum = np.zeros(1, dtype=arr.dtype)
#     for i in prange(arr.size):
#         out = atomic_array_add(accum, 0, arr[i])
#         # print(i, out)

#     return accum[0]


# total = foo(np.arange(10, dtype=np.intp))
# print(total)

# foo.inspect_cfg(foo.signatures[0]).display(view=True)
#######################
