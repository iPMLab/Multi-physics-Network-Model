import functools
import inspect

import numpy as np
from ..util import ravel


def inspect_args(func=None, check_mpn=True, check_ids=True, update_inner_info=False):
    # 如果 `func` 是 None，说明装饰器带参数
    if func is None:
        return lambda f: inspect_args(f, check_mpn, check_ids, update_inner_info)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        params = sig.parameters
        # 处理 'mpn' 参数
        if check_mpn:
            if "mpn" in params:
                mpn = bound_args.arguments.get("mpn")
                # if not mpn["throat.conns"].flags["F_CONTIGUOUS"]:
                #     mpn["throat.conns"] = np.asfortranarray(mpn["throat.conns"])
                #     print(
                #         "Warning: 'throat.conns' array is not Fortran contiguous. Converted to Fortran contiguous."
                #     )
                #     print(
                #         "To avoid this warning, ensure 'throat.conns' is Fortran contiguous before passing to the function."
                #     )
                if "pore.void" not in mpn:
                    mpn["pore.void"] = ~mpn["pore.all"].copy()
                if "pore.solid" not in mpn:
                    mpn["pore.solid"] = ~mpn["pore.all"].copy()
        # 处理 'ids' 参数（仅在 check_id=True 时）
        if check_ids:
            if "ids" in params:
                ids = bound_args.arguments.get("ids")
                bound_args.arguments["ids"] = ravel(ids)
        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


def multi_deco(*decs, is_reversed=False):
    def deco(f):
        if is_reversed:
            for dec in reversed(decs):
                f = dec(f)
        else:
            for dec in decs:
                f = dec(f)
        return f

    return deco
