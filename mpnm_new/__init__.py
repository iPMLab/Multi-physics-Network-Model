"""
This package contains the classes and functions necessary to run the mpnm algorithm.
"""

__version__ = "0.1.0"
import os as _os
from pathlib import Path as _Path


_os.environ["NUMBA_CACHE_DIR"] = str(_Path(__file__).parent.resolve() / "_numba_cache")
_os.environ["NUMBA_OPT"] = "max"
# _os.environ["NUMBA_SLP_VECTORIZE"] = "1"
_os.environ["NUMBA_ENABLE_AVX"] = "1"
_os.environ["NUMBA_FUNCTION_CACHE_SIZE"] = "1024"
_os.environ["NUMBA_THREADING_LAYER_PRIORITY"] = "tbb omp workqueue"
import numba as _nb


_nb.config.reload_config()
# nb.set_num_threads(1)

# import numpy as _np
# _np.seterr(divide="ignore", invalid="ignore")


# ===== 4. 公开 API =====
__all__ = [
    "__version__",
    "algorithm",
    "enum",
    "extraction",
    "network",
    "topotool",
    "util",
]


def __dir__():
    return __all__.copy()
