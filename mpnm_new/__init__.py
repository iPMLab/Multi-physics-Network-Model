"""
This package contains the classes and functions necessary to run the mpnm algorithm.
"""

__version__ = "0.1.0"
import os
import numpy as _np

os.environ["NUMBA_OPT"] = "max"
os.environ["NUMBA_SLP_VECTORIZE"] = "1"
os.environ["NUMBA_ENABLE_AVX"] = "1"
os.environ["NUMBA_FUNCTION_CACHE_SIZE"] = "1024"
os.environ["NUMBA_THREADING_LAYER_PRIORITY"] = "tbb omp workqueue"
import numba as _nb

_nb.config.reload_config()
# nb.set_num_threads(1)


_np.seterr(divide="ignore", invalid="ignore")

# top-level module anymore.
__all__ = [
    "__version__",
    "algorithm",
    "enum",
    "network",
    "topotool",
    "util",
    "extraction",
]


def __dir__():
    return __all__.copy()
