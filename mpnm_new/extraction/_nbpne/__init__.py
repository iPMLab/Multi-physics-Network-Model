import os

os.environ["NUMBA_OPT"] = "max"
# os.environ["NUMBA_SLP_VECTORIZE"] = "1"
os.environ["NUMBA_ENABLE_AVX"] = "1"
import numba as nb

nb.config.reload_config()
from .extract_util import extract

__all__ = [
    "extract",
]
