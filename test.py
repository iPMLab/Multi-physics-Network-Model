import numpy as np
import pandas as pd
import numba as nb

@nb.njit(parallel=True, fastmath=True)
def a_(a):
    if a==1:
        return 1
    else:
        return None

print(a_(1))