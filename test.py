import time

import numpy as np
import pandas as pd
import numba as nb
from scipy.sparse import csr_matrix



pn={'throat.conns':np.ones((2000000000,2))}
t0=time.time()
rows = np.empty(shape=len(pn['throat.conns']) * 2, dtype=int)
# cols = np.empty(shape=len(pn['throat.conns']) * 2, dtype=int)
rows[:len(pn['throat.conns'])] = pn['throat.conns'][:, 1]
rows[len(pn['throat.conns']):] = pn['throat.conns'][:, 1]
print(time.time()-t0)

t0=time.time()
z=rows.repeat(2)
print(time.time()-t0)