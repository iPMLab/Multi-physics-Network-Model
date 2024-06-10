import numpy as np
import pandas as pd
import numba as nb
from scipy.sparse import csr_matrix

a=np.zeros((20,2))
a[19,0]=1
a_unique,index=np.unique(a,return_index=True)
print(np.where(a==0))