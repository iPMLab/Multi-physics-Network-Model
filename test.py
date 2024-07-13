import fastremap
import numpy as np
from scipy.spatial import KDTree

# a=np.random.randn(200000,2)
#
# tree=KDTree(a)
# distances,indice=tree.query(a,k=2)
# print(distances[:,1])



import numpy as np
a = np.array([[100, 10, 11], [3, 2, 12]])
b=a.ravel()
b[0]=-1
print(a[np.unravel_index(5, a.shape)])
print(a)


