import fastremap
import numpy as np
from scipy.spatial import KDTree

# a=np.random.randn(200000,2)
#
# tree=KDTree(a)
# distances,indice=tree.query(a,k=2)
# print(distances[:,1])



import numpy as np
index=np.array([[1,2,3],[4,5,6]])
value=np.array([[7,8,9],[10,11,12]])
print(np.column_stack((index.flatten(), value.flatten())))

print(np.where(index.flatten()==1))
print(type(index))