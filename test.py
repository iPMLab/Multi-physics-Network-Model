import numpy as np
import pandas as pd

data=np.array([[1,2,3],
              [4,5,6]])
a=pd.DataFrame(columns=['a','b','c'],data=data)
c=pd.concat((a,a),axis=1)
print(c)
z=c.loc[:,'a']
print(z)