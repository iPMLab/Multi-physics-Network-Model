import numpy as np

N = np.array((2.5, 3.455, 4.353, 4.869, 5.0, 10.000))
N_max = 10.000
# N = radius/block length when N = 10,Vol = 1/1
scale = (N / N_max) ** 3
Vol = np.round(1 / scale)
print(Vol)
