import numpy as np
import pandas as pd

pd.set_option("display.float_format", lambda x: "%.5e" % x)
np.random.seed(0)
points_1_8 = np.random.randint(0, 499, size=(200, 3)) * 1e-6
points_1_8 = pd.DataFrame(points_1_8, columns=["x", "y", "z"])
points_1_8.insert(0, "id_o", np.arange(len(points_1_8)))
points_1_8 = points_1_8.drop(points_1_8.index[[3, 4, 5]]).reset_index(drop=True)
print(points_1_8.iloc[:50,:])

points_1_16 = np.random.randint(0, 249, size=(200, 3)) * 1e-6
points_1_16 = pd.DataFrame(points_1_16, columns=["x", "y", "z"])
points_1_16.insert(0, "id_o", np.arange(len(points_1_16)))
# points_1_16 = points_1_16.drop(points_1_16.index[[3,4]]).reset_index(drop=True)
# print(points_1_16.iloc[:50, :])
