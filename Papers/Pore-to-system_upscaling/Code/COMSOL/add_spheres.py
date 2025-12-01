import numpy as np

import mph
from tqdm import tqdm, trange
import pandas as pd
from pathlib import Path

path = Path(
    r"D:\yjp\Workdir\Code\ZJU\Study\Python\heat_tranfer\3D_Study\COMSOL\big\3D_Finney_results\block.csv"
)

path_abs = str(path)
path_parent = path.parent
filename = path.name
filename_mph = "Finney.mph"
df_xyzr = pd.read_csv(path_abs)

# zyxr = df_xyzr.to_numpy()
# scale = 5e-5
# zyxr *= scale
# np.savetxt(
#     r"D:\yjp\Workdir\Code\ZJU\Study\Python\heat_tranfer\3D_Study\Fluent\xyzr.csv",
#     zyxr,
#     delimiter=",",
# )

# client = mph.start()
# model = client.load(f"{path_parent}/{filename_mph}")
# mph.tree(model)
# for i in trange(len(df_xyzr)):
#     center = df_xyzr.loc[i, ["x", "y", "z"]].to_numpy()
#     radius = df_xyzr.loc[i, "r"]
#     Sphere_i = (model / "geometries/Geometry 1").create(
#         "Sphere", name="Sphere " + str(i + 1)
#     )
#     Sphere_i.property(name="x", value=f"{center[0]}*scale")
#     Sphere_i.property(name="y", value=f"{center[1]}*scale")
#     Sphere_i.property(name="z", value=f"{center[2]}*scale")
#     # Sphere_i.property(name='r',value=str(radius))
#     Sphere_i.property(name="r", value="r")
# model.save()
