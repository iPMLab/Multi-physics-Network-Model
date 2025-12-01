from Papers.P1.Code.COMSOL.comsol_params import Paths_dict, u_inlets

Path_comsol = Paths_dict["comsol"]
import pandas as pd
import numpy as np

image_binary = (
    pd.read_csv(Path_comsol / f"Finney_6000_u{u_inlets[0]:.5f}.csv", engine="pyarrow")
    .loc[:, "P"]
    .to_numpy()
    == -1
).astype(np.uint8)
print(image_binary)
image_binary.tofile(Paths_dict["binary_raw"])
