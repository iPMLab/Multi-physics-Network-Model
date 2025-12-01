import numpy as np
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_1_1_100000W, extract_u_hf
import pandas as pd
from tqdm import tqdm, trange

PARAM = ComsolParams_1_1_100000W
Path_comsol = PARAM.Path_comsol
shape = PARAM.raw_shape
heat_flux = PARAM.heat_flux_in
u_inlets = PARAM.u_inlets
txts = list(Path_comsol.glob("*_u*_hf*.txt"))
for u_inlet, txt in zip(u_inlets, txts):
    file_name = str(txt.name)
    _u, _hf = extract_u_hf(file_name)

    data_i = pd.read_csv(
        txt,
        header=None,
        sep=r"\s+",
    ).to_numpy()
    data_i = np.nan_to_num(data_i.T.reshape(6, *shape), nan=-1)
    np.savez_compressed(
        f"{Path_comsol}/raw_u{float(_u):.5f}_hf{int(_hf)}.npz", arr=data_i
    )
# np.loadtxt()