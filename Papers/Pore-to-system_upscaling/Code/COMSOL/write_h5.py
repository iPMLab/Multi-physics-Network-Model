import numpy as np
import mph
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_1_1, extract_u_hf
from tqdm import tqdm, trange
import h5py

PARAM = ComsolParams_1_1
Path_comsol_data = PARAM.Path_comsol
data_all = {}
comsol_data_list = list(Path_comsol_data.glob(r"*_u*_hf*.npz"))
with h5py.File(Path_comsol_data / "data_all.h5", mode="w", swmr=True) as f:
    # f.swmr_mode = True
    for comsol_data in tqdm(comsol_data_list):
        data = np.load(comsol_data)["arr"]
        # print(data)
        _u, _hf = extract_u_hf(comsol_data.name)
        _u = float(_u)
        _hf = int(_hf)
        print(_u, _hf)
        f.create_dataset(f"_u{_u:.5f}_hf{_hf}", data=data,compression="gzip" )
