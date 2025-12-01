from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_1_8_2 as PARAM
import h5py

PARAM = PARAM()
u_inlets = PARAM.u_inlets
num_params = PARAM.num_comsol_params
Path_comsol = PARAM.Path_comsol
prefix = PARAM.prefix
shape = PARAM.raw_shape

heat_flux = 50000


def write_h5_data(single_file_mode=True):
    """
    Write COMSOL data to HDF5 file, with option for single or multiple input txt files.

    Parameters:
    - single_file_mode: bool (default=True)
        If True, reads from a single combined txt file (original write_h5_single_txt behavior)
        If False, reads from multiple individual txt files (original write_h5_multi_txt behavior)
    """
    with h5py.File(PARAM.Path_data_h5, "w") as hf:
        for i, u_inlet in tqdm(enumerate(u_inlets)):
            if single_file_mode:
                # Single file mode - load from combined file and slice
                Path_txt = Path(f"{Path_comsol}/{prefix}_{heat_flux}W.txt")
                if i == 0:  # Only load the full data once
                    data = pd.read_csv(
                        Path_txt, engine="pyarrow", sep=",", header=None
                    ).to_numpy()
                data_i = data[:, num_params * i : num_params * (i + 1)]
            else:
                # Multiple files mode - load individual files
                txt_name_i = f"{prefix}_u{float(u_inlet):.5f}_hf{heat_flux}W.txt"
                Path_txt_i = Path(f"{Path_comsol}/{txt_name_i}")
                data_i = pd.read_csv(
                    Path_txt_i, engine="pyarrow", sep=",", header=None
                ).to_numpy()

            # Common processing for both modes
            data_i = data_i.T.reshape(num_params, *shape)
            dataset_name = f"_u{float(u_inlet):.5f}_hf{int(heat_flux)}"
            hf.create_dataset(
                dataset_name,
                data=data_i,
                compression="gzip",
            )


single = bool(0)
write_h5_data(single_file_mode=single)
