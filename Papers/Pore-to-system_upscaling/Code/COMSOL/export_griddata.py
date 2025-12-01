import time
import mph
from pathlib import Path
import numpy as np
import pyvista as pv
from tqdm import tqdm
import h5py
import pandas as pd
from Papers.P1.Code.COMSOL.comsol_params import *

import sys

sys.path.append("../../")
from mpnm_new.util import unique_rows, find_throat_conns_map


def export(client, PARAM):
    PARAM = PARAM()
    shape = PARAM.raw_shape
    resolution = PARAM.resolution
    Path_h5 = PARAM.Path_data_h5
    Path_comsol = PARAM.Path_comsol
    Path_binary_raw = PARAM.Path_binary_raw
    print(PARAM.prefix)
    father = "_".join(PARAM.prefix.split("_")[:2])
    sample = re.search(r"(\d+)$", PARAM.prefix).group(1)

    model = client.load(
        rf"D:\yjp\Workdir\Code\ZJU\Study\Python\heat_tranfer\3D_Finney_results\{father}\Finney_{sample}_hf10000W_lowRe.mph"
    )

    hf = 10000
    params = PARAM.comsol_params
    num_params = len(params)
    prefix = "griddata"

    model_java = model.java
    export_tags = list(model_java.result().export().tags())
    result_node = model_java.result()

    # if "datatemp" in export_tags:
    # T_node.set("fullprec", "off")

    Res = PARAM.Res
    Path_comsol.mkdir(exist_ok=True, parents=True)
    print("Evaluating value...")
    for i, Re in tqdm(enumerate(Res)):
        stem = f"_Re{Re:.5f}_hf{hf}"

        ##### griddata export ######
        result_node.export().remove("GD")
        result_node.export().create("GD", "Data")
        GD_node = result_node.export("GD")
        GD_node.set("exporttype", "text")
        GD_node.set("location", "regulargrid")
        GD_node.set("expr", [f"gpeval(2,{param})" for param in params])
        GD_node.set("innerinput", "manual")
        GD_node.set("solnum", [str(i + 1)])
        GD_node.set("header", "off")
        GD_node.set("includecoords", "off")
        GD_node.set("includenan", "on")
        GD_node.set("recover", "ppr")
        GD_node.set("separator", ",")
        GD_node.set("regulargridx3", f"{shape[2]}")
        GD_node.set("regulargridy3", f"{shape[1]}")
        GD_node.set("regulargridz3", f"{shape[0]}")

        GD_node.set("filename", str((Path_comsol / f"{prefix}{stem}.txt").resolve()))
        GD_node.run()

    print("Creating h5 dataset...")
    Path_h5.parent.mkdir(exist_ok=True, parents=True)

    Path_h5 = Path(rf"C:\Users\yjp\Desktop\新建文件夹\griddata{PARAM.prefix}.h5")
    with h5py.File(Path_h5, "a") as f:
        # group_i = f.require_group(f"{stem}")
        for i, Re in tqdm(enumerate(Res)):
            stem = f"_Re{Re:.5f}_hf{hf}"
            point_data_T = pd.read_csv(
                Path_comsol / f"{prefix}{stem}.txt",
                sep=",",
                header=None,
                engine="pyarrow",
            ).to_numpy(dtype=np.float32)
            point_data_T = point_data_T.T.reshape((num_params, *shape))
            f.create_dataset(
                name=stem,
                data=point_data_T,
                compression="gzip",
            )
    client.clear()

    print("done")


client = mph.start()
PARAMS = PARAMS_N4_353

t0 = time.time()
for PARAM in PARAMS:
    export(client, PARAM)
t1 = time.time()
print(f"Time cost: {t1 - t0:.2f} s")
# mesh = pv.UnstructuredGrid(
#     np.insert(cells_T_s, 0, 4, axis=1).reshape(-1),
#     np.full(cells_T_s.shape[0], pv.CellType.TETRA),
#     points_T_s,
# )

# mesh.point_data["point.cfluxx"] = point_data_T[:, T_par_map["ht.dfluxx"]]
# mesh.plot()


# 77.97
# 235.28
# 570.63
# 920
# 1023.82
