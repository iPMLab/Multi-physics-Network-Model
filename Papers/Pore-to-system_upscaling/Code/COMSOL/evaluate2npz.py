import numpy as np
import mph
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_1_1_300000W
from tqdm import tqdm, trange

PARAM = ComsolParams_1_1_300000W
Path_comsol = PARAM.Path_comsol
shape = PARAM.raw_shape
client = mph.start()
# create a model
model = client.load(
    r"D:\yjp\Workdir\Code\ZJU\Study\Python\heat_tranfer\3D_Study\COMSOL\big\3D_Finney_results\Finney_6000W_sweep_100000W.mph"
)

model_java = model.java
eg1 = model_java.result().evaluationGroup("eg1")
pev1 = eg1.feature("pev1")
pev1.set("innerinput", "manual")


expr_list = ("p", "T", "spf.U", "u", "v", "w")
data = np.empty((6, *shape), dtype=np.float32)
for i in tqdm(range(1, 12)):
    pev1.set("solnumindices", str(i))
    for j, expr in enumerate(expr_list):
        pev1.set("expr", [expr])
        eg1.run()
        res = np.asarray(eg1.getReal()).reshape(-1)
        u_inlet = np.round(res[0], 5)
        data[j] = res[1:].reshape(*shape)

    np.savez_compressed(f"{Path_comsol}/u{u_inlet:.5f}.npz", arr=data)
