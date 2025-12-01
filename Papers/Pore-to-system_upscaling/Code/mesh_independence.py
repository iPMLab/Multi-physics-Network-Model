import numpy as np
from Common_Vars import _Path_fig, plt

# 表格数据
data = {
    "Re": [0.001, 0.005, 0.02, 0.1, 1],
    # "Kozeny-Carman": [0.152, 0.761, 3.044, 15.207, 152.070],
    r"$2.24 \times 10^{16} \, \text{Cells/m}^3$": [
        0.178,
        0.889,
        3.55,
        17.775,
        177.89,
    ],
    r"$5.36 \times 10^{16} \, \text{Cells/m}^3$": [
        0.163,
        0.817,
        3.270,
        16.347,
        163.520,
    ],
    r"$8.58 \times 10^{16} \, \text{Cells/m}^3$": [
        0.165,
        0.824,
        3.294,
        16.468,
        164.723,
    ],
}

# 创建画布
plt.figure(figsize=(13 / 2, 4.8))

# 折线图
# plt.subplot(1, 2, 1)
for col in list(data.keys())[1:]:
    plt.plot(data["Re"], data[col], marker="o", label=col)
plt.xscale("log")
# plt.yscale('log')
plt.xlabel("Re")
plt.ylabel("Pressure drop (Pa)")
# plt.title('Mesh independence validation')
plt.legend(frameon=False)
# plt.grid(True, which="both", ls="--")
plt.grid(False)
plt.tight_layout()
plt.savefig(_Path_fig / "mesh_independence.png", dpi=330)
plt.show()
