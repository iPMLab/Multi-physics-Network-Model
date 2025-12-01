from Papers.P1.Code.COMSOL.comsol_params import *
import matplotlib.pyplot as plt


PARAMS_list = [
    # PARAMS_N2_000,
    # PARAMS_N3_455,
    # PARAMS_N4_353,
    # # !PARAMS_N4_689,
    # PARAMS_N4_869,
    PARAMS_N5_000,
]
porosities_list = []

for PARAMS in PARAMS_list:
    porosities = []
    for PARAM in PARAMS:
        image = np.fromfile(PARAM.Path_binary_raw, dtype=np.uint8).reshape(
            PARAM.raw_shape
        )
        porosity = np.count_nonzero(image == 0) / np.prod(PARAM.raw_shape)
        print(f"{PARAM.prefix}: {porosity}")
        porosities.append(porosity)
    porosities_list.append(porosities)

means = []
stds = []
ptps = []
mins = []
maxs = []
medians = []
for porosities in porosities_list:
    mean = np.mean(porosities)
    std = np.std(porosities)
    means.append(mean)
    medians.append(np.median(porosities))
    stds.append(std)
    ptps.append(np.ptp(porosities))
    mins.append(np.min(porosities))
    maxs.append(np.max(porosities))

labels = [
    # "N=2.545",
    # "N=3.455",
    # "N=4.353",
    # # !"N=4.689",
    # "N=4.869",
    "N=5.000",
]
print(f"Medians: {medians}")
print(f"Means: {means}")
print(f"Stds: {stds}")
print(f"Ptp: {ptps}")
print(f"Mins: {mins}")
print(f"Maxs: {maxs}")
plt.figure(figsize=(8, 6))
plt.boxplot(
    porosities_list,
    tick_labels=labels,
    patch_artist=True,
    boxprops={"facecolor": "skyblue"},
    medianprops={"color": "red"},
)
plt.title("Porosity Distribution", fontsize=14)
plt.ylabel("Porosity")
plt.grid(axis="y", alpha=0.3)
plt.show()
