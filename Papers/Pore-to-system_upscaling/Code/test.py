import pandas as pd
from sklearn.metrics import r2_score
df = pd.read_csv("plot_pore_N10.000.csv")
print(df)
P_comsol_original = df["P_comsol_original"].values
P_dualn_original = df["P_dualn_original"].values
P_comsol_optimized = df["P_comsol_optimized"].values
P_dualn_optimized = df["P_dualn_optimized"].values

r2_pressure_original = r2_score(P_comsol_original, P_dualn_original)
r2_pressure_optimized = r2_score(P_comsol_optimized, P_dualn_optimized)
print(r2_pressure_original)
print(r2_pressure_optimized)
