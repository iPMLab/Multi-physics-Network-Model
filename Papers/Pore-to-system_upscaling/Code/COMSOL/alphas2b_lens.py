import numpy as np

# 设置打印精度
# np.set_printoptions(precision=6)

# 已知参数
phi = 0.365
r_ball = 5e-5

# 计算球的体积和单元体积
V_ball = (4 / 3) * np.pi * r_ball**3
V_element = V_ball / (1 - phi)

print("V_ball =", V_ball)
print("V_element =", V_element)


alpha_max = 1e-3**3 / V_element
print("alpha_max =", alpha_max)
alphas = np.array([20, 50, 100, 125,140, 152])

# 计算 b_lens
b_lens = (alphas * V_element) ** (1 / 3)

print("Calculated b_lens =", b_lens)
