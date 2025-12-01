import numpy as np

np.set_printoptions(precision=3)
phi = 0.365
r_ball = 5e-5

V_ball = 4 / 3 * np.pi * r_ball**3

V_element = V_ball / (1 - phi)

print("V_ball = ", V_ball)
print("V_element = ", V_element)


b_len = 1e-3


b_lens = np.array([b_len, b_len / 2, b_len / 3, b_len / 4])
Vs = b_lens**3
alphas = Vs / V_element
print("alphas = ", alphas)
