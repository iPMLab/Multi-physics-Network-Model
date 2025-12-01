import numpy as np

np.set_printoptions(precision=3)
phi = 0.365
r_ball = 5e-5

b_lens = np.array(
    [0.00025454, 0.00034546, 0.00043525, 0.00046886, 0.00048691, 0.0005, 0.001]
)

print("Ns = ", b_lens / (2 * r_ball))
