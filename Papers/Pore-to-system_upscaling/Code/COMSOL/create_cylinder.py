import numpy as np
import pyvista as pv
from sympy.physics.mechanics.functions import gravity

# # 原始四面体的方向（未归一化）
# directions = np.array([
#     [1, 1, 1],
#     [-1, -1, 1],
#     [-1, 1, -1],
#     [1, -1, -1]
# ])/np.sqrt(3)

# 原点

distance = 1.0  # 延伸长度
sqrt_2 = np.sqrt(2)
sqrt_6 = np.sqrt(6)
point = np.array([0, 0, 0])
directions = (
    np.array(
        [
            [0.0, 0.0, 1.0],
            [2 * sqrt_2 / 3, 0.0, -1 / 3],  # y=0
            [-sqrt_2 / 3, sqrt_6 / 3, -1 / 3],
            [-sqrt_2 / 3, -sqrt_6 / 3, -1 / 3],
        ]
    )
    * distance
)


side_length = directions[2, 1] - directions[3, 1]
print(side_length)

##### pos  #####
delta_z = directions[0, 2] - directions[1, 2]
delta_y = directions[2, 1] - directions[3, 1]
delta_x = directions[1, 0] - directions[2, 0]
delta_gravity_center = side_length / 6 * np.sqrt(3)
print(delta_z, delta_y, delta_x)
print(np.linalg.norm(directions - np.array([0, 0, 0]), axis=1))


# 延伸后的点
all_points = point + directions
all_points = np.vstack((point, all_points))

print(all_points)

# 绘制点
pv.PolyData(all_points).plot(show_edges=True)
max_z = 10
max_y = 10
max_x = 10
I = np.arange(round(max_x / delta_x))
J = np.arange(round(max_y / delta_y))
K = np.arange(round(max_z / delta_z))


I, J, K = np.meshgrid(I, J, K, indexing="ij")  # 网格化
# print(np.column_stack((I.ravel(), J.ravel(), K.ravel())))
# 计算中心点坐标
x = I * delta_x + ((K % 3) - 1) * delta_gravity_center
y = J * delta_y + ((K % 3) - 1) * (delta_y / 2) + (I % 2) * (delta_y / 2)
z = K * delta_z

# 展平并合并坐标
centers = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

# 过滤超出范围的点（确保 x <= 100 和 y <= 100）
centers = centers[
    (centers[:, 0] <= max_x) & (centers[:, 1] <= max_y) & (centers[:, 2] < max_z)
]
# import matplotlib.pyplot as plt
# plt.scatter(centers[:, 0], centers[:, 1], s=1)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('无缝平铺的三角形中心点')
# plt.grid(True)
# plt.axis('equal')
# plt.show()
print(len(centers))

# print(centers)


import mph
from tqdm import trange, tqdm

client = mph.start()
model = client.load(r"C:\Users\yjp\Desktop\cylinder_test.mph")

# centers = centers[:20]
for i, center in tqdm(enumerate(centers), total=len(centers)):
    for j, dir_ in enumerate(directions):
        # dir_ = dir_ + center
        Cylinder_i = (model / "geometries/Geometry 1").create(
            "Cylinder", name=f"Cylinder_{i}_{j}"
        )

        Cylinder_i.property(name="pos", value=f"{center[0]} {center[1]} {center[2]}")
        Cylinder_i.property(name="axistype", value="cartesian")
        Cylinder_i.property(name="r", value="radius")
        Cylinder_i.property(name="h", value="height")
        Cylinder_i.property(name="axis", value=f"{dir_[0]} {dir_[1]} {dir_[2]}")

model.save()
