import sys

sys.path.append("../../../")
from mpnm_new.util import remap
from mpnm_new import network as net
import pyvista as pv
import numpy as np
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from skimage import measure

C0 = "#1f77b4"
C1 = "#ff7f0e"
C2 = "#2ca02c"
# labeled_img_i = np.fromfile(
#     r"D:\yjp\Workdir\Code\ZJU\Study\Python\multi-physic-network-model\Papers\0\Data\_N10.000_sample0\pne\images\image_500_500_500_mix.raw",
#     dtype=np.int32,
# ).reshape(500, 500, 500)
# resolution = 0.001 / 500
# # grid = pv.ImageData()
# # grid.dimensions = np.array(labeled_img_i.shape) + 1  # 维度比体素多1
# # grid.spacing = (resolution, resolution, resolution)  # 体素间距
# # grid.origin = (0, 0, 0)  # 网格原点
# # grid.cell_data["values"] = labeled_img_i.flatten(order="F")


# # 创建一个绘图窗口
# plotter = pv.Plotter(off_screen=True, line_smoothing=True, polygon_smoothing=True)
# plotter.enable_parallel_projection()
# # plotter.enable_eye_dome_lighting()

# plotter.enable_anti_aliasing("ssaa")
# plotter.enable_depth_peeling(32)
# # 添加体素图像到绘图窗口


# labeled_img_i_solid = 1840 < labeled_img_i
# labeled_img_i_solid = np.pad(
#     labeled_img_i_solid, pad_width=2, mode="constant", constant_values=False
# )
# verts, faces, _, _ = measure.marching_cubes(labeled_img_i_solid > 0)
# verts *= resolution
# mesh = pv.PolyData(verts, np.insert(faces, 0, 3, axis=1))
# plotter.add_mesh(
#     mesh,
#     color=C1,
#     show_scalar_bar=False,
#     opacity=0.3,
# )
# solid_network = net.vtk2network(
#     r"D:\yjp\Workdir\Code\ZJU\Study\Python\multi-physic-network-model\Papers\0\Data\_N10.000_sample0\pne\vtps\solid_network.vtp"
# )
# # 提取并添加网络中的节点到绘图窗口
# pore_coords = solid_network["pore.coords"][:, ::-1]
# pore_radii = solid_network["pore.radius"]
# # # 提取并添加网络中的喉道到绘图窗口
# edges = solid_network["throat.conns"]
# throat_radii = solid_network["throat.radius"]
solid_pore_radius_coe = 2.5
solid_throat_radius_coe = 1
# for i, coords in enumerate(pore_coords):
#     plotter.add_mesh(
#         pv.Sphere(radius=pore_radii[i] / solid_pore_radius_coe, center=coords),
#         opacity=0.5,
#         color="white",
#     )
# edges = np.unique(np.sort(edges, axis=1), axis=0)
# for edge in edges:
#     start_point = pore_coords[edge[0]]
#     end_point = pore_coords[edge[1]]
#     # line = pv.Line(start_point, end_point)
#     tube = pv.Tube(
#         pointa=start_point,
#         pointb=end_point,
#         radius=throat_radii[i] / solid_throat_radius_coe,
#     )
#     plotter.add_mesh(tube, color="white")


# # plotter.show()
# plotter.screenshot(
#     "solid_network.png",
#     window_size=[512, 512],  # 图像尺寸（宽, 高），单位是像素
#     scale=2,  # 超采样因子（实际分辨率 = window_size * scale）
# )
# plotter.close()

# plotter = pv.Plotter(off_screen=True, line_smoothing=True, polygon_smoothing=True)
# plotter.enable_parallel_projection()
# plotter.enable_anti_aliasing("ssaa")
# # plotter.enable_eye_dome_lighting()
# plotter.enable_depth_peeling(32)
# # void
# labeled_img_i_void = (labeled_img_i <= 1840) & (labeled_img_i > 0)
# labeled_img_i_void = np.pad(
#     labeled_img_i_void, pad_width=2, mode="constant", constant_values=False
# )
# verts, faces, _, _ = measure.marching_cubes(labeled_img_i_void)
# verts *= resolution
# mesh = pv.PolyData(verts, np.insert(faces, 0, 3, axis=1))
# plotter.add_mesh(
#     mesh,
#     color=C0,
#     show_scalar_bar=False,
#     opacity=0.3,
# )
# pore_network = net.vtk2network(
#     r"D:\yjp\Workdir\Code\ZJU\Study\Python\multi-physic-network-model\Papers\0\Data\_N10.000_sample0\pne\vtps\pore_network.vtp"
# )

# # 提取并添加网络中的节点到绘图窗口
# pore_coords = pore_network["pore.coords"][:, ::-1]
# pore_radii = pore_network["pore.radius"]
# # # 提取并添加网络中的喉道到绘图窗口
# edges = pore_network["throat.conns"]
# throat_radii = pore_network["throat.radius"]
void_pore_radius_coe = 1.5
void_throat_radius_coe = 5

# for i, coords in enumerate(pore_coords):
#     plotter.add_mesh(
#         pv.Sphere(radius=pore_radii[i] / void_pore_radius_coe, center=coords),
#         opacity=0.5,
#         color="white",
#     )
# edges = np.unique(np.sort(edges, axis=1), axis=0)
# for i, edge in enumerate(edges):
#     start_point = pore_coords[edge[0]]
#     end_point = pore_coords[edge[1]]
#     # line = pv.Line(start_point, end_point)
#     tube = pv.Tube(
#         pointa=start_point,
#         pointb=end_point,
#         radius=throat_radii[i] / void_throat_radius_coe,
#     )
#     plotter.add_mesh(tube, color="white")


# plotter.screenshot(
#     "pore_network.png",
#     window_size=[512, 512],  # 图像尺寸（宽, 高），单位是像素
#     scale=2,  # 超采样因子（实际分辨率 = window_size * scale）
# )

# plotter.close()


dual_network = net.vtk2network(
    r"D:\yjp\Workdir\Code\ZJU\Study\Python\multi-physic-network-model\Papers\0\Data\_N10.000_sample0\pne\vtps\dual_network.vtp"
)


plotter = pv.Plotter(off_screen=True, line_smoothing=True, polygon_smoothing=True)
plotter.enable_parallel_projection()
# plotter.enable_eye_dome_lighting()

plotter.enable_anti_aliasing("ssaa")
plotter.enable_depth_peeling(32)
# 添加体素图像到绘图窗口

# 提取并添加网络中的节点到绘图窗口
pore_coords = dual_network["pore.coords"][:, ::-1]
pore_radii = dual_network["pore.radius"]
# # 提取并添加网络中的喉道到绘图窗口
edges = dual_network["throat.conns"]
edges = np.unique(np.sort(edges, axis=1), axis=0)
throat_radii = dual_network["throat.radius"]
solid_pore_radius_coe = 2.5
solid_throat_radius_coe = 1


nodes_solid = dual_network["pore._id"][dual_network["pore.solid"]]
nodes_void = dual_network["pore._id"][dual_network["pore.void"]]

pore_coords_solid = pore_coords[nodes_solid]
pore_radii_solid = pore_radii[nodes_solid]
pore_coords_void = pore_coords[nodes_void]
pore_radii_void = pore_radii[nodes_void]

throat_radii_solid = throat_radii[dual_network["throat.solid"]]
throat_radii_void = throat_radii[dual_network["throat.void"]]
throat_radii_conect = throat_radii[dual_network["throat.connect"]]
sphere_geom = pv.Sphere()  # 降低分辨率加速
# ==============================
# 1. 批量绘制 solid pores（球体）
# ==============================
if len(pore_coords_solid) > 0:
    coords_s = np.asarray(pore_coords_solid)
    radii_s = np.asarray(pore_radii_solid)
    cloud_s = pv.PolyData(coords_s)
    cloud_s["radius"] = radii_s

    spheres_s = cloud_s.glyph(scale="radius", geom=sphere_geom, orient=False)
    plotter.add_mesh(spheres_s, color=C1, opacity=1)

# ==============================
# 2. 批量绘制 void pores（球体）
# ==============================
if len(pore_coords_void) > 0:
    coords_v = np.asarray(pore_coords_void)
    radii_v = np.asarray(pore_radii_void)
    cloud_v = pv.PolyData(coords_v)
    cloud_v["radius"] = radii_v
    spheres_v = cloud_v.glyph(scale="radius", geom=sphere_geom, orient=False)
    plotter.add_mesh(spheres_v, color=C0, opacity=1)


# ==============================
# 3. 批量绘制 solid throats（圆柱）
# ==============================
def create_tubes(coords, edges, radii, radius_coe, n_sides=20):
    """高效创建并合并多个 tube"""
    tubes = []
    for i in range(len(radii)):
        p1 = coords[edges[i, 0]]
        p2 = coords[edges[i, 1]]
        line = pv.Line(p1, p2)
        tube = line.tube(radius=radii[i] / radius_coe, n_sides=n_sides)
        tubes.append(tube)
    if not tubes:
        return None
    return pv.MultiBlock(tubes).combine()


# Solid throats
if len(throat_radii_solid) > 0:
    tubes_s = create_tubes(
        pore_coords, edges, throat_radii_solid, solid_throat_radius_coe * 7
    )
    if tubes_s is not None:
        plotter.add_mesh(tubes_s, color="white")

# Void throats
if len(throat_radii_void) > 0:
    tubes_v = create_tubes(
        pore_coords, edges, throat_radii_void, void_throat_radius_coe
    )
    if tubes_v is not None:
        plotter.add_mesh(tubes_v, color="white")

# Connecting throats (使用固定缩放系数 20)
if len(throat_radii_conect) > 0:
    # 注意：这里假设 throat_radii_conect 对应 edges 的前 len(throat_radii_conect) 条边
    # 如果 edges 长度不一致，请确保索引对齐
    tubes_c = []
    for i in range(len(throat_radii_conect)):
        p1 = pore_coords[edges[i, 0]]
        p2 = pore_coords[edges[i, 1]]
        line = pv.Line(p1, p2)
        tube = line.tube(radius=throat_radii_conect[i] / 10)
        tubes_c.append(tube)
    if tubes_c:
        merged_c = pv.MultiBlock(tubes_c).combine()
        plotter.add_mesh(merged_c, color="white")


# plotter.show()
plotter.screenshot(
    "dual_network.png",
    window_size=[512, 512],  # 图像尺寸（宽, 高），单位是像素
    scale=2,  # 超采样因子（实际分辨率 = window_size * scale）
)
plotter.close()
