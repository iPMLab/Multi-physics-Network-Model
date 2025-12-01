import numpy as np
import numba as nb
import time
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:36:28 2023
Modified for 3D Interpolation

@author: DU
"""

import numpy as np
from scipy.spatial import Delaunay

def interp_weights_3d(x, y, z, tx, ty, tz, d=3):
    """
    三维插值权重计算
    
    参数说明
    ----------
    x, y, z : array_like
        原始三维坐标点 (N个点)
    tx, ty, tz : array_like
        目标网格坐标 (M个点)
    d : int, optional
        空间维度，三维默认为3

    返回
    -------
    vertices : 包含目标点所在四面体顶点的索引数组
    weights : 对应的插值权重矩阵
    """
    def _flatten_3d(data):
        if data.ndim > 1:
            data = data.ravel()
        return data
    
    # 数据展平处理
    x, y, z = map(_flatten_3d, [x, y, z])
    tx, ty, tz = map(_flatten_3d, [tx, ty, tz])
    
    # 构建三维Delaunay三角剖分
    source_points = np.column_stack((x, y, z))
    target_points = np.column_stack((tx, ty, tz))
    t0 = time.time()
    tri = Delaunay(source_points)
    print(f"Delaunay triangulation took {time.time() - t0} seconds.")
    # 查找目标点所在的四面体
    t0 = time.time()
    simplex = tri.find_simplex(target_points)
    print(f"Finding simplex took {time.time() - t0} seconds.")
    # 获取四面体顶点索引
    vertices = np.take(tri.simplices, simplex, axis=0)
    
    # 计算重心坐标
    transform = np.take(tri.transform, simplex, axis=0)
    delta = target_points - transform[:, d]
    bary_coords = np.einsum('njk,nk->nj', transform[:, :d, :], delta)
    
    # 添加最后一个权重分量
    weights = np.hstack((bary_coords, 1 - bary_coords.sum(axis=1, keepdims=True)))
    
    return vertices, weights

# def interpolate_3d(values, vertices, weights, target_shape=None):
#     """
#     三维插值执行函数
    
#     参数说明
#     ----------
#     values : array_like
#         原始数据值，形状需与(x,y,z)一致
#     vertices : 来自interp_weights_3d的输出
#     weights : 来自interp_weights_3d的输出
#     target_shape : 目标数据形状
    
#     返回
#     -------
#     插值结果数组
#     """
#     # 数据展平处理
#     if values.ndim == 3:
#         values = values.ravel()
    
#     # 执行插值计算
#     interp_values = np.einsum('nj,nj->n', np.take(values, vertices), weights)
    
#     # 结果重塑
#     if target_shape is not None:
#         return interp_values.reshape(target_shape)
#     else:
#         return interp_values
    
@nb.njit(fastmath=True, parallel=True,cache=True,nogil=True)
def numba_interp(values, vertices, weights):
    """Numba加速插值计算"""
    values = values.ravel()
    n = vertices.shape[0]
    result = np.empty(n,np.float64)
    for i in nb.prange(n):
        vtx = vertices[i]
        w = weights[i]
        result[i] = np.dot(values[vtx], w)
    return result
    
def interpolate_3d(values, vertices, weights,shape=None):
    res = numba_interp(values.ravel(), vertices, weights)
    if shape:
        res = res.reshape(shape)
    else:
        pass
    return res

# 使用示例 ---------------------------------------------------------------------
if __name__ == "__main__":
    # 生成测试数据
    # 原始数据 (10x10x10 立方体)
    x_orig = np.linspace(0, 9, 500)
    y_orig = np.linspace(0, 9, 500)
    z_orig = np.linspace(0, 9, 500)
    X_orig, Y_orig, Z_orig = np.meshgrid(x_orig, y_orig, z_orig, indexing='ij')
    values_orig = np.sin(X_orig) + np.cos(Y_orig) + np.tan(Z_orig*0.1)

    # 目标网格 (20x20x20 加密网格)
    x_tgt = np.linspace(0, 9, 1000)
    y_tgt = np.linspace(0, 9, 1000)
    z_tgt = np.linspace(0, 9, 1000)
    X_tgt, Y_tgt, Z_tgt = np.meshgrid(x_tgt, y_tgt, z_tgt, indexing='ij')

    # 计算插值权重
    t0 = time.time()
    vtx, wts = interp_weights_3d(X_orig.ravel(), Y_orig.ravel(), Z_orig.ravel(),
                                 X_tgt.ravel(), Y_tgt.ravel(), Z_tgt.ravel())
    print(f'interp_weights_3d time: {time.time()-t0:.2f}s')

    t0 = time.time()
    # 执行插值
    interp_result = interpolate_3d(values_orig,vtx,wts,X_tgt.shape)
    print(f'interpolate_3d time: {time.time()-t0:.2f}s')

    print(interp_result[0,0,2])
    print(values_orig[0,0,1])
    import matplotlib.pyplot as plt
    plt.imshow(interp_result[0,:,:])
    plt.show()
    plt.imshow(values_orig[0,:,:])
    plt.show()