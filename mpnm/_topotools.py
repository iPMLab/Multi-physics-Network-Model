#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:55:27 2022

@author: htmt
"""
import numpy as np
# import openpnm as op
from mpnm._Base import *
import copy
from scipy.sparse import coo_matrix
from mpnm.tools_numba import *
import scipy.spatial as spt
import numba as nb
import os

num_threads = os.environ.get('num_threads')
if num_threads == None:
    pass
else:
    nb.set_num_threads(int(num_threads))

if 'update_inner_info' not in os.environ.keys():
    os.environ['update_inner_info'] = 'False'


def check_input(**kwarg):
    if 'pn' in kwarg:
        if 'pore.solid' not in kwarg['pn']:
            kwarg['pn']['pore.solid'] = np.zeros_like(kwarg['pn']['pore.void'])
        else:
            pass
        if 'inner_info' not in kwarg['pn']:
            kwarg['pn'].update(topotools().update_inner_info(kwarg['pn']))
        else:
            pass
        if 'inner_start2end' not in kwarg['pn']:
            kwarg['pn'].update(topotools().update_inner_info(kwarg['pn']))
        else:
            pass
        if os.environ['update_inner_info'] == 'True':
            kwarg['pn'].update(topotools().update_inner_info(kwarg['pn']))
        if 'ids' in kwarg:
            if isinstance(kwarg['ids'], np.ndarray):
                ids = kwarg['ids']
            elif type(kwarg[
                          'ids']) == int or float or np.int64 or np.int16 or np.int8 or np.float32 or np.float64 or np.float16 or np.float8:
                ids = np.array([kwarg['ids']])
            elif type(kwarg['ids']) == list:
                ids = np.array(kwarg['ids'])
            else:
                raise Exception('index type error')
            return ids


class topotools(Base):

    @staticmethod
    def find_surface_KDTree(pn, status='x', imsize=0, resolution=0, label_1='left', label_2='right'):
        # t1 = time.time()
        workers = -1  # 使用所有线程
        k = 1  # 寻找的邻点数
        distance_factor = 1.2  # 深度
        distance_factor2 = 0  # 平面外移
        id_coord = np.concatenate((np.array([pn['pore._id']]).T, pn['pore.coords'],
                                   np.array([pn['pore.radius']]).T), axis=1)
        coords = id_coord[:, 1:4]
        length_temp = np.percentile(np.sort(pn['throat.length']), 0.05)
        length_min = length_temp / 4
        # coords[:, 0] = (coords[:, 0] - np.min(coords[:, 0])) / (np.max(coords[:, 0]) - np.min(coords[:, 0]))
        # coords[:, 1] = (coords[:, 1] - np.min(coords[:, 1])) / (np.max(coords[:, 1]) - np.min(coords[:, 1]))
        # coords[:, 2] = (coords[:, 2] - np.min(coords[:, 2])) / (np.max(coords[:, 2]) - np.min(coords[:, 2]))
        # kt = spt.KDTree(data=coords, leafsize=10)  # 用于快速查找的KDTree类
        ckt = spt.cKDTree(coords)  # 用C写的查找类，执行速度更快
        x_min = np.min(coords[:, 0])
        x_max = np.max(coords[:, 0])
        y_min = np.min(coords[:, 1])
        y_max = np.max(coords[:, 1])
        z_min = np.min(coords[:, 2])
        z_max = np.max(coords[:, 2])

        x_block = int(np.ceil((x_max - x_min) / length_min))
        y_block = int(np.ceil((y_max - y_min) / length_min))
        z_block = int(np.ceil((z_max - z_min) / length_min))
        # x_block=1000
        # y_block=1000
        # z_block=1000

        if status == 'x':
            diff = x_max - x_min
            m1 = np.linspace(y_min, y_max, y_block)
            m2 = np.linspace(z_min, z_max, z_block)
            m1, m2 = np.meshgrid(m1, m2)
            m1 = m1.flatten()
            m2 = m2.flatten()
            m3 = np.full((len(m1)), x_min - distance_factor2 * diff)
            m4 = np.full((len(m1)), x_max + distance_factor2 * diff)
            # print('num_nodes', len(m1))

            mesh1 = np.vstack((m3, m1, m2)).transpose()
            mesh2 = np.vstack((m4, m1, m2)).transpose()
            distance1, index1 = ckt.query(mesh1, workers=workers, k=k)  # 返回最近邻点的距离d和在数组中的顺序x
            index1 = index1.flatten()
            distance1 = distance1.flatten()
            index1 = np.unique(index1[distance1 < distance_factor * np.mean(distance1)])
            distance2, index2 = ckt.query(mesh2, workers=workers, k=k)
            index2 = index2.flatten()
            distance2 = distance2.flatten()
            index2 = np.unique(index2[distance2 < distance_factor * np.mean(distance2)])

            pore_number1 = np.full((id_coord.shape[0]), 0)
            pore_number2 = np.full((id_coord.shape[0]), 0)
            name_label1 = 'pore.' + label_1
            name_label2 = 'pore.' + label_2
            pore_number1[index1] = 1
            pn[name_label1] = pore_number1
            pore_number2[index2] = 1
            pn[name_label2] = pore_number2
            pn[name_label1], pn[name_label2] = pn[name_label1].astype(bool), pn[name_label2].astype(
                bool)
        if status == 'y':
            diff = y_max - y_min
            m1 = np.linspace(x_min, x_max, x_block)
            m2 = np.linspace(z_min, z_max, z_block)
            m1, m2 = np.meshgrid(m1, m2)
            m1 = m1.flatten()
            m2 = m2.flatten()
            m3 = np.full((len(m1)), y_min - distance_factor2 * diff)
            m4 = np.full((len(m1)), y_max + distance_factor2 * diff)
            # print('num_nodes', len(m1))

            mesh1 = np.vstack((m1, m3, m2)).transpose()
            mesh2 = np.vstack((m1, m4, m2)).transpose()
            distance1, index1 = ckt.query(mesh1, workers=workers, k=k)  # 返回最近邻点的距离d和在数组中的顺序x
            index1 = index1.flatten()
            distance1 = distance1.flatten()
            index1 = np.unique(index1[distance1 < distance_factor * np.mean(distance1)])
            distance2, index2 = ckt.query(mesh2, workers=workers, k=k)
            index2 = index2.flatten()
            distance2 = distance2.flatten()
            index2 = np.unique(index2[distance2 < distance_factor * np.mean(distance2)])

            pore_number1 = np.full((id_coord.shape[0]), 0)
            pore_number2 = np.full((id_coord.shape[0]), 0)
            name_label1 = 'pore.' + label_1
            name_label2 = 'pore.' + label_2
            pore_number1[index1] = 1
            pn[name_label1] = pore_number1
            pore_number2[index2] = 1
            pn[name_label2] = pore_number2
            pn[name_label1], pn[name_label2] = pn[name_label1].astype(bool), pn[name_label2].astype(
                bool)

        if status == 'z':
            diff = z_max - z_min
            m1 = np.linspace(x_min, x_max, x_block)
            m2 = np.linspace(y_min, y_max, y_block)
            m1, m2 = np.meshgrid(m1, m2)
            m1 = m1.flatten()
            m2 = m2.flatten()
            m3 = np.full((len(m1)), z_min - distance_factor2 * diff)
            m4 = np.full((len(m1)), z_max + distance_factor2 * diff)
            # print('num_nodes', len(m1))

            mesh1 = np.vstack((m1, m2, m3)).transpose()
            mesh2 = np.vstack((m1, m2, m4)).transpose()
            distance1, index1 = ckt.query(mesh1, workers=workers, k=k)  # 返回最近邻点的距离d和在数组中的顺序x
            index1 = index1.flatten()
            distance1 = distance1.flatten()
            index1 = np.unique(index1[distance1 < distance_factor * np.mean(distance1)])
            distance2, index2 = ckt.query(mesh2, workers=workers, k=k)
            index2 = index2.flatten()
            distance2 = distance2.flatten()
            index2 = np.unique(index2[distance2 < distance_factor * np.mean(distance2)])

            pore_number1 = np.full((id_coord.shape[0]), 0)
            pore_number2 = np.full((id_coord.shape[0]), 0)
            name_label1 = 'pore.' + label_1
            name_label2 = 'pore.' + label_2
            pore_number1[index1] = 1
            pn[name_label1] = pore_number1
            pore_number2[index2] = 1
            pn[name_label2] = pore_number2
            pn[name_label1], pn[name_label2] = pn[name_label1].astype(bool), pn[name_label2].astype(
                bool)

    # @staticmethod
    # def find_surface(pn, status, imsize, resolution, label_1='surface', label_2='surface', start=0.2,
    #                  end=0.8):
    #     if status == 'x':
    #         size1 = int(imsize[1] / 100)
    #         size2 = int(imsize[2] / 100)
    #         # size=int(imsize[2]/100 )
    #
    #     elif status == 'y':
    #         # size=int(imsize[1]/100)
    #         size1 = int(imsize[0] / 100)
    #         size2 = int(imsize[2] / 100)
    #     elif status == 'z':
    #         # size=int(imsize[1]/100)
    #         size1 = int(imsize[0] / 100)
    #         size2 = int(imsize[1] / 100)
    #     marker = np.ones((size1, size2, 3))
    #     for i in np.arange(size1):
    #         for j in np.arange(size2):
    #             a = np.linspace(start, end, size1)[i]
    #             b = np.linspace(start, end, size2)[j]
    #             marker[i, j, 0] = 1.1
    #             marker[i, j, 1] = a
    #             marker[i, j, 2] = b
    #     if status == 'x':
    #         marker_i = marker.reshape(size1 * size2, 3)
    #         marker_j = np.copy(marker_i)
    #         marker_j[:, 0] = -0.1
    #         # markers_0=marker_j*imsize*resolution
    #         # markers_1=marker_i*imsize*resolution
    #     elif status == 'y':
    #         marker_i = marker.reshape(size1 * size2, 3)
    #         marker_i[:, [0, 1, 2]] = marker_i[:, [1, 0, 2]]
    #         marker_j = np.copy(marker_i)
    #         marker_j[:, 1] = -0.1
    #         # markers_0=marker_j*imsize[[1,0,2]]*resolution
    #         # markers_1=marker_i*imsize[[1,0,2]]*resolution
    #     elif status == 'z':
    #         marker_i = marker.reshape(size1 * size2, 3)
    #         marker_i[:, [0, 1, 2]] = marker_i[:, [1, 2, 0]]
    #         marker_j = np.copy(marker_i)
    #         marker_j[:, 2] = -0.1
    #     else:
    #         raise Exception('error status should be x,y,z')
    #     markers_0 = marker_j * imsize * resolution
    #     markers_1 = marker_i * imsize * resolution
    #     op.topotools.find_surface_pores(pn, markers_0, label=label_1)
    #     op.topotools.find_surface_pores(pn, markers_1, label=label_2)

    @staticmethod
    def find_surface_s(pn, status='x', imsize=0, resolution=0, label_1='left', label_2='right'):
        # label1对应小值，label2对应大值，所有变量也是这个顺序
        id_coord = np.concatenate((np.array([pn['pore._id']]).T, pn['pore.coords'],
                                   np.array([pn['pore.radius']]).T), axis=1)
        percent = 0.2  # 计算的孔的比例
        angle = 30  # 向量与平面的夹角0~90，如果大于angle则将该点去除
        coverage = 0.8  # 定义为覆盖率=两圆距离/两圆半径之和
        error = 1  # 如果坐标>标准差+均值*error，则认为是异常值，将其去除
        id_small2big = []
        if status == 'x':
            norm_vector = np.array([1, 0, 0])

            tem1 = pn['pore._id'][
                pn['pore.coords'][:, 0] < np.percentile(pn['pore.coords'][:, 0], percent * 100)]
            tem2 = pn['pore._id'][
                pn['pore.coords'][:, 0] > np.percentile(pn['pore.coords'][:, 0], (1 - percent) * 100)]
            id_small2big = [tem1, tem2]


        elif status == 'y':
            norm_vector = np.array([0, 1, 0])
            tem1 = pn['pore._id'][
                pn['pore.coords'][:, 1] < np.percentile(pn['pore.coords'][:, 1], percent * 100)]
            tem2 = pn['pore._id'][
                pn['pore.coords'][:, 1] > np.percentile(pn['pore.coords'][:, 1], (1 - percent) * 100)]
            id_small2big = [tem1, tem2]
        elif status == 'z':
            norm_vector = np.array([0, 0, 1])
            tem1 = pn['pore._id'][
                pn['pore.coords'][:, 2] < np.percentile(pn['pore.coords'][:, 2], percent * 100)]
            tem2 = pn['pore._id'][
                pn['pore.coords'][:, 2] > np.percentile(pn['pore.coords'][:, 2], (1 - percent) * 100)]
            id_small2big = [tem1, tem2]

        side = np.array([0, 1])
        id_throat = [[], []]
        id_circle = [[], []]
        id_circle_all = [[], []]
        for i in side:
            list_temp = []
            for j in id_small2big[i]:

                start = np.argwhere(pn['throat.conns'][:, 0] == j).flatten()
                end = np.argwhere(pn['throat.conns'][:, 1] == j).flatten()

                cos_vector_angle = np.array([])
                if start.shape != (0,):
                    vector = pn['pore.coords'][pn['throat.conns']][start][:, 1] - \
                             pn['pore.coords'][pn['throat.conns']][
                                 start][:, 0]
                    cos_vector_angle = np.append(cos_vector_angle, vector @ norm_vector.T / np.sqrt(
                        vector[:, 0] ** 2 + vector[:, 1] ** 2 + vector[:, 2] ** 2))
                if end.shape != (0,):
                    vector = pn['pore.coords'][pn['throat.conns']][end][:, 0] - \
                             pn['pore.coords'][pn['throat.conns']][
                                 end][:, 1]
                    cos_vector_angle = np.append(cos_vector_angle, vector @ norm_vector.T / np.sqrt(
                        vector[:, 0] ** 2 + vector[:, 1] ** 2 + vector[:, 2] ** 2))
                if i == 0:
                    if np.rad2deg(np.arccos(np.min(cos_vector_angle))) < 90 + angle:
                        list_temp.append(j)
                else:
                    if np.rad2deg(np.arccos(np.max(cos_vector_angle))) > 90 - angle:
                        list_temp.append(j)
            id_throat[i] = list_temp
            # 使用Circle复检
            external_id = id_throat[i][0]
            for j in id_throat[i][1:]:
                if status == 'x':
                    index_overlap = np.argwhere(np.sqrt(
                        (id_coord[j, 2] - id_coord[external_id, 2]) ** 2 + (
                                id_coord[j, 3] - id_coord[external_id, 3]) ** 2) < coverage * (
                                                        id_coord[j, 4] + id_coord[
                                                    external_id, 4])).flatten()  # 在external列表中所有重叠的id的index
                elif status == 'y':
                    index_overlap = np.argwhere(np.sqrt(
                        (id_coord[j, 1] - id_coord[external_id, 1]) ** 2 + (
                                id_coord[j, 3] - id_coord[external_id, 3]) ** 2) < coverage * (
                                                        id_coord[j, 4] + id_coord[
                                                    external_id, 4])).flatten()  # 在external列表中所有重叠的id的index
                elif status == 'z':
                    index_overlap = np.argwhere(np.sqrt(
                        (id_coord[j, 2] - id_coord[external_id, 2]) ** 2 + (
                                id_coord[j, 1] - id_coord[external_id, 1]) ** 2) < coverage * (
                                                        id_coord[j, 4] + id_coord[
                                                    external_id, 4])).flatten()  # 在external列表中所有重叠的id的index
                else:
                    index_overlap = np.argwhere(np.sqrt(
                        (id_coord[j, 2] - id_coord[external_id, 2]) ** 2 + (
                                id_coord[j, 3] - id_coord[external_id, 3]) ** 2) < coverage * (
                                                        id_coord[j, 4] + id_coord[
                                                    external_id, 4])).flatten()  # 在external列表中所有重叠的id的index
                external_id = np.append(external_id, j)
                if index_overlap.shape != (0,):
                    overlap = np.append(j, external_id[index_overlap])
                    if status == 'x':
                        overlap_status = id_coord[overlap, 1]  # 该方向的对应id的坐标
                    elif status == 'y':
                        overlap_status = id_coord[overlap, 2]
                    elif status == 'z':
                        overlap_status = id_coord[overlap, 3]
                    else:
                        overlap_status = id_coord[overlap, 1]
                    if i == 0:
                        overlap_delete = overlap[overlap_status > np.min(overlap_status)]
                    else:
                        overlap_delete = overlap[overlap_status < np.max(overlap_status)]
                    external_id = np.setdiff1d(external_id, overlap_delete)
                id_circle_all[i] = external_id
            if status == 'x':
                id_circle_abnormal_min = np.mean(id_coord[id_circle_all[i], 1]) - np.std(
                    id_coord[id_circle_all[i], 1]) * error
                id_circle_abnormal_max = np.mean(id_coord[id_circle_all[i], 1]) + np.std(
                    id_coord[id_circle_all[i], 1]) * error
                id_circle_abnormal = id_circle_all[i][(id_coord[id_circle_all[i], 1] < id_circle_abnormal_min) | (
                        id_coord[id_circle_all[i], 1] > id_circle_abnormal_max)]
            elif status == 'y':
                id_circle_abnormal_min = np.mean(id_coord[id_circle_all[i], 2]) - np.std(
                    id_coord[id_circle_all[i], 2]) * error  # 用标准差和均值，定义超过4倍就算异常值
                id_circle_abnormal_max = np.mean(id_coord[id_circle_all[i], 2]) + np.std(
                    id_coord[id_circle_all[i], 2]) * error
                id_circle_abnormal = id_circle_all[i][(id_coord[id_circle_all[i], 2] < id_circle_abnormal_min) | (
                        id_coord[id_circle_all[i], 2] > id_circle_abnormal_max)]

            elif status == 'z':
                id_circle_abnormal_min = np.mean(id_coord[id_circle_all[i], 3]) - np.std(
                    id_coord[id_circle_all[i], 3]) * error  # 用标准差和均值，定义超过4倍就算异常值
                id_circle_abnormal_max = np.mean(id_coord[id_circle_all[i], 3]) + np.std(
                    id_coord[id_circle_all[i], 3]) * error
                id_circle_abnormal = id_circle_all[i][(id_coord[id_circle_all[i], 3] < id_circle_abnormal_min) | (
                        id_coord[id_circle_all[i], 3] > id_circle_abnormal_max)]
            else:
                id_circle_abnormal_min = np.mean(id_coord[id_circle_all[i], 1]) - np.std(
                    id_coord[id_circle_all[i], 1]) * error  # 用标准差和均值，定义超过4倍就算异常值
                id_circle_abnormal_max = np.mean(id_coord[id_circle_all[i], 1]) + np.std(
                    id_coord[id_circle_all[i], 1]) * error
                id_circle_abnormal = id_circle_all[i][(id_coord[id_circle_all[i], 1] < id_circle_abnormal_min) | (
                        id_coord[id_circle_all[i], 1] > id_circle_abnormal_max)]
            id_circle[i] = np.setdiff1d(id_circle_all[i], id_circle_abnormal)

        pore_number1 = np.zeros(id_coord.shape[0]).astype(bool)
        pore_number2 = np.zeros(id_coord.shape[0]).astype(bool)
        name_label1 = 'pore.' + label_1
        name_label2 = 'pore.' + label_2
        pore_number1[id_circle[0]] = True
        pn[name_label1] = pore_number1
        pore_number2[id_circle[1]] = True
        pn[name_label2] = pore_number2

    '''
    def find_surface(self,pn,status,imsize,resolution,label_1='surface',label_2='surface',start=0.15,end=0.9):
        if   status=='x':        
            #size1=int(imsize[0]/100 )
            #size2=int(imsize[2]/100 )
            size=int(imsize[2]/100 )

        elif status =='y':
            size=int(imsize[1]/100)
            #size1=int(imsize[0]/100 )
            #size2=int(imsize[1]/100 )
        elif status =='z':
            size=int(imsize[1]/100)
            #size1=int(imsize[1]/100 )
            #size2=int(imsize[2]/100 )        
        marker=np.ones((size,size,3))
        for i in np.arange(size):
            for j in np.arange(size):
                a=np.linspace(start,end,size)[i]
                b=np.linspace(start,end,size)[j]
                marker[i,j,0]=1.1
                marker[i,j,1]=a
                marker[i,j,2]=b
        if   status=='x':        
            marker_i=marker.reshape(size**2,3)
            marker_j=np.copy(marker_i)
            marker_j[:,0]=-0.1
        elif status =='y':
            marker_i=marker.reshape(size**2,3)
            marker_i[:,[0,1]]=marker_i[:,[1,0]]
            marker_j=np.copy(marker_i)        
            marker_j[:,1]=-0.1
        elif status =='z':
            marker_i=marker.reshape(size**2,3)
            marker_i[:,[0,2]]=marker_i[:,[2,0]]
            marker_j=np.copy(marker_i)
            marker_j[:,2]=-0.1
        markers_0=marker_j*imsize*resolution   
        markers_1=marker_i*imsize*resolution
        op.topotools.find_surface_pores(pn,markers_0,label=label_1)    
        op.topotools.find_surface_pores(pn,markers_1,label=label_2)    
    '''

    @staticmethod
    def trim_surface(pn):
        for i in ['left', 'right']:
            for j in ['back', 'front']:
                for k in ['bottom', 'top']:
                    back = np.copy(pn['pore.' + i + '_surface'] * pn['pore.' + k + '_surface'])
                    pn['pore.' + i + '_surface'][back] = False
                    pn['pore.' + k + '_surface'][back] = False
                    back = np.copy(pn['pore.' + j + '_surface'] * pn['pore.' + k + '_surface'])
                    pn['pore.' + j + '_surface'][back] = False
                    pn['pore.' + k + '_surface'][back] = False
                back = np.copy(pn['pore.' + i + '_surface'] * pn['pore.' + j + '_surface'])
                pn['pore.' + i + '_surface'][back] = False
                pn['pore.' + j + '_surface'][back] = False

    @staticmethod
    def devide_layer(pn, n, size, resolution):
        layer = {}
        # step=size*resulation/n
        layer[0] = {}  # left right
        layer[1] = {}  # back front
        layer[2] = {}  # bottom top
        for i in np.arange(3):
            index = min(pn['pore.coords'][:, i])
            step = (max(pn['pore.coords'][:, i]) - min(pn['pore.coords'][:, i])) / n[i]
            for j in np.arange(n[i]):
                layer[i][j] = np.copy(pn['pore.all'])
                layer[i][j][(pn['pore.coords'][:, i] - index) < j * step] = False
                layer[i][j][(pn['pore.coords'][:, i] - index) > (j + 1) * step] = False
        return layer

    @staticmethod
    def devide_layer_throat(pn, n, size, resolution):
        layer = {}
        # step=size*resulation/n
        layer[0] = {}  # left right
        layer[1] = {}  # back front
        layer[2] = {}  # bottom top
        for i in np.arange(3):
            index = min(pn['throat.coords'][:, i])
            step = size[i] * resolution / n[i]
            for j in np.arange(n[i]):
                layer[i][j] = np.copy(pn['throat.all'])
                layer[i][j][(pn['throat.coords'][:, i] - index) < j * step] = False
                layer[i][j][(pn['throat.coords'][:, i] - index) > (j + 1) * step] = False
        return layer

    @staticmethod
    def pore_health(pn, connections=1, equal=False):
        pores_in_conns, counts = np.unique(pn['throat.conns'], return_counts=True)
        if equal:
            pores_in_conns = pores_in_conns[counts >= connections]
        else:
            pores_in_conns = pores_in_conns[counts > connections]
        pores = np.arange(len(pn['pore.all']))
        single_pores = np.setdiff1d(pores, pores_in_conns, assume_unique=True)

        health = {}
        health['single_pore'] = single_pores
        health['single_throat'] = np.where(np.isin(pn['throat.conns'], single_pores))[0]

        return health

    @staticmethod
    def pore_health_s(pn):
        number = len(pn['pore.all'])
        conns = np.copy(pn['throat.conns'])
        health = {}
        health['single_pore'] = []
        health['single_throat'] = []

        for i in np.arange(number):
            val0 = len(conns[:, 0][conns[:, 0] == i])
            val1 = len(conns[:, 1][conns[:, 1] == i])

            if val1 + val0 <= 1:

                health['single_pore'].append(i)
                ind0 = np.argwhere(conns[:, 0] == i)
                ind1 = np.argwhere(conns[:, 1] == i)
                if len(ind0) > 0 or len(ind1) > 0:
                    health['single_throat'].append(np.concatenate((ind0, ind1))[0][0])

        return health

    @staticmethod
    def trim_pore(pn, pores=None, throats=None, bound_cond=False):
        if pores is None:
            pores = np.array([],dtype=int)
        if throats is None:
            throats = np.array([],dtype=int)
        if not isinstance(pores,np.ndarray):
            pores=np.array(pores)
        if not isinstance(throats,np.ndarray):
            throats=np.array(throats)
        backup = {}
        backup.update(pn)
        throats_isolated = np.where(np.any(np.isin(backup['throat.conns'], pores), axis=1))[0]
        throats=np.append(throats,throats_isolated)
        for i in pn:
            if 'pore' in i and 'throat' not in i:
                backup[i] = np.delete(backup[i], pores, axis=0)
            if 'throat' in i:
                backup[i] = np.delete(backup[i], throats, axis=0)

        pores_isolated=np.where(np.isin(backup['pore._id'],backup['throat.conns'])==False)[0]
        for i in pn:
            if 'pore' in i and 'throat' not in i:
                backup[i] = np.delete(backup[i], pores_isolated, axis=0)

        backup['pore._id'] = np.arange(len(backup['pore.all']))
        backup['throat._id'] = np.arange(len(backup['throat.all']))
        index_th = backup['throat._id'][(backup['throat.conns'][:, 0] >= 0)]
        throat_internal = backup['throat.conns'][(backup['throat.conns'][:, 0] >= 0)]
        index = np.digitize(throat_internal[:, 0], backup['pore.label']) - 1
        # throat_internal[:, 0] = backup['pore._id'][index]
        throat_internal[:, 0]=index
        # throat_internal[:, 0] = np.argsort(throat_internal[:, 0])
        index = np.digitize(throat_internal[:, 1], backup['pore.label']) - 1
        # throat_internal[:, 1] = backup['pore._id'][index]
        throat_internal[:, 1] = index
        if bound_cond:
            throat_out_in = backup['throat.conns'][(backup['throat.conns'][:, 0] < 0)]
            index = np.digitize(throat_out_in[:, 1], backup['pore.label']) - 1
            throat_out_in[:, 1] = backup['pore._id'][index]

            backup['throat.conns'] = np.concatenate((throat_out_in, throat_internal), axis=0)
        else:
            backup['throat.conns'] = throat_internal
        for i in pn:
            if 'throat' in i and 'conns' not in i:
                backup[i] = backup[i][index_th]

        backup['throat._id'] = np.arange(len(backup['throat.all']))
        backup['pore.label'] = np.arange(len(backup['pore.all']))
        backup['throat.label'] = np.arange(len(backup['throat.all']))
        return backup

    @staticmethod
    def trim_phase(pn, pores, throat):
        # count=len(pn)
        backup = {}
        for i in pn:
            if 'pore' in i and 'throat' not in i:

                backup[i] = np.delete(pn[i], pores,
                                      axis=0)  # if len(pn[i].shape)>1 else np.delete(pn[i],pores,axis=0)
            elif 'throat' in i:
                backup[i] = np.delete(pn[i], throat,
                                      axis=0)  # if len(pn[i].shape)>1 else np.delete(pn[i],throat,axis=0)
        backup['pore._id'] = np.arange(len(backup['pore.all']))
        backup['throat._id'] = np.arange(len(backup['throat.all']))
        return backup

    @staticmethod
    def find_if_surface(pn, index):
        res = []
        for j in index:
            b = 0
            for i in ['right', 'left', 'back', 'front', 'top', 'bottom']:
                if pn['pore.' + i + '_surface'][j]:
                    b = 'pore.' + i + '_surface'
                    res.append(b)
        return res

    @staticmethod
    def find_whereis_pore(pn, parameter, index):
        index1 = np.sort(parameter)[index]
        index2 = np.argwhere(parameter == index1)[0]
        index3 = topotools.find_if_surface(pn, index2)
        return [index1, index2, index3]

    @staticmethod
    def find_throat(pn, ids):
        ids = np.array(ids).flatten()[0]
        ind1 = np.argwhere(pn['throat.conns'][:, 0] == ids)
        ind2 = np.argwhere(pn['throat.conns'][:, 1] == ids)
        res = np.append(ind1, ind2)
        return res

    @staticmethod
    def find_neighbor_nodes(pn, ids):

        num_pore = len(pn['pore._id'])
        A = coo_matrix((pn['throat._id'], (pn['throat.conns'][:, 1], pn['throat.conns'][:, 0])),
                       shape=(num_pore, num_pore), dtype=np.float64).tolil()
        A = (A.getH() + A).tolil()
        throat_inf = np.sort(A[ids].toarray()[0])[1:]
        throat_inf = np.unique(throat_inf)
        node_inf = np.argwhere(A[ids].toarray()[0] != 0)
        node_inf = np.sort(node_inf.reshape(len(node_inf)))

        return throat_inf, node_inf

    @staticmethod
    def find_neighbor_ball(pn, ids):
        ids = np.array(ids).flatten()[0]
        res = {}
        res['total'] = topotools.find_throat(pn, ids)
        res['solid'] = []
        res['pore'] = []
        res['interface'] = []
        # num_pore=np.count_nonzero(pn['pore.void']) if 'pore.void' in pn else 0
        if res['total'].size > 0:
            for i in res['total']:
                index = pn['throat.conns'][i]
                tem = index if index[0] == ids else [ids, index[0]]
                if pn['pore.void'][ids]:
                    if pn['pore.void'][tem[1]]:
                        res['pore'].append(np.append(tem, i))
                    else:
                        res['interface'].append(np.append(tem, i))
                else:
                    if pn['pore.solid'][tem[1]]:
                        res['solid'].append(np.append(tem, i))
                    else:
                        res['interface'].append(np.append(tem, i))

            res['pore'] = np.array(res['pore'], dtype=np.int64)
            res['solid'] = np.array(res['solid'], dtype=np.int64)
            res['interface'] = np.array(res['interface'], dtype=np.int64)

            return res
        else:
            return 0

    @staticmethod
    def find_neighbor_ball_(pn, ids):
        ids = check_input(pn=pn, ids=ids)
        network_pore_void = pn['pore.void']
        network_pore_solid = pn['pore.solid']
        return find_neighbor_ball_nb(network_pore_void, network_pore_solid, pn['throat.conns'], ids)

    @staticmethod
    def H_P_fun(r, l, vis):
        g = np.pi * r ** 4 / 8 / vis / l
        return g

    @staticmethod
    def Mass_conductivity(pn):
        g_ij = []
        cond = lambda r, G, k, v: (r ** 4) / 16 / G * k / v
        g_i = cond(pn['pore.radius'][pn['throat.conns'][:, 0]],
                   pn['pore.real_shape_factor'][pn['throat.conns'][:, 0]],
                   pn['pore.real_k'][pn['throat.conns'][:, 0]],
                   pn['pore.viscosity'][pn['throat.conns'][:, 0]])
        g_j = cond(pn['pore.radius'][pn['throat.conns'][:, 1]],
                   pn['pore.real_shape_factor'][pn['throat.conns'][:, 1]],
                   pn['pore.real_k'][pn['throat.conns'][:, 1]],
                   pn['pore.viscosity'][pn['throat.conns'][:, 1]])
        g_t = cond(pn['throat.radius'],
                   pn['throat.real_shape_factor'],
                   pn['throat.real_k'],
                   pn['throat.viscosity'])

        if 'throat.conduit_lengths_pore1' in pn.keys():
            li = pn['throat.conduit_lengths_pore1']
            lj = pn['throat.conduit_lengths_pore2']
            lt = pn['throat.conduit_lengths_throat']

        elif 'throat.conduit_lengths.pore1' in pn.keys():
            li = pn['throat.conduit_lengths.pore1']
            lj = pn['throat.conduit_lengths.pore2']
            lt = pn['throat.conduit_lengths.throat']
        g_ij = (li + lj + lt) / (li / g_i + lj / g_j + lt / g_t)
        return g_ij

    @staticmethod
    # this function the viscosity has not change
    def Boundary_cond_cal(pn, throat_inlet1, throat_inlet2, fluid, newPore, Pores):
        throat_inlet_cond = []
        BndG1 = (np.sqrt(3) / 36 + 0.00001)
        BndG2 = 0.07
        for i in np.arange(len(throat_inlet1)):
            indP2 = newPore[throat_inlet1[i, 2].astype(int) - 1].astype(int)
            GT = 1 / fluid['viscosity'] * (throat_inlet1[i, 3] ** 2 / 4 / (throat_inlet1[i, 4])) ** 2 * throat_inlet1[
                i, 4] * (throat_inlet1[i, 4] < BndG1) * 0.6
            GT += 1 / fluid['viscosity'] * throat_inlet1[i, 3] ** 2 * throat_inlet1[i, 3] ** 2 / 4 / 8 / (
                    1 / 4 / np.pi) * (throat_inlet1[i, 4] > BndG2)
            GT += 1 / fluid['viscosity'] * (throat_inlet1[i, 3] ** 2 / 4 / (1 / 16)) ** 2 * (1 / 16) * (
                    (throat_inlet1[i, 4] >= BndG1) * (throat_inlet1[i, 4] <= BndG2)) * 0.5623
            GP2 = 1 / fluid['viscosity'] * (Pores[indP2, 2] ** 2 / 4 / (Pores[indP2, 3])) ** 2 * Pores[indP2, 3] * (
                    Pores[indP2, 3] < BndG1) * 0.6
            GP2 += 1 / fluid['viscosity'] * Pores[indP2, 2] ** 2 * Pores[indP2, 2] ** 2 / 4 / 8 / (1 / 4 / np.pi) * (
                    Pores[indP2, 3] > BndG2)
            GP2 += 1 / fluid['viscosity'] * (Pores[indP2, 2] ** 2 / 4 / (1 / 16)) ** 2 * (1 / 16) * (
                    (Pores[indP2, 3] >= BndG1) * (Pores[indP2, 3] <= BndG2)) * 0.5623
            # LP1 = throat_inlet2[i,3]
            LP2 = throat_inlet2[i, 4]
            LT = throat_inlet2[i, 5]
            throat_inlet_cond.append([indP2, 1 / (LT / GT + LP2 / GP2)])

        return np.array(throat_inlet_cond)

    @staticmethod
    def species_balance_conv(pn, g_ij, Tem, thermal_con_dual, P_profile, ids):
        ids = check_input(pn=pn, ids=ids)
        network_pore_void = pn['pore.void']
        network_pore_solid = pn['pore.solid']
        inner_info = pn['inner_info']
        inner_start2end = pn['inner_start2end']
        result = species_balance_conv_nb(pn['throat.radius'], pn['throat.length'], pn['throat.Cp'],
                                         pn['throat.density'], g_ij, Tem, thermal_con_dual, P_profile, ids,
                                         network_pore_void, network_pore_solid, pn['throat.conns'],
                                         inner_info, inner_start2end)

        if len(ids) == 1:
            return result[0]
        else:
            return result

    @staticmethod
    def calculate_species_flow(pn, Boundary_condition, g_ij, Tem_c, thermal_con_dual, P_profile, Num):
        check_input(pn=pn)
        network_pore_void = pn['pore.void']
        network_pore_solid = pn['pore.solid']
        inner_info = pn['inner_info']
        inner_start2end = pn['inner_start2end']
        output = {}
        total = 0
        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                result = species_balance_conv_nb(pn['throat.radius'], pn['throat.length'],
                                                 pn['throat.Cp'],
                                                 pn['throat.density'], g_ij, Tem_c, thermal_con_dual, P_profile,
                                                 pn['pore._id'][pn[n]], network_pore_void,
                                                 network_pore_solid,
                                                 pn['throat.conns'], inner_info, inner_start2end)[:, 0]
                output.update({n: np.sum(result)})
        for i in output:
            total += output[i]
        output.update({'total': np.sum(total)})
        return output

    @staticmethod
    def energy_balance_conv(pn, g_ij, Tem, thermal_con_dual, P_profile, ids):
        ids = check_input(pn=pn, ids=ids)
        network_pore_void = pn['pore.void']
        network_pore_solid = pn['pore.solid']
        inner_info = pn['inner_info']
        inner_start2end = pn['inner_start2end']
        result = energy_balance_conv_nb(pn['throat.radius'], pn['throat.length'],
                                        pn['throat.Cp'],
                                        pn['throat.density'], g_ij, Tem, thermal_con_dual, P_profile, ids,
                                        network_pore_void,
                                        network_pore_solid, pn['throat.conns'], inner_info, inner_start2end)

        if len(ids) == 1:
            return result[0]
        else:
            return result

    @staticmethod
    def calculate_heat_flow(pn, Boundary_condition, g_ij, Tem_c, thermal_con_dual, P_profile, Num):
        check_input(pn=pn)
        network_pore_void = pn['pore.void']
        network_pore_solid = pn['pore.solid']
        inner_info = pn['inner_info']
        inner_start2end = pn['inner_start2end']
        output = {}
        total = 0
        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                result = energy_balance_conv_nb(pn['throat.radius'], pn['throat.length'],
                                                pn['throat.Cp'],
                                                pn['throat.density'], g_ij, Tem_c, thermal_con_dual, P_profile,
                                                pn['pore._id'][pn[n]], network_pore_void,
                                                network_pore_solid,
                                                pn['throat.conns'], inner_info, inner_start2end)[:, 0]
                output.update({n: np.sum(result)})
        for i in output:
            total += output[i]
        output.update({'total': np.sum(total)})
        return output

    @staticmethod
    def mass_balance_conv(pn, g_ij, P_profile, ids):
        ids = check_input(pn=pn, ids=ids)
        network_pore_void = pn['pore.void']
        network_pore_solid = pn['pore.solid']
        inner_info = pn['inner_info']
        inner_start2end = pn['inner_start2end']
        result = mass_balance_conv_nb(g_ij, P_profile, ids, network_pore_void,
                                      network_pore_solid,
                                      pn['throat.conns'], inner_info, inner_start2end)

        if len(ids) == 1:
            return result[0]
        else:
            return result

    @staticmethod
    def mass_balance_conv_o(pn, g_ij, P_profile, ids):
        ids = check_input(pn=pn, ids=ids)
        # res=find_neighbor_ball(pn,[ids])
        res = topotools.find_neighbor_ball(pn, ids)
        if res == 0:
            result = 0
        elif res['pore'].size >= 1:

            delta_p = np.array(P_profile[res['pore'][:, 0].T] - P_profile[res['pore'][:, 1].T])

            cond_f = g_ij[res['pore'][:, 2].T]
            if len(delta_p) >= 1:
                flux = delta_p * cond_f
            else:
                flux = 0
            result = np.sum(flux)

        else:
            result = 0
        # print('h_conv_f=%f,h_cond_f=%f, h_cond_sf=%f,h_cond_s=%f'%(h_conv_f,h_cond_f, h_cond_sf,h_cond_s))
        return result

    @staticmethod
    def calculate_mass_flow(pn, Boundary_condition, g_ij, P_profile, Num):
        check_input(pn=pn)
        network_pore_void = pn['pore.void']
        network_pore_solid = pn['pore.solid']
        inner_info = pn['inner_info']
        inner_start2end = pn['inner_start2end']
        result = mass_balance_conv_nb(g_ij, P_profile, pn['pore._id'], network_pore_void,
                                      network_pore_solid,
                                      pn['throat.conns'], inner_info, inner_start2end)
        output = {}
        total = 0
        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                output.update({n: np.sum(result[pn[n]])})
                # abs_perm=np.sum(result[pn[n]])
        for i in output:
            total += output[i]
        output.update({'total': np.sum(total)})
        return output

    @staticmethod
    def cal_pore_veloc(pn, fluid, g_ij, P_profile, ids):
        ids = check_input(pn=pn, ids=ids)
        network_pore_void = pn['pore.void']
        network_pore_solid = pn['pore.solid']
        inner_info = pn['inner_info']
        inner_start2end = pn['inner_start2end']
        result = cal_pore_veloc_nb(pn['throat.area'], pn['pore.radius'], pn['pore.real_shape_factor'],
                                   g_ij, P_profile, ids, network_pore_void,
                                   network_pore_solid,
                                   pn['throat.conns'], inner_info, inner_start2end)

        if len(ids) == 1:
            return result[0]
        else:
            return result

    @staticmethod
    def cal_pore_flux(pn, Boundary_condition, g_ij, P_profile, Num):
        check_input(pn=pn)
        network_pore_void = pn['pore.void']
        network_pore_solid = pn['pore.solid']
        inner_info = pn['inner_info']
        inner_start2end = pn['inner_start2end']
        result = mass_balance_conv_nb(g_ij, P_profile, pn['pore._id'], network_pore_void,
                                      network_pore_solid,
                                      pn['throat.conns'], inner_info, inner_start2end)
        return result

    '''
    def add_pores(self,pn1,pn2,trail=True):#pn=pn1,pn2
        num_p=len(pn1['pore.all'])
        num_t=len(pn1['throat.all'])
        pn2['throat._id']+=num_t
        pn2['pore.label']+=num_p

        pn={}
        if trail:
            pn2['throat.conns']+=num_p
        for i in pn2:
            if i not in pn1:
                pn[i]=np.zeros(num_p).astype(bool)
                pn[i]=np.concatenate((pn[i],pn2[i])) 

            else:
                pn[i]=np.concatenate((pn1[i],pn2[i])) 
        return pn
    '''

    @staticmethod
    def add_pores(pn1, pn2, trail=True):  # pn=pn1,pn2
        num_p = len(pn1['pore.all'])
        num_t = len(pn1['throat.all'])
        pn2['throat._id'] += num_t
        pn2['pore.label'] += num_p

        pn = {}
        if trail:
            pn2['throat.conns'] += num_p
        for i in pn2:
            if i not in pn1:
                pn[i] = np.zeros(num_p).astype(bool) if 'pore' in i else np.zeros(num_t).astype(bool)
                pn[i] = np.concatenate((pn[i], pn2[i]))

            else:
                pn[i] = np.concatenate((pn1[i], pn2[i]))
        pn['throat._id'] = np.sort(pn['throat._id'])
        pn['pore._id'] = np.sort(pn['pore.label'])
        pn['throat.label'] = np.sort(pn['throat._id'])
        pn['pore.label'] = np.sort(pn['pore.label'])
        return pn

    @staticmethod
    def clone_pores(pn, pores, label='clone_p'):
        clone = {}
        num = len(pn['pore.all'][pores])
        for i in pn:
            if 'pore' in i and 'throat' not in i:
                if '_id' in i:
                    clone[i] = np.arange(num)

                elif pn[i].dtype == bool:
                    clone[i] = np.zeros(num).astype(bool)
                else:
                    clone[i] = pn[i][pores]


            elif 'throat' in i:
                clone[i] = np.ones(num) * np.average(pn[i])
        clone['pore.all'] = np.ones(num).astype(bool)
        clone[label] = np.ones(num).astype(bool)
        return clone

    @staticmethod
    def merge_clone_pore(pn, pores, radius, resolution, imsize, side, label='clone_p'):
        pn['pore.coords'] += [resolution, resolution, resolution]
        clone = topotools.clone_pores(pn, pores, label=label)
        org_coords = np.copy(clone['pore.coords'])
        if side == 'left':
            index = np.min(clone['pore.coords'][:, 0])
            clone['pore.coords'] *= [0, 1, 1]

            clone['pore.coords'] += [index - resolution, 0, 0]
        elif side == 'right':
            index = np.max(clone['pore.coords'][:, 0])
            clone['pore.coords'] *= [0, 1, 1]

            clone['pore.coords'] += [index + resolution, 0, 0]
        elif side == 'back':
            index = np.min(clone['pore.coords'][:, 1])
            clone['pore.coords'] *= [1, 0, 1]

            clone['pore.coords'] += [0, index - resolution, 0]
        elif side == 'front':
            index = np.max(clone['pore.coords'][:, 1])
            clone['pore.coords'] *= [1, 0, 1]

            clone['pore.coords'] += [0, index + resolution, 0]
        elif side == 'bottom':
            index = np.min(clone['pore.coords'][:, 2])
            clone['pore.coords'] *= [1, 1, 0]

            clone['pore.coords'] += [0, 0, index - resolution]
        elif side == 'top':
            index = np.max(clone['pore.coords'][:, 2])
            clone['pore.coords'] *= [1, 1, 0]

            clone['pore.coords'] += [0, 0, index + resolution]
        num = pn['pore.all'].size
        clone['throat.conns'] = np.vstack((clone['pore.label'], clone['pore._id'] + num)).T
        clone['pore.label'] = clone['pore._id']
        num_T = len(clone['throat.conns'])
        clone['throat.solid'] = np.zeros(num_T).astype(bool)
        # clone['throat.solid'][(clone['throat.conns'][:,0]>=num_pore)&(clone['throat.conns'][:,1]>=num_pore)]=True
        clone['throat.solid'][pn['pore.solid'][clone['throat.conns'][:, 0]]] = True
        clone['throat.connect'] = np.zeros(num_T).astype(bool)
        clone['throat.connect'][pn['pore.void'][clone['throat.conns'][:, 0]]] = True
        clone['throat.void'] = np.zeros(num_T).astype(bool)
        clone['throat.void'] = ~(clone['throat.solid'] | clone['throat.connect'])
        clone['throat.label'] = np.arange(num_T)
        clone['throat._id'] = np.arange(num_T)
        clone['throat.all'] = np.ones(num_T).astype(bool)
        clone['throat.radius'] = radius  # clone['throat.all']*radius
        clone['throat.length'] = np.abs(np.linalg.norm(org_coords - clone['pore.coords'], axis=1))
        clone['throat.volume'] = np.pi * radius ** 2 * clone['throat.length']
        return clone

    @staticmethod
    def connect_repu_network(pn, side):
        if side in ['right', 'left']:
            way = 0
            side1 = np.array(['right', 'left'])[~(np.array(['right', 'left']) == side)][0]
        elif side in ['back', 'front']:
            way = 1
            side1 = np.array(['back', 'front'])[~(np.array(['back', 'front']) == side)][0]
        elif side in ['bottom', 'top']:
            way = 2
            side1 = np.array(['bottom', 'top'])[~(np.array(['bottom', 'top']) == side)][0]
        else:
            raise Exception('The side is not correct, should be right,left,back,front,top,bottom')

        pn2 = copy.deepcopy(pn)
        copy_surf = np.max(pn['pore.coords'][pn['pore.' + side + '_surface']][:, way])
        num_node = len(pn['pore.all'])
        pn2['pore.coords'][:, way] = 2 * copy_surf - pn2['pore.coords'][:, way]
        # conn_t,conn_n=topotools().find_neighbor_nodes(pn, pn['pore._id'][pn['pore.'+side+'_surface']])

        for i in pn['pore._id'][pn['pore.' + side + '_surface']]:
            pn2['throat.conns'][:, 0][pn2['throat.conns'][:, 0] == i] = i - num_node
            pn2['throat.conns'][:, 1][pn2['throat.conns'][:, 1] == i] = i - num_node

        pn2 = topotools.trim_phase(pn2, pn['pore._id'][pn['pore.' + side + '_surface']].astype(int),
                                   [])
        pn = topotools.add_pores(pn, pn2, trail=True)
        li, indices, counts = np.unique(pn['throat.conns'], axis=0, return_counts=True, return_index=True)
        duplicates = indices[np.where(counts > 1)]
        pn = topotools.trim_pore(pn, [], duplicates.astype(int))

        pn['pore._id'] = np.arange(len(pn['pore._id']))
        index = np.digitize(pn['throat.conns'][:, 0], pn['pore.label']) - 1
        pn['throat.conns'][:, 0] = pn['pore._id'][index]
        index = np.digitize(pn['throat.conns'][:, 1], pn['pore.label']) - 1
        pn['throat.conns'][:, 1] = pn['pore._id'][index]

        pn['pore.label'] = np.arange(len(pn['pore._id']))
        del pn['pore.' + side + '_surface'], pn['pore.' + side1 + '_surface']
        '''
        if side1 in ['left','back','bottom']:

            topotools().find_surface(pn,np.array(['x','y','z'])[way],
                                 imsize[way]*2,resolution,label_1=side1+'_surface',label_2=side+'_surface')
        else:
            topotools().find_surface(pn,np.array(['x','y','z'])[way],
                                 imsize[way]*2,resolution,label_1=side+'_surface',label_2=side1+'_surface') 
        '''
        return pn

    @staticmethod
    def update_inner_info(pn):
        print('Updating pore conns information')
        '''
        0=Pore, 1=Solid, 2=Interface
        '''
        if 'pore.void' in pn.keys():
            pass
        else:
            pn['pore.void'] = np.zeros(len(pn['pore._id']), dtype=bool)
        if 'pore.solid' in pn.keys():
            pass
        else:
            pn['pore.solid'] = np.zeros(len(pn['pore._id']), dtype=bool)

        len_throat_conns = len(pn['throat.conns'])
        index_throat = np.arange(0, len_throat_conns)
        total_information = np.zeros((2 * len_throat_conns, 4), dtype=np.int64)
        total_information[:len_throat_conns, 0] = pn['throat.conns'][:, 0]
        total_information[len_throat_conns:, 0] = pn['throat.conns'][:, 1]
        total_information[:len_throat_conns, 1] = pn['throat.conns'][:, 1]
        total_information[len_throat_conns:, 1] = pn['throat.conns'][:, 0]
        total_information[:len_throat_conns, 2] = index_throat
        total_information[len_throat_conns:, 2] = index_throat
        # total_information[:,3]=-1
        total_information = total_information[np.argsort(total_information[:, 0])]
        elements, counts = np.unique(total_information[:, 0], return_counts=True)
        start = np.concatenate((np.array([0]), np.cumsum(counts)[0:-1]), axis=0)
        end = np.cumsum(counts)
        start2end = -np.ones((len(pn['pore._id']), 2), dtype=np.int64)
        start2end[elements] = np.concatenate((np.array([start]).T, np.array([end]).T), axis=1)
        total_information[:, 3] = np.where(
            pn['pore.void'][total_information[:, 0]] & pn['pore.void'][total_information[:, 1]], 0, np.where(
                pn['pore.solid'][total_information[:, 0]] & pn['pore.solid'][total_information[:, 1]], 1, 2))

        temp = {'inner_info': total_information, 'inner_start2end': start2end}
        pn.update(temp)
        print('Finished updating pore info')
        return pn
