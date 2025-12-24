import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import numpy as np
import scipy
# from tqdm.notebook import tqdm
import time
import psutil
from datetime import datetime
from functools import wraps
from joblib import Parallel, delayed
import matplotlib
hartree = 27.2113845

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
import os 
np.set_printoptions(precision=8)
np.set_printoptions(suppress=True, floatmode='fixed')
from matplotlib import rcParams
config = {
    "font.family":'serif',
    "font.size": 10,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'], # simsun字体中文版就是宋体
}

plt.rc('font',family='Times New Roman')

rcParams.update(config)
def smooth_curve(x, y, num_points=1000):
    """
    Smooth a curve using interpolation.

    Parameters:
        x (array-like): Array of x coordinates.
        y (array-like): Array of y coordinates.
        num_points (int): Number of points for interpolation (default: 1000).

    Returns:
        array-like: Smoothed y coordinates.
    """
    f = interp1d(x, y, kind='cubic')
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    return x_smooth, f(x_smooth)

def plot_band_structure_new_(ax, x, bands, values,lw, label, linestyle, offset, num,loc,vmin,vmax,cmap='viridis_r',bar_show=False,**kwargs):
    """
    Plot a band structure with values represented by color.

    Parameters:
        x (array-like): Array of x coordinates.
        bands (array-like): 2D array of y coordinates representing the bands.
        values (array-like): Array of values corresponding to each point in bands.

    Returns:
        None
    """
    # Create line segments for each band
    segments_list = []
    for band in bands.T:
        points = np.array([x, band]).T.reshape(-1, 1, 2)
        segments_list.append(np.concatenate([points[:-1], points[1:]], axis=1))

    # Create a colormap and normalize
    cmap = plt.get_cmap(cmap)
    # cmap = plt.get_cmap('viridis')
    if vmin == 0 and vmax == 1:
        norm = plt.Normalize(np.min(values), np.max(values))
    else:
        norm = plt.Normalize(vmin, vmax)
    print("shape of segments_list",np.shape(segments_list))
    # Create subplots
    # fig, ax = plt.subplots()

    # Loop through each set of line segments and create LineCollection
    # for i, segments in enumerate(segments_list):
    #     if i == offset:
    #         print("shit")
    #         lc = LineCollection(segments, cmap=cmap, norm=norm, label=label)
    #         lc = LineCollection(segments, cmap=cmap, norm=norm)
    #         lc.set_array(values[i])
    #         lc.set_linewidth(1)
    #         line = ax.add_collection(lc)
    #     elif i < offset + num:
    #         lc = LineCollection(segments, cmap=cmap, norm=norm)
    #         lc.set_array(values[i])
    #         lc.set_linewidth(1)
    #         ax.add_collection(lc)
        # Loop through each set of line segments and create LineCollection
    for i, segments in enumerate(segments_list):
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(values[:, i])  # 设置每个点的值
        lc.set_linewidth(lw)
        ax.add_collection(lc)
        if i == offset:
            lc.set_label(label)  # 设置标签

    # for i in range(num):
    #     if i == num-1:
    #         ax.plot(x, bands[:,i+offset], color=cmap(norm(np.mean(values[:,i+offset]))), linewidth=lw, linestyle=linestyle, label=label)
    #     else:
    #         ax.plot(x, bands[:,i+offset], color=cmap(norm(np.mean(values[:,i+offset]))), linewidth=lw, linestyle=linestyle)
    

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if bar_show:
        cbar = plt.colorbar(sm, ax=ax)
        # cbar.set_label('Values')
        print(ax.get_position())
        # cbar.ax.set_position(loc)
        # 设置颜色条的刻度间隔
        cbar.locator = MultipleLocator(0.02)
        cbar.update_ticks()
    # plt.show()
    # cbar.ax.tick_params(labelsize=12)

    # Add colorbar
    # cbar = fig.colorbar(line, ax=ax)
    # cbar.set_label('Values')

    # Set axis limits
    # ax.set_xlim(x.min(), x.max())
    # ax.set_ylim(np.min(bands)*0-0.12, np.max(bands)+0.02)

def plot_band_structure_new(ax, x, bands, values, lw, label, linestyle, offset, num, loc, vmin=0, vmax=1, cmap='viridis_r',bar_show=False,**kwargs):
    """
    Plot a band structure with values represented by color.

    Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot.
        x (array-like): Array of x coordinates.
        bands (array-like): 2D array of y coordinates representing the bands.
        values (array-like): Array of values corresponding to each point in bands.
        lw (float): Line width for the bands.
        label (str): Label for the plot.
        linestyle (str): Line style for the bands.
        offset (float): Offset value for shifting the bands.
        num (int): Number of bands to plot.
        loc (str): Location for the legend.
        vmin (float): Minimum value for normalization.
        vmax (float): Maximum value for normalization.
        cmap (str): Colormap for representing values.

    Returns:
        None
    """
    # Ensure values are in the correct shape
    values = np.array(values)

    # Create a colormap and normalize
    cmap = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin, vmax)

    for i in range(num):
        band = bands[:, i] + offset
        proj = values[:, i]

        # Create line segments for each band
        points = np.array([x, band]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a LineCollection for the current band
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(proj)
        lc.set_linewidth(lw)
        lc.set_linestyle(linestyle)
        if i == 0:
            lc.set_label(label)
        ax.add_collection(lc)

    # Add colorbar
    if bar_show:
        cbar = plt.colorbar(lc, ax=ax)
    # cbar.set_label('Projection Intensity')

    # Set labels and title
    # ax.set_xlabel('k-point')
    # ax.set_ylabel('Energy (eV)')
    ax.set_title(label)

    # Adjust axis limits
    # ax.set_xlim(x.min(), x.max())
    # ax.set_ylim(np.min(bands + offset), np.max(bands + offset))

    # Add legend
    # ax.legend(loc=loc)

def plot_band(ax, kx1, band, c="dodgerblue", lw=1.5, linestyle='-', label='', offset=2000, num=200, smooth=False,num_points=1000,values=[],
              proj=False,loc=[0.9,0,0.02,0.5],vmin=0,vmax=1,cmap='viridis_r',sca=False,bar_show=False,**kwargs):
    """
    Plot a band structure.

    Parameters:
        ax (Axes): Matplotlib Axes object to plot on.
        kx1 (array-like): Array of x coordinates.
        band (array-like): 2D array of y coordinates representing the bands.
        c (str): Color of the curve (default: 'dodgerblue').
        lw (float): Line width (default: 1.5).
        linestyle (str): Line style (default: '-').
        label (str): Label for the curve (default: 'TAPW').
        offset (int): Offset index (default: 2000).
        num (int): Number of curves to plot (default: 200).
        smooth (bool): Whether to apply smoothing (default: False).

    Returns:
        None
    """
    # for i in range(num):
        # y = band[:, offset + i]
        # if smooth:
        #     y = smooth_curve(kx1, y, num_points=num_points)
        # if i == 0:
        #     ax.plot(kx1, y, c=c, linestyle=linestyle, lw=lw, label=label)
        # else:
        #     ax.plot(kx1, y, c=c, linestyle=linestyle, lw=lw)
    if smooth:
        bands_smooth = []
        for i in range(np.shape(band)[1]):
            kx1_smooth, y_smooth = smooth_curve(kx1, band[:, i], num_points=num_points)
            bands_smooth.append(y_smooth)
        bands = np.array(bands_smooth).T
        kx = kx1_smooth
    else:
        bands = band
        kx = kx1
    
    if proj:
        plot_band_structure_new(ax, kx, bands, values, lw, label,linestyle,offset,num,loc,vmin,vmax,cmap=cmap,bar_show=bar_show,**kwargs)
    else:
        if sca:
            for i in range(num):
                if len(label)>0:
                    if i == num-1:
                        ax.scatter(kx, bands[:,i+offset], c=c, s=lw, label=label, **kwargs)
                    else:
                        ax.scatter(kx, bands[:,i+offset], c=c, s=lw, **kwargs) 
                else:
                    ax.scatter(kx, bands[:,i+offset], c=c, s=lw, **kwargs)
        else:
            for i in range(num):
                if len(label)>0:
                    if i == num-1:
                        ax.plot(kx, bands[:,i+offset], c=c, linestyle=linestyle, lw=lw, label=label, **kwargs)
                    else:
                        ax.plot(kx, bands[:,i+offset], c=c, linestyle=linestyle, lw=lw, **kwargs) 
                else:
                    ax.plot(kx, bands[:,i+offset], c=c, linestyle=linestyle, lw=lw, **kwargs)

def set_pic(ax,ymin,ymax,ylim = True,meV=True,xticks =np.array([0, 0.125, 0.197, 0.341]),legend_show=True,transparent_flag = False,
            xlabels = [r'${\Gamma}_{\mathrm{M}}$', r'${\mathrm{M}}_{\mathrm{M}}$', r'${\mathrm{K}}_{\mathrm{M}}$', r'${\Gamma}_{\mathrm{M}}$'], 
            title = '4.41 Bilayer tMoTe2 w SOC G vally',save = False,savepath = '',legend_fontsize = 9):
    # Add reference lines
    ax.plot([0, np.max(xticks)], [0, 0], c='black', lw=0.6, linestyle='-',alpha=0.4)
    for i in range(len(xticks)-2):
        ax.plot([xticks[i+1], xticks[i+1]], [ymin-0.1, ymax + 0.1], lw=0.6,c='black', linestyle='-',alpha=0.4)
    # ax.plot([0.125, 0.125], [ymin-0.1, ymax + 0.1], lw=1,c='black', linestyle='--')
    # ax.plot([0.197, 0.197], [ymin-0.1, ymax + 0.1], lw=1,c='black', linestyle='--')
    # ax.plot([0.341, 0.341], [emin, emax + 0.01], c='blue', linestyle='--')

    # Set y-axis limits
    if ylim:
        ax.set_ylim(ymin, ymax)

    # Set x-axis limits and ticks
    ax.set_xlim(np.min(xticks), np.max(xticks))
    ax.set_xticks(xticks)
    print(xticks)
    print(xlabels)
    scale = 1.5
    ax.set_xticklabels(xlabels,fontsize=11*scale,fontfamily='Times New Roman')  # Bold x-axis tick labels
    if meV:
        # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x*1000)))
        ax.set_ylabel('Energy (meV)', fontsize=12*scale,fontfamily='Times New Roman')  # Bold y-axis label
    else:
        ax.set_ylabel('Energy (eV)', fontsize=12*scale,fontfamily='Times New Roman')  # Bold y-axis label 

    # ax.set_yticks(fontproperties = 'Times New Roman', size = 12)
    # ax.tick_params(axis='y', labelsize=12, labelfamily='Times New Roman')
    y1_label = ax.get_yticklabels() 
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
    [y1_label_temp.set_fontsize(11*scale) for y1_label_temp in y1_label]
    [y1_label_temp.set_position((0.01, y1_label_temp.get_position()[1])) for y1_label_temp in y1_label]  # Adjust the x position



    # Set plot title
    ax.set_title(title,fontsize=12*scale,fontfamily= 'Times New Roman')
    if legend_show:
        legend = ax.legend(loc='upper right',fontsize=legend_fontsize*scale)
        labelss = legend.get_texts()
        [label.set_fontname('Times New Roman') for label in labelss]

    # Show the plot
    if save:
        ax.set_facecolor('none')
        plt.savefig(savepath, dpi=300,bbox_inches='tight',transparent=transparent_flag)
    plt.show()

Nk = 11
kx1 = np.hstack(
    (np.hstack((np.linspace(0,0.125,Nk+1),np.linspace(0.125,0.197177,Nk+1)[1:])),
    np.linspace(0.197177,0.341521,Nk+1)[1:]))
"""  0.000000   0.000000   0.000000   0.000000
  0.333333   0.333333   0.000000   0.065313
  0.666667  -0.333333   0.000000   0.130626
  0.000000   0.500000   0.000000   0.217027
  0.500000   0.000000   0.000000   0.273589
  0.500000  -0.500000   0.000000   0.330152 """
kx_GK1K2M1M2M3 = np.array([0.000000, 0.065313, 0.130626, 0.217027, 0.273589, 0.330152])
Nk = 20
kx61 = np.hstack(
    (np.hstack((np.linspace(0,0.125,Nk+1),np.linspace(0.125,0.197177,Nk+1)[1:])),
    np.linspace(0.197177,0.341521,Nk+1)[1:]))

Nk = 6
kx19 = np.hstack(
    (np.hstack((np.linspace(0,0.125,Nk+1),np.linspace(0.125,0.197177,Nk+1)[1:])),
    np.linspace(0.197177,0.341521,Nk+1)[1:]))


def timing_decorator_factory(process_id):
    def timing_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if process_id == 0:
                start_time = time.time()
                process = psutil.Process()
                mem_before = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB

                result = func(*args, **kwargs)

                mem_after = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
                end_time = time.time()
                duration = end_time - start_time
                mem_peak = mem_after - mem_before

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] Function '{func.__name__}' executed in {duration:.6f} seconds, Memory peak: {mem_peak:.6f} GB")
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return timing_decorator

class KPathGenerator:
    def __init__(self, Amat):
        self.Amat = Amat
        self.astar, self.bstar, self.cstar = self.calculate_reciprocal_vectors(Amat)

        self.x_ticks = []
        self.labels_ticks = []
        self.kpoints = []
        self.high_symmetry_points = []
        self.labels = []
        self.segment_points = 0

    @staticmethod
    def calculate_reciprocal_vectors(Amat):
        a, b, c = Amat
        vol = np.dot(a, np.cross(b, c))
        astar = 2 * np.pi * np.cross(b, c) / vol
        bstar = 2 * np.pi * np.cross(c, a) / vol
        cstar = 2 * np.pi * np.cross(a, b) / vol
        return astar, bstar, cstar
    
    def read_and_generate_kpath(self, file_path: str, output_file_path: str) -> None:
        # 读取kpath文件
        with open(file_path, 'r') as input_file:
            lines = input_file.readlines()

        self.segment_points = int(lines[1])
        self.high_symmetry_points = []
        self.labels = []

        for i in range(4, len(lines)):
            coordinates = np.array([float(coord) for coord in lines[i].split()[:3]])
            if len(coordinates) == 3:
                label = lines[i].split()[-1]
                if label in ['GAMMA', '\\GAMMA', 'G']:
                    label = r'$\Gamma$'
                self.high_symmetry_points.append(coordinates)
                self.labels.append(label)

        self.high_symmetry_points = np.array(self.high_symmetry_points)

        # 生成kpath
        Amat_reciprocal = np.array([self.astar, self.bstar, self.cstar])
        x = 0.0
        num_high_symmetry_points = len(self.high_symmetry_points)
        self.x_ticks = []
        self.labels_ticks = []
        self.kpoints = []

        with open(output_file_path, 'w') as output_file:
            self.x_ticks.append(x)
            self.labels_ticks.append(self.labels[0])
            for i in range(int(num_high_symmetry_points / 2)):
                delta = self.distance(
                    self.direct_cart_real(Amat_reciprocal, self.high_symmetry_points[2 * i + 1]),
                    self.direct_cart_real(Amat_reciprocal, self.high_symmetry_points[2 * i])
                ) / self.segment_points

                for j in range(self.segment_points):
                    fraction = 1.0 * j / self.segment_points
                    interpolated_point = (1.0 - fraction) * self.high_symmetry_points[2 * i] + fraction * self.high_symmetry_points[2 * i + 1]

                    output_file.write(f'{interpolated_point[0]:>10.6f} {interpolated_point[1]:>10.6f} {interpolated_point[2]:>10.6f} {x:>10.6f}\n')
                    self.kpoints.append(np.append(interpolated_point, x))
                    x += delta

                if i < int(num_high_symmetry_points / 2) - 1 and self.labels[2 * i + 2] != self.labels[2 * i + 1]:
                    output_file.write(f'{self.high_symmetry_points[2 * i + 1][0]:>10.6f} {self.high_symmetry_points[2 * i + 1][1]:>10.6f} {self.high_symmetry_points[2 * i + 1][2]:>10.6f} {x:>10.6f}\n')
                    self.kpoints.append(np.append(self.high_symmetry_points[2 * i + 1], x))

                if i < int(num_high_symmetry_points / 2) - 1:
                    if self.labels[2 * i + 2] == self.labels[2 * i + 1]:
                        self.x_ticks.append(x)
                        self.labels_ticks.append(self.labels[2 * i + 1])
                    else:
                        self.x_ticks.append(x)
                        self.labels_ticks.append(f"{self.labels[2 * i + 1]}|{self.labels[2 * i + 2]}")

            self.x_ticks.append(x)
            self.labels_ticks.append(self.labels[-1])
            output_file.write(f'{self.high_symmetry_points[2 * i + 1][0]:>10.6f} {self.high_symmetry_points[2 * i + 1][1]:>10.6f} {self.high_symmetry_points[2 * i + 1][2]:>10.6f} {x:>10.6f}\n')
            self.kpoints.append(np.append(self.high_symmetry_points[2 * i + 1], x))
            self.kpoints = np.array(self.kpoints)

    @staticmethod
    def distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    @staticmethod
    def cart_direct_real(Amat, pos_cart):
        return np.dot(np.linalg.inv(Amat.T), np.array(pos_cart))

    @staticmethod
    def direct_cart_real(Amat, pos_direct):
        return np.dot(Amat.T, np.array(pos_direct))

    @staticmethod
    def generate_chern_kmesh(num):
        num_k = num ** 2
        kpoints = np.zeros((num_k, 3))
        for i in range(num):
            for j in range(num):
                kpoints[i * num + j] = np.array([i / (num - 1), j / (num - 1), 0])
        return kpoints

    @staticmethod
    def generate_foundamental_kmesh(num, a1, a2):
        num_k = num ** 2
        kpoints = np.zeros((num_k, 3))
        for i in range(num):
            for j in range(num):
                kpoints[i * num + j] = i / (num - 1) * a1 + j / (num - 1) * a2
        return kpoints

    @staticmethod
    def generate_foundamental_kmesh_kpBC(delta_a1, delta_a2, num, BM):
        # num = int(np.linalg.norm(a2)/delta+1e-3)+1
        num_k = num ** 3
        kpoints = np.zeros((num_k, 3))
        for i in range(num):
            for j in range(num):
                kpoints[i * num + j] = (i-int(num/2))*delta_a1 + (j-int(num/2))*delta_a2
        # kpoints = kpoints@np.linalg.inv(BM)
        return kpoints

def rot(vec,theta):
    theta = theta/180*np.pi
    rot_mat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return np.dot(rot_mat,vec)


SYMMETRIZE_GLOBAL_CACHE = {}
SYMMETRIZE_MONOMIAL_OP_CACHE = {}
SYMMETRIZE_MONOMIAL_OP_VALIDATED = set()
SYMMETRIZE_COMPOSED_OP_CACHE = {}
SYMMETRIZE_COMPOSED_OP_VALIDATED = set()




#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整连续模型生成与系数提取框架
==================================

【主要功能】
1. 根据输入的 Q 点（两层 Q_set，第一层与第二层各自的轨道数可能不同）、
   倒格基矢 bM1、bM2 以及跃迁 harmonic 映射（intra_harmonics_map 与 inter_harmonics_map），
   自动构造连续模型中所有 term（项）。每个 term 的指标（包括 Mz、Mz_star、层、轨道、p）
   由用户输入的多项式阶数限制决定（例如 Kinect 项中 Mz+Mz_star 最大为 max_order["Kinect"]，
   并且 Kinect 条件为层内且 p=0 且轨道相同，其中 Mz=Mz^*=0 对应 onsite 项）。
2. 每个 term 都有一个基函数 Y_basis(k)（由 make_Y_basis_function 生成），
   此基函数的矩阵维度为：第一层 Q 点数量×第一层轨道数 + 第二层 Q 点数量×第二层轨道数，
   且索引按：第一层 Q 的第1轨道、第2轨道…，然后第二层 Q 的第1轨道、第2轨道…排序。
3. 对称化：每个 term 都附带一组对称操作（例如 "C3z"、"C2yT"、"C2zT"），
   对称化采用公式
         Y_symm(k) = (1/N) Σ_g D(g) Y(g⁻¹k) D(g)†,
   最后对结果做 Hermitian 化，即 (Y + Y†)/2。  
   此处对称操作矩阵 D(g)不是固定的，而是根据输入的 Q 自动生成，
   例如 C3z 的矩阵由 SymmetryGenerator.get_C3z_operator() 生成，其维度与 Y_basis 保持一致。
4. 每个 term 分成两部分：一部分是直接由 Y_basis 得到，对应 r_value_real，
   另一部分是先乘以 i 后再进行对称化，对应 r_value_imag。
5. 根据 term 的 tag（"Kinect", "intra", "inter"）分组，
   分别对每组 term（以及各部分）在多 k 点下采样（block_diag拼接后）进行正交化，
   并利用数值哈密顿量 heff 与正交化后的基函数取迹求解线性方程组得到系数。
6. 最后将所有 term 的系数保存（例如保存在一个字典中），同时提供 assemble_hamiltonian(k)
   方法组装连续模型哈密顿量。

【注意】  
– 本代码中所有对外接口均封装在 ContinuumModelBuilder 类中；  
– 对称操作矩阵由 SymmetryGenerator 类根据输入 Q 自动生成；  
– 各函数均带有详细注释，可根据实际模型调整。

"""

import numpy as np
import scipy.linalg
from scipy import sparse
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, List, Any
import json
import time
import sys
from datetime import datetime
from functools import wraps
import psutil
from joblib import Parallel, delayed

def timing_decorator_factory(process_id):
    def timing_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if process_id == 0:
                start_time = time.time()
                process = psutil.Process()
                mem_before = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB

                result = func(*args, **kwargs)

                mem_after = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
                end_time = time.time()
                duration = end_time - start_time
                mem_peak = mem_after - mem_before

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] Function '{func.__name__}' executed in {duration:.6f} seconds, Memory peak: {mem_peak:.6f} GB")
                sys.stdout.flush() 
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return timing_decorator

def generate_orb(classname,l1,l2, max_M_sum, max_p_order,orb_list,intra_harmonics_map, symm=[{"name":"TR"}]):
    """
    classname: same_spin_diag, same_spin_offdiag, diff_spin_diag, diff_spin_offdiag
    max_M_sum: 最大 M_sum
    max_p_order: 最大 p_order
    """
    key_list = []
    if classname == "same_spin_diag":
        for a in range(1,orb_list[l1-1]+1):
            for b in range(1,orb_list[l2-1]+1):
                if (a-b)%2 != 0:
                    continue
                for M_sum in range(0, max_M_sum+1):
                    for Mz in range(0, M_sum+1):
                        Mz_star = M_sum - Mz
                        for p_order in range(1, 2):
                            p = intra_harmonics_map[p_order]
                            key = ContinuumTermKey(Mz, Mz_star, l1, l2, a, b, tuple(p))
                            key_list.append(key)
    elif classname == "same_spin_offdiag":
        for a in range(1,orb_list[l1-1]+1):
            for b in range(1,orb_list[l2-1]+1):
                if (a-b)%2 != 0:
                    continue
                for M_sum in range(0, max_M_sum+1):
                    for Mz in range(0, M_sum+1):
                        Mz_star = M_sum - Mz
                        for p_order in range(2, max_p_order+1):
                            p = intra_harmonics_map[p_order]
                            key = ContinuumTermKey(Mz, Mz_star, l1, l2, a, b, tuple(p))
                            key_list.append(key)
    elif classname == "diff_spin_diag":
        for a in range(1,orb_list[l1-1]+1):
            for b in range(1,orb_list[l2-1]+1):
                if (a-b)%2 == 0:
                    continue
                for M_sum in range(0, max_M_sum+1):
                    for Mz in range(0, M_sum+1):
                        Mz_star = M_sum - Mz
                        for p_order in range(1, 2):
                            p = intra_harmonics_map[p_order]
                            key = ContinuumTermKey(Mz, Mz_star, l1, l2, a, b, tuple(p))
                            key_list.append(key)
    elif classname == "diff_spin_offdiag":
        for a in range(1,orb_list[l1-1]+1):
            for b in range(1,orb_list[l2-1]+1):
                if (a-b)%2 == 0:
                    continue
                for M_sum in range(0, max_M_sum+1):
                    for Mz in range(0, M_sum+1):
                        Mz_star = M_sum - Mz
                        for p_order in range(2, max_p_order+1):
                            p = intra_harmonics_map[p_order]
                            key = ContinuumTermKey(Mz, Mz_star, l1, l2, a, b, tuple(p))
                            key_list.append(key)
    elif classname == "Kinect":
        for a in range(1,orb_list[l1-1]+1):
            b = a
            for M_sum in range(0, max_M_sum+1):
                for Mz in range(0, M_sum+1):
                    Mz_star = M_sum - Mz
                    if Mz + Mz_star == 0 or (Mz - Mz_star) % 3 != 0 or Mz_star>Mz:
                        continue
                    for p_order in range(1, 2):
                        p = intra_harmonics_map[p_order]
                        key = ContinuumTermKey(Mz, Mz_star, l1, l2, a, b, tuple(p))
                        key_list.append(key)
    elif classname == "Onsite":
        for a in range(1,orb_list[l1-1]+1):
            b = a
            key = ContinuumTermKey(0, 0, l1, l2, a, b, (0.0, 0.0))
            key_list.append(key)
    else:
        raise ValueError("Invalid classname. Choose from 'same_spin_diag', 'same_spin_offdiag', 'diff_spin_diag', 'diff_spin_offdiag'.")
    
    #根据对称性删除一些key
    def get_opposite_spin(a):
        return 2-(a-1)%2+(a-1)//2*2
    for symm_op in symm:
        if symm_op["name"] == "TR":

            key_list_new = []
            
            spin_up_dn_list = []
            spin_up_dn_list_diag = []
            spin_up_dn_list_offdiag = []
            
            for a in range(1,orb_list[l1-1]+1):
                for b in range(1,orb_list[l2-1]+1):
                    if (a-b)%2 == 1:
                        spin_up_dn_list.append((a,b))
            for a,b in spin_up_dn_list:
                a_, b_ = get_opposite_spin(a), get_opposite_spin(b)
                if a_ > a or a < b:
                    spin_up_dn_list_diag.append((a,b))
                if a < b-1 or (b%2 !=0 and b-a ==1):
                    spin_up_dn_list_offdiag.append((a,b))
            
            for key in key_list:
                a, b, p = key.orbital_from, key.orbital_to, key.p
                if a%2 == 0 and b%2 == 0: #spin down
                    continue
                elif a%2 == 1 and b%2 == 1: #spin up
                    if a <= b and (np.sum(np.abs(np.array(p))) < 1e-5 and classname not in  ["Kinect", "Onsite"]):
                        continue
                elif (a-b)%2 == 1: #spin up - spin down
                    if (a,b) in spin_up_dn_list_diag and np.sum(np.abs(np.array(p))) < 1e-5:
                        continue
                    if (a,b) in spin_up_dn_list_offdiag and np.sum(np.abs(np.array(p))) > 1e-5:
                        continue
                # print(f"a = {a}, b = {b}, p = {p}")
                key_list_new.append(key)
            return key_list_new
    
    return key_list

def make_hashable(obj: Any) -> Any:
    """
    将对象转换为可哈希的版本。
    - 对于 list 和 tuple，递归将每个元素转换为 tuple。
    - 对于 dict，将其转换为 sorted 的 (key, value) tuple。
    - 对于 numpy 数组，使用 .tobytes()（也可以用 tuple(obj.tolist())）
    - 其它类型保持不变。
    """
    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, np.ndarray):
        # 使用 tobytes() 保证数组的位表示是唯一的  
        return obj.tobytes()
    elif isinstance(obj, ContinuumTerm):
        # 利用 term 的 key，该 key 需是可哈希的（比如 @dataclass(frozen=True)）
        return make_hashable(obj.key)
    # elif isinstance(obj, int):
    #     return obj
    # elif isinstance(obj, float):
    #     return obj
    # #如果是None，直接返回
    # elif obj is None:
    #     return 0
    # elif isinstance(obj, str):
    #     return obj
    # elif isinstance(obj, complex):
    #     return (obj.real, obj.imag)
    # elif hasattr(obj, '__hash__'):
    #     # 如果对象有 __hash__ 方法，则直接返回其哈希值
    #     return hash(obj)
    else:
        return obj
        raise TypeError(f"Unsupported type for hashing: {type(obj)}")

def get_function_hash(fn: Callable) -> int:
    try:
        closure = fn.__closure__
        if closure is not None:
            closure_values = tuple(make_hashable(c.cell_contents) for c in closure)
        else:
            closure_values = None
        defaults = make_hashable(fn.__defaults__)
        return hash((fn.__code__.co_code, defaults, closure_values))
    except AttributeError:
        # 如果函数没有 __code__ 属性，则退化处理
        return id(fn)
    
def truncate_indices(Q_set1: np.ndarray, Q_set2: np.ndarray, n_orb1: int, n_orb2: int, cutoff_shells: int, spin: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    按“圈数”对平面波基做截断。

    参数
    ----
    Q_sets : list of (n_Q_i, 2) ndarray  
        每层的 Q 向量列表，Q_sets[i].shape == (n_Q_i, 2)  
    orbitals_per_layer : list of int  
        每层的轨道数 len(orbitals_per_layer)==len(Q_sets)  
    cutoff_shells : int  
        截断的圈数（包含中心的第 0 圈），例如 cutoff_shells=3 表示保留半径最小的 4 个壳：第 0 圈 + 前 3 圈  
    spin : bool  
        是否包含自旋。如果 True，则每个轨道先放完所有 up 的 Q，再放所有 down 的 Q  

    返回
    ----
    keep_idx, remove_idx : ndarray of int  
        分别为需要保留和平面波截断后丢弃的全局基函数下标  
    """
    keep = []
    remove = []
    offset = 0
    Q_sets = [Q_set1, Q_set2]
    orbitals_per_layer = [n_orb1, n_orb2]

    for Q, n_orb in zip(Q_sets, orbitals_per_layer):
        # 计算这一层各 Q 的径向距离，找到“壳”半径
        r = np.linalg.norm(Q, axis=1)
        shells = np.sort(np.unique(np.round(r, 6)))
        # 阈值半径：第 cutoff_shells 个半径（如果索引越界，就取最大壳）
        if cutoff_shells < len(shells):
            r_thresh = shells[cutoff_shells]
        else:
            r_thresh = shells[-1]
        mask = r <= r_thresh+1e-5  # True 表示保留

        n_Q = Q.shape[0]
        for orb in range(n_orb):
            if spin:
                # up 态
                for iq in range(n_Q):
                    idx = offset + iq
                    (keep if mask[iq] else remove).append(idx)
                offset += n_Q
                # down 态
                for iq in range(n_Q):
                    idx = offset + iq
                    (keep if mask[iq] else remove).append(idx)
                offset += n_Q
            else:
                # 无自旋
                for iq in range(n_Q):
                    idx = offset + iq
                    (keep if mask[iq] else remove).append(idx)
                offset += n_Q

    return np.array(keep, dtype=int), np.array(remove, dtype=int)


def truncate_Q_indices(Q_set1: np.ndarray, cutoff_shells: int) -> tuple[np.ndarray, np.ndarray]:
    """
    按“圈数”对平面波基做截断。

    参数
    ----
    Q_set1 : (n_Q1, 2) ndarray
        第一层的 Q 向量列表，Q_set1.shape == (n_Q1, 2)
    cutoff_shells : int  
        截断的圈数（包含中心的第 0 圈），例如 cutoff_shells=3 表示保留半径最小的 4 个壳：第 0 圈 + 前 3 圈  

    返回
    ----
    keep1_indices, remove1_indices : ndarray of int  
        第一层需要保留和平面波截断后丢弃的全局基函数下标
    """
    keep1 = []
    remove1 = []
    
    # 计算第一层各 Q 的径向距离，找到“壳”半径
    r1 = np.linalg.norm(Q_set1, axis=1)
    shells1 = np.sort(np.unique(np.round(r1, 6)))
    # 阈值半径：第 cutoff_shells 个半径（如果索引越界，就取最大壳）
    if cutoff_shells < len(shells1):
        r_thresh1 = shells1[cutoff_shells]
    else:
        r_thresh1 = shells1[-1]
    mask1 = r1 <= r_thresh1+1e-5  # True 表示保留


    n_Q1 = Q_set1.shape[0]

    for iq in range(n_Q1):
        idx = iq
        (keep1 if mask1[iq] else remove1).append(idx)



    return np.array(keep1, dtype=int), np.array(remove1, dtype=int)



# =============================================================================
# 1. 数据结构定义
# =============================================================================

@dataclass(frozen=True)
class ContinuumTermKey:
    """
    唯一标识连续模型中一项 term 的指标

    属性：
      Mz, Mz_star: 多项式阶数（满足 Mz+Mz_star <= max_order）
      layer_from, layer_to: 层号（1 或 2）
      orbital_from, orbital_to: 轨道编号（从1开始）
      p: 跃迁动量，以 tuple 表示，如 (px, py)
    """
    Mz: int
    Mz_star: int
    layer_from: int
    layer_to: int
    orbital_from: int
    orbital_to: int
    p: Tuple[float, float]
    
    def __post_init__(self):
        # 但 dataclass(frozen=True) 下不能直接赋值；可以用 object.__setattr__
        object.__setattr__(self, 'p', (float(self.p[0]), float(self.p[1])))

@dataclass
class ContinuumTerm:
    """
    表示连续模型中的一项 term

    属性：
      key: 唯一标识 term 的指标
      Y_basis: 一个函数，输入 k 返回基函数矩阵（维度由各层 Q 数和轨道数决定）
      r_value_real, r_value_imag: 分别对应原基和 i×原基的系数（待求解）
      active: 正交化后标记该项是否为线性无关
      tag: "Kinect", "intra", "inter"，用于区分 onsite（Kinect且Mz=Mz_star=0）与耦合项
      symmetry_ops: 对称操作列表，每个元素为字典，如 {"name": "C3z", "params": 1}
    """
    key: ContinuumTermKey
    Y_basis: Callable[[np.ndarray], np.ndarray]
    r_value_real: Any = 0
    r_value_imag: Any = 0
    active: bool = False
    tag: str = "intra"
    symmetry_ops: List[Dict[str, Any]] = field(default_factory=list)

class ContinuumModel:
    """
    存储所有 term 的集合，并提供组装连续模型哈密顿量的方法。

    组装公式：
       H_cont(k) = Σ_{term active} [ r_value_real * Y_symm(k) + r_value_imag * (i*Y_symm(k)) ]
    """
    def __init__(self):
        self.terms: Dict[ContinuumTermKey, ContinuumTerm] = {}
    
    def add_term(self, key: ContinuumTermKey, Y_basis: Callable[[np.ndarray], np.ndarray],
                 tag: str = "intra", symmetry_ops: List[Dict[str, Any]] = None):
        if symmetry_ops is None:
            symmetry_ops = []
        if key in self.terms:
            print(f"Warning: Term {key} already exists, overwriting.")
        self.terms[key] = ContinuumTerm(key, Y_basis, tag=tag, symmetry_ops=symmetry_ops)
    
    def update_term_coefficients(self, key: ContinuumTermKey, r_real: complex, r_imag: complex):
        if key not in self.terms:
            raise ValueError(f"Term {key} does not exist.")
        self.terms[key].r_value_real = r_real
        self.terms[key].r_value_imag = r_imag

    def assemble_hamiltonian(self, k: np.ndarray, symmetry_gen: any, use_cache=True) -> np.ndarray:
        """
        对所有 active term 组装哈密顿量
        """
        H_cont = 0
        # 注意：这里组装时调用的是 builder 中封装的对称化方法（由 builder 实例调用）
        # 本方法仅简单地遍历各 term
        for term in self.terms.values():
            # print(f"Assembling term {term.key}... ativated: {term.active}")
            if term.active:
                if term.r_value_real is None or term.r_value_imag is None:
                    raise ValueError(f"Term {term.key} coefficients not assigned!")
                # 使用外部封装好的对称化函数（注意：此处仅作为占位，实际由 builder 调用）
                Y_symm, Y_symm_imag = ContinuumModelBuilder.symmetrize_Y_and_iY_basis_static(
                    term.Y_basis, k, term.symmetry_ops, symmetry_gen, term, use_cache
                )
                contribution = term.r_value_real * Y_symm + term.r_value_imag * Y_symm_imag
                if term.tag == "onsite":
                    print(f"Adding onsite energy {term.r_value_real} {term.r_value_imag} to term {term.key}")
                    print(f"Y basis for onsite term {term.key}:\n{Y_symm}")
                H_cont += contribution
        return H_cont

# =============================================================================
# 2. 对称操作生成封装：SymmetryGenerator
# =============================================================================

class SymmetryGenerator:
    """
    根据输入的 Q 数据生成对称操作矩阵，其维度与基函数矩阵一致。
    """

    def __init__(self, Qlayer1: np.ndarray, Qlayer2: np.ndarray, nlow_state: List[int]):
        self.Qlayer1 = Qlayer1
        self.Qlayer2 = Qlayer2
        self.nlow_state = nlow_state
        self.Q_set = np.concatenate([Qlayer1, Qlayer2], axis=0)

        # 预计算所有操作矩阵并缓存
        self.cached_operators = {}
        # self._cache_all_operators()

    def _cache_all_operators(self):
        """
        预计算并缓存所有的对称操作矩阵。
        这里考虑 C3z 有参数，因此我们缓存一个字典存储不同参数的 C3z 操作矩阵。
        """
        # 预缓存C3z矩阵（假设params为0，1，2）
        for params in range(3):
            self.cached_operators[f'C3z_{params}'] = self.get_C3z_operator(params)
        
        # 计算C2yT矩阵
        self.cached_operators['C2yT'] = self.get_C2yT_operator()
        # 计算C2zT矩阵
        self.cached_operators['C2zT'] = self.get_C2zT_operator()

    def rotation_matrix(self, theta: float) -> np.ndarray:
        """生成二维旋转矩阵"""
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])

    def get_C3z_operator(self, params: int) -> np.ndarray:
        """
        生成 C3z 对称操作的投影矩阵（示例代码，维度与各层 Q 数和轨道数匹配）
        这里利用输入的 Qlayer 与 nlow_state 生成 block_diag 矩阵
        """
        q1norm = np.max(np.linalg.norm(self.Q_set, axis=1)) - np.min(np.linalg.norm(self.Q_set, axis=1))
        value = np.exp(-1j*np.pi/3)
        value = 1
        C3_matrix = []
        for i in range(2):
            Qlayer = self.Qlayer1 if i == 0 else self.Qlayer2
            num_low_orb = self.nlow_state[i]
            for j in range(num_low_orb):
                if j%2 == 0:
                    value = np.exp(1j*np.pi/3)
                    value = 1
                    value = np.exp(1j*np.pi/3)
                    
                    # value = np.exp(2j*np.pi/3)
                else:
                    value = np.exp(1j*np.pi/3)
                    value = -1
                    value = np.exp(1j*np.pi/3)
                    
                    # value = np.exp(2j*np.pi/3)
                matrix = np.array([
                    [value if np.linalg.norm(ii - self.rotation_matrix(np.deg2rad(120)) @ jj) < q1norm/60 else 0
                     for jj in Qlayer]
                    for ii in Qlayer
                ])
                C3_matrix.append(matrix)
        C3_proj_matrix = scipy.linalg.block_diag(*C3_matrix)
        # C3_temp = np.load("/data/work/zy/software/TAPW_tmdc/dft_relax_from_mlff/3.48_same/2soc/Q_shell_7/band_data/C3_matrix.npy")
        # if params == 2:
        #     C3_temp = C3_temp @ C3_temp
        # if params == 0:
        #     C3_temp = np.eye(C3_temp.shape[0], dtype=complex)
        # if np.sum(np.abs(C3_proj_matrix@C3_proj_matrix - C3_proj_matrix.T)) > 1e-8:
        # if np.sum(np.abs(C3_proj_matrix - C3_temp.T)) > 1e-8:
        #     raise ValueError("C3z operator not orthogonal.")
        if params == 2:
            C3_proj_matrix = C3_proj_matrix @ C3_proj_matrix
        if params == -1:
            C3_proj_matrix = np.linalg.inv(C3_proj_matrix)
        if params == -2:
            C3_proj_matrix = np.linalg.inv(C3_proj_matrix) @ np.linalg.inv(C3_proj_matrix)
        return C3_proj_matrix

    def get_time_reversal_matrix(self) -> np.ndarray:
        """
        构造仅含自旋部分的时间反演酉算符矩阵 (i * sigma_y)。
        
        假设每个层对轨道与自旋的排列顺序是：
          - 对第 n 个轨道：
            先遍历该轨道的 “down” 自旋在所有 Q 的分量，
            再遍历该轨道的 “up” 自旋在所有 Q 的分量，
          - 然后再切换到 (n+1)-th 轨道，重复上述 down->up。

        记法：对第 i 层，
          nQ = len(Qlayer_i),
          num_low_orb = self.nlow_state[i]
          注意：这里的 nlow_state 在本脚本中约定为“包含自旋的低能态数”，
               即 num_low_orb = 2 * (physical_orbital_count)。
          则该层总维度为 dim_layer = nQ * num_low_orb。

        注意：这里只构造 spin-space 上的 i*sigma_y 块，
             未包含复共轭 K，也未做 Q->-Q 的交换。

        Returns
        -------
        TR_proj_matrix : np.ndarray
            时间反演(自旋部分)在整个多层空间的投影矩阵（分块对角拼接）。
        """

        # 先写好 i*sigma_y 在基 (down, up) 下的 2x2 矩阵：
        #   i*sigma_y = [[0, -1],
        #                [1,  0]]
        # 表示下->-上, 上->下
        spin_block = np.array([
            [0, -1],
            [1,  0]
        ], dtype=complex)

        # 存储每一层的矩阵块
        T_blocks = []

        # 两层循环（如需更多层可在此扩展）
        for i in range(2):
            if i == 0:
                spin_block = np.array([
                    [0, -1],
                    [1,  0]
                ], dtype=complex)
            else:
                spin_block = -np.array([
                    [0, 1],
                    [-1,  0]
                ], dtype=complex)
            Qlayer = self.Qlayer1 if i == 0 else self.Qlayer2
            nQ = len(Qlayer)
            num_low_orb = self.nlow_state[i]

            # 该层总维度 = (包含自旋的) num_low_orb * nQ
            dim_layer = int(num_low_orb) * nQ
            T_matrix_layer = np.zeros((dim_layer, dim_layer), dtype=complex)

            # 定义一个 index 函数，用来返回 (orb, spin, q) 在矩阵里的行列号
            # 这里 spin=0 表示 down, spin=1 表示 up
            # “先遍历该轨道 down 在所有Q, 再该轨道 up 在所有Q”，然后下个轨道
            def idx(orb, spin, q):
                # 每个轨道有 2*nQ 维度 (down block + up block)
                # orb_offset = orb*(2*nQ)
                # spin_offset = spin*(nQ)
                # return orb_offset + spin_offset + q
                return orb*(2*nQ) + spin*nQ + q

            # 在该层内部构造时间反演(自旋部分)的耦合
            for orb_i in range(int(num_low_orb) // 2):
                for q_i in range(nQ):
                    for q_j in range(nQ):
                        if np.sum(np.abs(Qlayer[q_i] + Qlayer[q_j])) < 1e-8:
                            # (down, up) => 用 spin_block 表示
                            row_down = idx(orb_i, 0, q_i)  # down
                            col_down = row_down
                            row_up = idx(orb_i, 1, q_j)    # up
                            col_up = row_up

                            # 根据 2x2 子矩阵 spin_block 填充：
                            # spin_block[0,0]  ->  (down, down)
                            # spin_block[0,1]  ->  (down, up)
                            # spin_block[1,0]  ->  (up, down)
                            # spin_block[1,1]  ->  (up, up)
                            T_matrix_layer[row_down, col_down] = spin_block[0, 0]
                            T_matrix_layer[row_down, col_up]   = spin_block[0, 1]
                            T_matrix_layer[row_up,   col_down] = spin_block[1, 0]
                            T_matrix_layer[row_up,   col_up]   = spin_block[1, 1]

            T_blocks.append(T_matrix_layer)

        # 分块对角拼接两层
        TR_proj_matrix = scipy.linalg.block_diag(*T_blocks)

        return TR_proj_matrix
    
    def get_C2yT_operator(self) -> np.ndarray:
        """
        生成 C2yT 对称操作的投影矩阵
        这里利用反射矩阵 R_y = diag(1,-1)
        """
        Qset = self.Q_set
        q1norm = np.min(np.linalg.norm(Qset, axis=1))
        mat = np.zeros((len(Qset), len(Qset)), dtype=complex)
        if self.nlow_state[0] != self.nlow_state[1] or len(self.Qlayer1) != len(self.Qlayer2) or self.nlow_state[0] != 1:
            raise ValueError("Different number of low energy states or Q points or not 1 low energy state per layer. Not supported C2yT.")
        
        for i in range(2):
            Qlayer_i = self.Qlayer1 if i == 0 else self.Qlayer2
            for j in range(2):
                Qlayer_j = self.Qlayer1 if j == 0 else self.Qlayer2
                R_y = np.array([[1,0],[0,-1]])
                for ii in range(len(Qlayer_i)):
                    for jj in range(len(Qlayer_j)):
                        if np.linalg.norm(Qlayer_i[ii] - R_y @ Qlayer_j[jj]) < q1norm/10:
                            mat[ii + i*len(Qlayer_i), jj + j*len(Qlayer_j)] = 1
        C2yT_proj = mat.T
        return C2yT_proj

    def get_C2zT_operator(self) -> np.ndarray:
        """
        生成 C2zT 对称操作的投影矩阵
        这里简单取负单位阵作为示例（占位符），但必须保证维度与当前基底一致：
        对第 i 层，基底维度为 len(Qlayer_i) * nlow_state[i]。
        """
        C2zT_matrices = []
        for i in range(2):
            Qlayer = self.Qlayer1 if i == 0 else self.Qlayer2
            n = len(Qlayer)
            dim_layer = n * int(self.nlow_state[i])
            C2zT_matrices.append(-np.eye(dim_layer, dtype=complex))
        C2zT_proj = scipy.linalg.block_diag(*C2zT_matrices)
        return C2zT_proj


    def get_C2x_operator(self, qtol: float | None = None) -> np.ndarray:
        """
        构造 C2x（绕 x 轴 180°）的酉算符矩阵。

        作用规则（与上次说明一致）：
        • 交换两层（z→-z）；
        • 面内倒格矢/动量变换： (kx, ky) → (kx, -ky)；
        • 轨道对 (p-↑, p+↓) 做 σ_x 交换（不加 -i，相当于令 C2x^2 = I，便于数值检查）。

        基底顺序假定与 get_C3z_operator 一致：
        [ layer1: orb0(Qs), orb1(Qs), ..., layer2: orb0(Qs), orb1(Qs), ... ]，
        其中每个 “orbj(Qs)” 是一个大小 nQ 的子块（先固定轨道，再遍历该轨道的全部 Q）。

        参数
        ----
        qtol : float | None
            Q 匹配容差；默认使用 (max|Q|-min|Q|)/60（与 C3z 实现保持一致）。

        返回
        ----
        U_C2x : np.ndarray (complex)
            整个 2 层 × 轨道 × Q 空间上的 C2x 表示矩阵（酉矩阵，且近似满足 U^2 = I）。
        """
        import numpy as np
        import scipy.linalg

        # ---- 基本量与容差 ----
        Q1 = np.asarray(self.Qlayer1, dtype=float)
        Q2 = np.asarray(self.Qlayer2, dtype=float)
        nQ1, nQ2 = len(Q1), len(Q2)
        m1, m2 = int(self.nlow_state[0]), int(self.nlow_state[1])

        if m1 != m2:
            raise ValueError(f"两层的 num_low_orb 不一致: {m1} vs {m2}")
        if nQ1 != nQ2:
            raise ValueError(f"两层的 Q 数不一致: {nQ1} vs {nQ2}")

        m, nQ = m1, nQ1

        if qtol is None:
            qset = getattr(self, "Q_set", None)
            if qset is None:
                qnorms = np.linalg.norm(np.vstack([Q1, Q2]), axis=1) if (nQ1 + nQ2) else np.array([0.0])
            else:
                qnorms = np.linalg.norm(np.asarray(qset, dtype=float), axis=1)
            qtol = (np.max(qnorms) - np.min(qnorms)) / 60.0 if len(qnorms) else 1e-12

        # 平面上的 C2x: (kx, ky) → (kx, -ky)
        C2x_inplane = np.array([[1.0, 0.0],
                                [0.0, -1.0]], dtype=float)

        # ---- 构造层间 Q 的置换矩阵：P12 把 layer2 的 Q 旋到 layer1 ----
        def build_perm(Q_src, Q_tgt, A, tol):
            P = np.zeros((len(Q_src), len(Q_tgt)), dtype=complex)
            for i, qi in enumerate(Q_src):
                Aq = (A @ Q_tgt.T).T  # 所有目标一次性变换
                d = np.linalg.norm(qi - Aq, axis=1)
                # 小于阈值则认为匹配；允许多对一时取最接近者
                if np.any(d < tol):
                    j = int(np.argmin(d))
                    P[i, j] = 1.0
            return P

        P12 = build_perm(Q1, Q2, C2x_inplane, qtol)  # map layer2 → layer1

        # ---- 轨道内部的 σ_x 交换（按相邻成对：0↔1, 2↔3, ...）----
        if m % 2 != 0:
            raise ValueError(f"期望每层的轨道数为偶数（成对交换），当前 m={m}")
        OrbX = np.zeros((m, m), dtype=complex)
        for p in range(0, m, 2):
            OrbX[p, p+1] = -1.0
            OrbX[p+1, p] = 1.0

        # 层间映射块：M12 = OrbX ⊗ P12
        M12 = np.kron(OrbX, P12)      # shape: (m*nQ, m*nQ)
        M21 = -M12.conj().T            # 保证酉性；令 U = [[0, M12],[M21, 0]]

        Z = np.zeros((m*nQ, m*nQ), dtype=complex)
        # top = np.hstack([Z,   M12])
        # bot = np.hstack([M21, Z  ])
        # U_C2x = np.vstack([top, bot])
        U_C2x = np.block([[Z, M12], 
                          [M21, Z]])

        # （可选）数值自检：U^†U≈I, U^2≈I
        # I_full = np.eye(U_C2x.shape[0], dtype=complex)
        # assert np.allclose(U_C2x.conj().T @ U_C2x, I_full, atol=1e-10)
        # assert np.allclose(U_C2x @ U_C2x, I_full, atol=1e-10)

        return U_C2x


    def get_operator(self, name: str, params: Any) -> np.ndarray:
        """
        根据对称操作名称返回变换后的 k（此处保持不变）和操作矩阵 D。
        根据名称调用相应函数。
        """
        # 检查缓存中是否存在对应的操作矩阵
        if name == 'C3z':
            operator_name = f'C3z_{params}'  # 对于 C3z 操作，使用参数来区分
        else:
            operator_name = name
        
        if operator_name not in self.cached_operators:
            if name == "C3z":
                D = self.get_C3z_operator(params)
            elif name == "C2yT":
                D = self.get_C2yT_operator()
            elif name == "C2zT":
                D = self.get_C2zT_operator()
            elif name == "TR":
                D = self.get_time_reversal_matrix()
            elif name == "C2x":
                D = self.get_C2x_operator()
            else:
                raise ValueError(f"Unknown symmetry operation: {name}")
            self.cached_operators[operator_name] = D
        return self.cached_operators[operator_name]

# =============================================================================
# 3. 模型构建封装：ContinuumModelBuilder
# =============================================================================
class ContinuumModelBuilder:
    """
    封装连续模型构建、基函数生成、对称化、正交化与系数提取等接口，
    对外只暴露 build_terms()、compute_coefficients() 与 assemble_hamiltonian() 等接口。

    构造时需要传入：
      Q_set1, Q_set2: 两层的 Q 点（二维数组）
      n_orb1, n_orb2: 两层轨道数
      bM1, bM2: 倒格基矢（二维向量）
      intra_harmonics_map, inter_harmonics_map: 跃迁 harmonic 映射（字典）
      max_order: dict，给出 "Kinect","intra","inter" 对应的 Mz+Mz_star 最大阶数
      symmetry_gen: 一个 SymmetryGenerator 实例（根据 Q 数据生成对称操作矩阵）
    """
    _SYMMETRIZE_GLOBAL_CACHE = SYMMETRIZE_GLOBAL_CACHE
    _SYMMETRIZE_MONOMIAL_OP_CACHE = SYMMETRIZE_MONOMIAL_OP_CACHE
    _SYMMETRIZE_MONOMIAL_OP_VALIDATED = SYMMETRIZE_MONOMIAL_OP_VALIDATED
    _SYMMETRIZE_COMPOSED_OP_CACHE = SYMMETRIZE_COMPOSED_OP_CACHE
    _SYMMETRIZE_COMPOSED_OP_VALIDATED = SYMMETRIZE_COMPOSED_OP_VALIDATED
    _SYMMETRIZE_ORBIT_CACHE: Dict[Tuple[int, Tuple[float, float], Tuple[Tuple[str, Any], ...]], Any] = {}
    _KZ_POW_CACHE: Dict[Tuple[int, Tuple[float, float]], np.ndarray] = {}
    _SYMM_ANTIUNITARY_OPS = frozenset({"TR", "C2yT", "C2zT"})
    _SYMM_UNITARY_OPS = frozenset({"C2x", "C2y"})
    _SYMM_VALIDATE_MONOMIAL = True
    _SYMM_VALIDATE_SPARSE = True
    _SYMM_USE_SPARSE_BASIS = True
    _SYMM_SPARSE_VALIDATED = False
    
    def __init__(self, Q_set1: np.ndarray, Q_set2: np.ndarray,
                 n_orb1: int, n_orb2: int,
                 bM1: np.ndarray, bM2: np.ndarray,
                 intra_harmonics_map: Dict[int, np.ndarray],
                 inter_harmonics_map: Dict[int, np.ndarray],
                 max_order: Dict[str, int],
                 symmetry_gen: SymmetryGenerator,
                 symmetry_map: Dict[str, List[Dict[str, Any]]] = None):
        self.Q_set1 = Q_set1
        self.Q_set2 = Q_set2
        self.n_orb1 = n_orb1
        self.n_orb2 = n_orb2
        self.bM1 = bM1
        self.bM2 = bM2
        self.intra_harmonics_map = intra_harmonics_map
        self.inter_harmonics_map = inter_harmonics_map
        self.max_order = max_order
        self.symmetry_gen = symmetry_gen
        # 如果没有传入 symmetry_map，则使用默认设置
        self.symmetry_map = symmetry_map if symmetry_map is not None else {
            "Onsite": [{"name": "C3z", "params": 1}],
            "Kinect": [{"name": "C3z", "params": 1}, {"name": "C3z", "params": 2},
                       {"name": "C2yT"}, {"name": "C2zT"}],
            "intra":  [{"name": "C3z", "params": 1}, {"name": "C3z", "params": 2},
                       {"name": "C2yT"}, {"name": "C2zT"}],
            "inter":  [{"name": "C3z", "params": 1}, {"name": "C3z", "params": 2},
                       {"name": "C2yT"}, {"name": "C2zT"}]
        }
        self.model = ContinuumModel()
    
    @staticmethod
    def make_Y_basis_function_(key: ContinuumTermKey,
                              Q_set1: np.ndarray, Q_set2: np.ndarray,
                              n_orb1: int, n_orb2: int, tol: float = 1e-5) -> Callable[[np.ndarray], np.ndarray]:
        """
        根据 term 的 key 与各层 Q 数据生成 Y_basis 函数

        实现公式：
          [Y(k)]_{row,col} = δ(row∈layer1,a1) δ(col∈layer2,a2)
                             × (k-Q)_z^{Mz} (k-Q)_z^*^{Mz_star}
                             ，当 Q' 满足 Q = Q'+p（tol 容差）时取值，否则 0
        注意：内部使用 get_global_index() 确保排列顺序正确
        """
        p_vec = np.array(key.p)
        Mz = key.Mz
        Mz_star = key.Mz_star
        l1 = key.layer_from
        l2 = key.layer_to
        a1 = key.orbital_from - 1  # 转为 0 开始
        a2 = key.orbital_to - 1

        def Y_func(k: np.ndarray) -> np.ndarray:
            dim = Q_set1.shape[0] * n_orb1 + Q_set2.shape[0] * n_orb2
            Y = np.zeros((dim, dim), dtype=complex)
            # 选择对应层的 Q 集合
            if l1 == 1:
                Q_rows = Q_set1
            else:
                Q_rows = Q_set2
            if l2 == 1:
                Q_cols = Q_set1
            else:
                Q_cols = Q_set2
            # 遍历 Q_rows
            for i, Q in enumerate(Q_rows):
                row_index = ContinuumModelBuilder.get_global_index(l1, i, a1, Q_set1, Q_set2, n_orb1, n_orb2)
                k_minus_Q = k - Q
                kz = k_minus_Q[0] + 1j*k_minus_Q[1]
                kz_star = np.conjugate(kz)
                for j, Qp in enumerate(Q_cols):
                    if np.linalg.norm(Q - p_vec - Qp) < tol:
                        col_index = ContinuumModelBuilder.get_global_index(l2, j, a2, Q_set1, Q_set2, n_orb1, n_orb2)
                        Y[row_index, col_index] = (kz**Mz) * (kz_star**Mz_star)

            # if l1 == l2 and a1 == a2 and np.linalg.norm(p_vec) < tol:
            #     if l1 == 1:
            #         Q_rows = Q_set2
            #     else:
            #         Q_rows = Q_set1
            #     if l2 == 1:
            #         Q_cols = Q_set2
            #     else:
            #         Q_cols = Q_set1
            #     # 遍历 Q_rows
            #     for i, Q in enumerate(Q_rows):
            #         row_index = ContinuumModelBuilder.get_global_index(l1+1, i, a1, Q_set1, Q_set2, n_orb1, n_orb2)
            #         k_minus_Q = k - Q
            #         kz = k_minus_Q[0] + 1j*k_minus_Q[1]
            #         kz_star = np.conjugate(kz)
            #         for j, Qp in enumerate(Q_cols):
            #             if np.linalg.norm(Q - p_vec - Qp) < tol:
            #                 col_index = ContinuumModelBuilder.get_global_index(l2+1, j, a2, Q_set1, Q_set2, n_orb1, n_orb2)
            #                 Y[row_index, col_index] = (kz**Mz) * (kz_star**Mz_star)            
                        
            if l1 == l2 and a1 == a2 and Mz != Mz_star and np.linalg.norm(p_vec) < tol:
                return Y + Y.conj().T
            # elif l1 == l2 and a1 == a2 and np.linalg.norm(p_vec) > tol:
            #     return Y + Y.conj().T
            else:
                return Y
        return Y_func

    @staticmethod
    def make_Y_basis_function(key: ContinuumTermKey,
                          Q_set1: np.ndarray, Q_set2: np.ndarray,
                          n_orb1: int, n_orb2: int, tol: float = 1e-5) -> Callable[[np.ndarray], np.ndarray]:
        """
        根据 term 的 key 与各层 Q 数据生成 Y_basis 函数

        实现公式：
        [Y(k)]_{row,col} = δ(row∈layer1,a1) δ(col∈layer2,a2)
                            × (k-Q)_z^{Mz} (k-Q)_z^*^{Mz_star}
                            ，当 Q' 满足 Q = Q'+p（tol 容差）时取值，否则 0
        注意：内部使用 get_global_index() 确保排列顺序正确
        """
        p_vec = np.array(key.p)
        Mz = key.Mz
        Mz_star = key.Mz_star
        l1 = key.layer_from
        l2 = key.layer_to
        a1 = key.orbital_from - 1  # 转为 0 开始
        a2 = key.orbital_to - 1
        # 总维度与 Q 集合在 term 生命周期内不变，可在闭包外预计算
        dim = Q_set1.shape[0] * n_orb1 + Q_set2.shape[0] * n_orb2
        Q_rows = Q_set1 if l1 == 1 else Q_set2
        Q_cols = Q_set1 if l2 == 1 else Q_set2

        row_indices = np.array(
            [
                ContinuumModelBuilder.get_global_index(l1, i, a1, Q_set1, Q_set2, n_orb1, n_orb2)
                for i in range(Q_rows.shape[0])
            ],
            dtype=int,
        )
        col_indices = np.array(
            [
                ContinuumModelBuilder.get_global_index(l2, j, a2, Q_set1, Q_set2, n_orb1, n_orb2)
                for j in range(Q_cols.shape[0])
            ],
            dtype=int,
        )

        diff_Q = Q_rows[:, None, :] - p_vec - Q_cols[None, :, :]
        mask = np.linalg.norm(diff_Q, axis=2) < tol
        i_idx, j_idx = np.nonzero(mask)
        sparse_rows = row_indices[i_idx]
        sparse_cols = col_indices[j_idx]
        sparse_row_q_idx = i_idx.astype(int, copy=False)
        if sparse_rows.size:
            pairs = sparse_rows.astype(np.int64) * np.int64(dim) + sparse_cols.astype(np.int64)
            sparse_unique = np.unique(pairs).size == pairs.size
        else:
            sparse_unique = True

        p_norm = float(np.linalg.norm(p_vec))
        hermitize_in_basis = (l1 == l2 and a1 == a2 and Mz != Mz_star and p_norm < tol)
        # 该分支理论上只会发生在对角结构（p=0 且 Q_set 无重复）；若不满足则回退到稠密构造以保证严格等价
        can_sparse_hermitize = (not hermitize_in_basis) or (sparse_rows.size == 0 or np.all(sparse_rows == sparse_cols))

        def eval_sparse(k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            # 缓存每个 (Q_rows, k) 的 kz^m（m=0..max(Mz,Mz_star)），多 term 复用，避免重复做复幂
            k_key = (float(k[0]), float(k[1]))
            pow_key = (id(Q_rows), k_key)
            kz_pows = ContinuumModelBuilder._KZ_POW_CACHE.get(pow_key)
            needed = max(Mz, Mz_star)
            if kz_pows is None or kz_pows.shape[0] <= needed:
                diff = k - Q_rows
                kz = diff[:, 0] + 1j * diff[:, 1]
                kz_pows = np.empty((needed + 1, kz.shape[0]), dtype=complex)
                kz_pows[0] = 1.0
                if needed >= 1:
                    kz_pows[1] = kz
                    for m in range(2, needed + 1):
                        kz_pows[m] = kz_pows[m - 1] * kz
                ContinuumModelBuilder._KZ_POW_CACHE[pow_key] = kz_pows
            value = kz_pows[Mz] * np.conjugate(kz_pows[Mz_star])
            vals = value[sparse_row_q_idx]
            if hermitize_in_basis:
                # 对角情形：Y + Y† -> 2*Re(Y)
                vals = vals + np.conjugate(vals)
            return sparse_rows, sparse_cols, vals

        def Y_func(k: np.ndarray) -> np.ndarray:
            Y = np.zeros((dim, dim), dtype=complex)
            if sparse_rows.size:
                rows, cols, vals = eval_sparse(k)
                # 若存在重复索引（通常不会），+= 的高级索引会不安全；此处保守用 add.at
                np.add.at(Y, (rows, cols), vals)
            if hermitize_in_basis and not can_sparse_hermitize:
                # 极少数（非对角）情况：退回原始定义，严格实现 Y + Y†
                return Y + Y.conjugate().T
            return Y

        # 仅在结构允许且无需额外 Y+Y† 展开时，暴露稀疏评估接口供对称化快速路径使用
        if can_sparse_hermitize:
            Y_func.eval_sparse = eval_sparse  # type: ignore[attr-defined]
            Y_func._moire_sparse_dim = dim  # type: ignore[attr-defined]
            Y_func._moire_sparse_unique = sparse_unique  # type: ignore[attr-defined]
        else:
            Y_func._moire_sparse_dim = dim  # type: ignore[attr-defined]
        return Y_func
    
    @staticmethod
    def get_global_index(layer: int, q_index: int, orb: int,
                         Q_set1: np.ndarray, Q_set2: np.ndarray,
                         n_orb1: int, n_orb2: int) -> int:
        """
        根据层号、Q 点索引与轨道号返回全局矩阵索引
        """
        if layer == 1:
            return Q_set1.shape[0] * orb + q_index
        elif layer == 2:
            return Q_set1.shape[0] * n_orb1 + Q_set2.shape[0] * orb + q_index
        else:
            raise ValueError("Layer must be 1 or 2.")

    @staticmethod
    def _extract_monomial_matrix(op_matrix: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray] | None:
        """
        若 op_matrix 为 monomial matrix（每行/每列仅一个非零元），返回 (perm, vals)：
        - perm[i] = 第 i 行非零元所在列
        - vals[i] = op_matrix[i, perm[i]]
        否则返回 None（自动回退到稠密乘法，保证结果正确）。
        """
        if op_matrix.ndim != 2 or op_matrix.shape[0] != op_matrix.shape[1]:
            return None
        n = op_matrix.shape[0]
        abs_op = np.abs(op_matrix)
        mask = abs_op > tol
        if not np.all(np.sum(mask, axis=1) == 1) or not np.all(np.sum(mask, axis=0) == 1):
            return None
        perm = np.argmax(abs_op, axis=1).astype(int)
        vals = op_matrix[np.arange(n), perm]
        return perm, vals

    @staticmethod
    def _get_monomial_op(symmetry_gen: Any, op_name: str, param: Any, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray] | None:
        cache_key = (id(symmetry_gen), op_name, param)
        cache = ContinuumModelBuilder._SYMMETRIZE_MONOMIAL_OP_CACHE
        if cache_key in cache:
            return cache[cache_key]
        op_matrix = symmetry_gen.get_operator(op_name, param)
        mono = ContinuumModelBuilder._extract_monomial_matrix(op_matrix, tol=tol)
        cache[cache_key] = mono
        return mono

    @staticmethod
    def _validate_monomial_op_once(symmetry_gen: Any, op_name: str, param: Any, mono: tuple[np.ndarray, np.ndarray] | None) -> bool:
        if not ContinuumModelBuilder._SYMM_VALIDATE_MONOMIAL or mono is None:
            return True
        validate_key = (id(symmetry_gen), op_name, param)
        validated = ContinuumModelBuilder._SYMMETRIZE_MONOMIAL_OP_VALIDATED
        if validate_key in validated:
            return True

        perm, vals = mono
        n = perm.shape[0]
        rng = np.random.default_rng(0)
        Y0 = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))

        if op_name == "C3z":
            D = symmetry_gen.get_operator(op_name, param)
            D_inv = symmetry_gen.get_operator(op_name, -param)
            dense = D @ Y0 @ D_inv
            fast = vals[:, None] * Y0[perm, :]
            fast = fast[:, perm] * (1.0 / vals)[None, :]
        elif op_name in ContinuumModelBuilder._SYMM_ANTIUNITARY_OPS:
            D = symmetry_gen.get_operator(op_name, param)
            dense = D @ Y0.conj() @ D.conj().T
            fast = vals[:, None] * Y0.conj()[perm, :]
            fast = fast[:, perm] * np.conjugate(vals)[None, :]
        else:
            D = symmetry_gen.get_operator(op_name, param)
            dense = D @ Y0 @ D.conj().T
            fast = vals[:, None] * Y0[perm, :]
            fast = fast[:, perm] * np.conjugate(vals)[None, :]

        ok = np.allclose(fast, dense, atol=1e-10, rtol=0.0)
        if not ok:
            ContinuumModelBuilder._SYMMETRIZE_MONOMIAL_OP_CACHE[validate_key] = None
        validated.add(validate_key)
        return ok

    @staticmethod
    def _apply_symmetry_op_to_matrix(Y: np.ndarray, op_name: str, param: Any, symmetry_gen: Any) -> np.ndarray:
        if symmetry_gen is None:
            raise ValueError("symmetry_gen must be provided when sym_ops is non-empty.")

        if op_name == "C3z":
            mono = ContinuumModelBuilder._get_monomial_op(symmetry_gen, op_name, param)
            if mono is not None and ContinuumModelBuilder._validate_monomial_op_once(symmetry_gen, op_name, param, mono):
                perm, vals = mono
                inv_vals = 1.0 / vals
                Y = Y[np.ix_(perm, perm)]
                Y *= vals[:, None]
                Y *= inv_vals[None, :]
                return Y
            D = symmetry_gen.get_operator(op_name, param)
            D_inv = symmetry_gen.get_operator(op_name, -param)
            return D @ Y @ D_inv

        if op_name in ContinuumModelBuilder._SYMM_ANTIUNITARY_OPS:
            mono = ContinuumModelBuilder._get_monomial_op(symmetry_gen, op_name, param)
            if mono is not None and ContinuumModelBuilder._validate_monomial_op_once(symmetry_gen, op_name, param, mono):
                perm, vals = mono
                Y = Y[np.ix_(perm, perm)]
                np.conjugate(Y, out=Y)
                Y *= vals[:, None]
                Y *= np.conjugate(vals)[None, :]
                return Y
            D = symmetry_gen.get_operator(op_name, param)
            return D @ Y.conj() @ D.conj().T

        if op_name in ContinuumModelBuilder._SYMM_UNITARY_OPS:
            mono = ContinuumModelBuilder._get_monomial_op(symmetry_gen, op_name, param)
            if mono is not None and ContinuumModelBuilder._validate_monomial_op_once(symmetry_gen, op_name, param, mono):
                perm, vals = mono
                Y = Y[np.ix_(perm, perm)]
                Y *= vals[:, None]
                Y *= np.conjugate(vals)[None, :]
                return Y
            D = symmetry_gen.get_operator(op_name, param)
            return D @ Y @ D.conj().T

        raise ValueError(f"Unknown symmetry operation: {op_name}")

    @staticmethod
    def _get_composed_symmetry_action(
        symmetry_gen: Any, op_seq_applied: Tuple[Tuple[str, Any], ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
        """
        将一串对称操作（按其在 apply_symm 中的实际应用顺序）合成为一个 monomial operator：
        返回 (perm, vals, inv_vals, is_anti_total)，用于一次性实现
          Y -> U · (Y 或 Y*) · U^{-1}。

        若某一步无法走 monomial 快速路径，则返回 None（外层会回退到逐步应用，保证结果正确）。
        """
        cache_key = (id(symmetry_gen), op_seq_applied)
        cache = ContinuumModelBuilder._SYMMETRIZE_COMPOSED_OP_CACHE
        if cache_key in cache:
            return cache[cache_key]

        anti_ops = ContinuumModelBuilder._SYMM_ANTIUNITARY_OPS
        is_anti_total = False

        perm_total: np.ndarray | None = None
        vals_total: np.ndarray | None = None

        for op_name, param in op_seq_applied:
            is_anti_op = op_name in anti_ops
            mono = ContinuumModelBuilder._get_monomial_op(symmetry_gen, op_name, param)
            if mono is None or not ContinuumModelBuilder._validate_monomial_op_once(symmetry_gen, op_name, param, mono):
                cache[cache_key] = None
                return None

            perm_op, vals_op = mono
            if perm_total is None:
                n = perm_op.shape[0]
                perm_total = np.arange(n, dtype=int)
                vals_total = np.ones(n, dtype=complex)

            if is_anti_op:
                vals_total = np.conjugate(vals_total)

            perm_total = perm_total[perm_op]
            vals_total = vals_op * vals_total[perm_op]
            is_anti_total = (is_anti_total ^ is_anti_op)

        inv_vals_total = 1.0 / vals_total
        composed = (perm_total, vals_total, inv_vals_total, is_anti_total)
        cache[cache_key] = composed
        return composed

    @staticmethod
    def _validate_composed_symmetry_action_once(
        symmetry_gen: Any,
        op_seq_applied: Tuple[Tuple[str, Any], ...],
        composed: tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None,
    ) -> bool:
        if not ContinuumModelBuilder._SYMM_VALIDATE_MONOMIAL:
            return True
        validate_key = (id(symmetry_gen), op_seq_applied)
        validated = ContinuumModelBuilder._SYMMETRIZE_COMPOSED_OP_VALIDATED
        if validate_key in validated:
            return True
        validated.add(validate_key)

        if composed is None:
            return False

        perm, vals, inv_vals, is_anti_total = composed
        n = perm.shape[0]
        rng = np.random.default_rng(1)
        Y0 = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))

        # 顺序应用（与 apply_symm 一致：对矩阵逐步施加每个 op）
        Y_seq = Y0
        for op_name, param in op_seq_applied:
            Y_seq = ContinuumModelBuilder._apply_symmetry_op_to_matrix(Y_seq, op_name, param, symmetry_gen)

        # 合并后一次性应用
        Y_fast = Y0.conj() if is_anti_total else Y0
        Y_fast = vals[:, None] * Y_fast[perm, :]
        Y_fast = Y_fast[:, perm] * inv_vals[None, :]

        ok = np.allclose(Y_fast, Y_seq, atol=1e-10, rtol=0.0)
        if not ok:
            ContinuumModelBuilder._SYMMETRIZE_COMPOSED_OP_CACHE[validate_key] = None
        return ok

    @staticmethod
    def _generate_symmetry_orbit(base_k: np.ndarray, sym_ops: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[List[Tuple[str, Any]]]]:
        def rot2d(kvec, theta_deg):
            theta = np.deg2rad(theta_deg)
            return np.array([np.cos(theta) * kvec[0] - np.sin(theta) * kvec[1],
                             np.sin(theta) * kvec[0] + np.cos(theta) * kvec[1]])

        points = [base_k.copy()]
        op_seqs: List[List[Tuple[str, Any]]] = [[]]

        op_actions = {
            "C3z": lambda kk, n: rot2d(kk, -120 * n),
            "C2yT": lambda kk: np.array([kk[0], -kk[1]]),
            "C2x": lambda kk: np.array([kk[0], -kk[1]]),
            "C2zT": lambda kk: -kk,
            "TR": lambda kk: -kk,
        }

        for op in sym_ops:
            op_name = op["name"]
            new_pts: List[np.ndarray] = []
            new_ops: List[List[Tuple[str, Any]]] = []
            for pt, seq in zip(points, op_seqs):
                if any(s[0] == op_name for s in seq):
                    continue
                if op_name == "C3z":
                    for n in (1, 2):
                        new_pts.append(op_actions[op_name](pt, n))
                        new_ops.append(seq + [(op_name, n)])
                else:
                    new_pts.append(op_actions[op_name](pt))
                    new_ops.append(seq + [(op_name, None)])
            points += new_pts
            op_seqs += new_ops

        return points, op_seqs

    @staticmethod
    def _get_symmetry_orbit_actions_cached(
        k: np.ndarray, sym_ops: List[Dict[str, Any]], symmetry_gen: Any
    ) -> List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | None, Tuple[Tuple[str, Any], ...], bool]]:
        """
        缓存某个 (k, sym_ops) 下的 orbit 以及每个 orbit 元素的 composed monomial action（如可用）。

        返回列表元素：
        (kk, action, op_seq_applied, is_anti)
        - kk: orbit k 点
        - action: (perm, inv_perm, vals, inv_vals, is_anti_total) 或 None（需要逐步回退）
        - op_seq_applied: 逐步回退时用的操作序列（已按实际应用顺序排列，即 reversed(op_seq)）
        - is_anti: 该 orbit 元素对应的 antiunitary 总奇偶（用于 iY 的符号与统计）
        """
        k_key = tuple(float(x) for x in k)
        sym_ops_key = tuple((op["name"], op.get("params", None)) for op in sym_ops)
        cache_key = (id(symmetry_gen), k_key, sym_ops_key)
        cached = ContinuumModelBuilder._SYMMETRIZE_ORBIT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        symm_points, op_seqs = ContinuumModelBuilder._generate_symmetry_orbit(k, sym_ops)
        anti_ops = ContinuumModelBuilder._SYMM_ANTIUNITARY_OPS
        orbit_actions: List[
            Tuple[
                np.ndarray,
                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | None,
                Tuple[Tuple[str, Any], ...],
                bool,
            ]
        ] = []

        for kk, op_seq in zip(symm_points, op_seqs):
            if not op_seq:
                orbit_actions.append((kk, None, tuple(), False))
                continue

            op_seq_applied = tuple(reversed(op_seq))
            composed = ContinuumModelBuilder._get_composed_symmetry_action(symmetry_gen, op_seq_applied)
            if composed is not None and ContinuumModelBuilder._validate_composed_symmetry_action_once(symmetry_gen, op_seq_applied, composed):
                perm, vals, inv_vals, is_anti_total = composed
                inv_perm = np.empty_like(perm)
                inv_perm[perm] = np.arange(perm.shape[0], dtype=int)
                orbit_actions.append((kk, (perm, inv_perm, vals, inv_vals, is_anti_total), tuple(), bool(is_anti_total)))
            else:
                is_anti = (sum(1 for op_name, _ in op_seq if op_name in anti_ops) % 2) == 1
                orbit_actions.append((kk, None, op_seq_applied, bool(is_anti)))

        ContinuumModelBuilder._SYMMETRIZE_ORBIT_CACHE[cache_key] = orbit_actions
        return orbit_actions

    @staticmethod
    def symmetrize_Y_basis_static(Y_basis: Callable[[np.ndarray], np.ndarray],
                                  k: np.ndarray,
                                  sym_ops: List[Dict[str, Any]],
                                  symmetry_gen: Any = None,
                                  term = ContinuumTerm,
                                  use_cache=False) -> np.ndarray:
        """
        静态版本的对称化函数（当不需要调用 symmetry_gen 时可传 None）
        如果 symmetry_gen 不为 None，则调用 symmetry_gen.get_operator()。
        对称平均后 Hermitian 化。
        """
        if ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE is None:
            ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE = {}

        cache_key = None
        if use_cache:
            Y_id = get_function_hash(Y_basis)
            k_key = tuple(float(x) for x in k)
            sym_ops_key = tuple((op["name"], op.get("params", None)) for op in sym_ops)
            term_key = None
            if isinstance(term, ContinuumTerm) and term.key is not None:
                term_key = (
                    term.key.Mz, term.key.Mz_star,
                    term.key.layer_from, term.key.layer_to,
                    term.key.orbital_from, term.key.orbital_to,
                    term.key.p
                )
            cache_key = (id(symmetry_gen), Y_id, k_key, sym_ops_key, term_key)
            cached = ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE.get(cache_key)
            if cached is not None:
                return cached

        use_sparse = (
            ContinuumModelBuilder._SYMM_USE_SPARSE_BASIS
            and hasattr(Y_basis, "eval_sparse")
            and symmetry_gen is not None
        )
        dim = getattr(Y_basis, "_moire_sparse_dim", None)

        if sym_ops and symmetry_gen is None:
            raise ValueError("symmetry_gen must be provided when sym_ops is non-empty.")

        orbit_actions = (
            ContinuumModelBuilder._get_symmetry_orbit_actions_cached(k, sym_ops, symmetry_gen)
            if sym_ops
            else [(k, None, tuple(), False)]
        )

        Y_symm = np.zeros((dim, dim), dtype=complex) if (use_sparse and isinstance(dim, int)) else None
        for kk, action, op_seq_applied, _is_anti in orbit_actions:
            if use_sparse and Y_symm is not None and op_seq_applied == tuple():
                rows, cols, vals0 = Y_basis.eval_sparse(kk)
                if action is not None:
                    perm, inv_perm, vals, inv_vals, is_anti_total = action
                    rr = inv_perm[rows]
                    cc = inv_perm[cols]
                    vv = np.conjugate(vals0) if is_anti_total else vals0
                    vv = vv * vals[rr] * inv_vals[cc]
                    np.add.at(Y_symm, (rr, cc), vv)
                else:
                    np.add.at(Y_symm, (rows, cols), vals0)
                continue

            # 稠密回退路径（保持原始语义）
            Y_part = Y_basis(kk)
            if action is not None:
                perm, _inv_perm, vals, inv_vals, is_anti_total = action
                Y_part = Y_part[np.ix_(perm, perm)]
                if is_anti_total:
                    np.conjugate(Y_part, out=Y_part)
                Y_part *= vals[:, None]
                Y_part *= inv_vals[None, :]
            elif op_seq_applied:
                for op_name, param in op_seq_applied:
                    Y_part = ContinuumModelBuilder._apply_symmetry_op_to_matrix(Y_part, op_name, param, symmetry_gen)

            if Y_symm is None:
                Y_symm = np.zeros_like(Y_part)
            Y_symm += Y_part

        if not np.allclose(Y_symm, Y_symm.conjugate().T):
            Y_symm = (Y_symm + Y_symm.conjugate().T)

        if use_cache and cache_key is not None:
            ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE[cache_key] = Y_symm

        return Y_symm

    @staticmethod
    def symmetrize_Y_and_iY_basis_static(
        Y_basis: Callable[[np.ndarray], np.ndarray],
        k: np.ndarray,
        sym_ops: List[Dict[str, Any]],
        symmetry_gen: Any = None,
        term = ContinuumTerm,
        use_cache: bool = False,
        out_real: np.ndarray | None = None,
        out_imag: np.ndarray | None = None,
        out_anti: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        同时计算 S[Y] 与 S[iY]，避免对同一 term/k 重复做一遍对称化。

        对于 unitary g:   g(iY) = i·g(Y)
        对于 antiunitary g: g(iY) = -i·g(Y)   （因为 antiunitary 会做复共轭，i -> -i）
        因而：
          S[iY](k) = Σ_g s(g)·i·g(Y)(k),
        其中 s(g)=+1 (unitary), s(g)=-1 (antiunitary)。
        """
        if ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE is None:
            ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE = {}

        cache_key = None
        if use_cache:
            Y_id = get_function_hash(Y_basis)
            k_key = tuple(float(x) for x in k)
            sym_ops_key = tuple((op["name"], op.get("params", None)) for op in sym_ops)
            term_key = None
            if isinstance(term, ContinuumTerm) and term.key is not None:
                term_key = (
                    term.key.Mz, term.key.Mz_star,
                    term.key.layer_from, term.key.layer_to,
                    term.key.orbital_from, term.key.orbital_to,
                    term.key.p
                )
            cache_key = ("pair", id(symmetry_gen), Y_id, k_key, sym_ops_key, term_key)
            cached = ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE.get(cache_key)
            if cached is not None:
                return cached

        use_sparse = (
            ContinuumModelBuilder._SYMM_USE_SPARSE_BASIS
            and hasattr(Y_basis, "eval_sparse")
            and symmetry_gen is not None
        )
        dim = getattr(Y_basis, "_moire_sparse_dim", None)

        if sym_ops and symmetry_gen is None:
            raise ValueError("symmetry_gen must be provided when sym_ops is non-empty.")

        orbit_actions = (
            ContinuumModelBuilder._get_symmetry_orbit_actions_cached(k, sym_ops, symmetry_gen)
            if sym_ops
            else [(k, None, tuple(), False)]
        )

        def compute_pair(use_sparse_path: bool, allow_out: bool = True) -> Tuple[np.ndarray, np.ndarray]:
            use_out = (
                allow_out
                and (not use_cache)
                and out_real is not None
                and out_imag is not None
                and out_anti is not None
                and out_real.shape == out_imag.shape == out_anti.shape
                and out_real.ndim == 2
                and out_real.shape[0] == out_real.shape[1]
            )

            if use_out:
                Y_symm_local = out_real
                Y_anti_local = out_anti
                Y_symm_local.fill(0)
                Y_anti_local.fill(0)
            else:
                Y_symm_local = np.zeros((dim, dim), dtype=complex) if isinstance(dim, int) else None
                Y_anti_local = np.zeros((dim, dim), dtype=complex) if isinstance(dim, int) else None

            for kk, action, op_seq_applied, is_anti in orbit_actions:
                if use_sparse_path and Y_symm_local is not None and op_seq_applied == tuple():
                    use_add_at = not getattr(Y_basis, "_moire_sparse_unique", True)
                    rows, cols, vals0 = Y_basis.eval_sparse(kk)
                    if action is not None:
                        perm, inv_perm, vals, inv_vals, is_anti_total = action
                        rr = inv_perm[rows]
                        cc = inv_perm[cols]
                        vv = np.conjugate(vals0) if is_anti_total else vals0
                        vv = vv * vals[rr] * inv_vals[cc]
                        if use_add_at:
                            np.add.at(Y_symm_local, (rr, cc), vv)
                        else:
                            Y_symm_local[rr, cc] += vv
                        if is_anti_total:
                            if use_add_at:
                                np.add.at(Y_anti_local, (rr, cc), vv)
                            else:
                                Y_anti_local[rr, cc] += vv
                    else:
                        if use_add_at:
                            np.add.at(Y_symm_local, (rows, cols), vals0)
                        else:
                            Y_symm_local[rows, cols] += vals0
                    continue

                Y_part = Y_basis(kk)
                is_anti_total = is_anti
                if action is not None:
                    perm, _inv_perm, vals, inv_vals, is_anti_total = action
                    Y_part = Y_part[np.ix_(perm, perm)]
                    if is_anti_total:
                        np.conjugate(Y_part, out=Y_part)
                    Y_part *= vals[:, None]
                    Y_part *= inv_vals[None, :]
                elif op_seq_applied:
                    for op_name, param in op_seq_applied:
                        Y_part = ContinuumModelBuilder._apply_symmetry_op_to_matrix(Y_part, op_name, param, symmetry_gen)

                if Y_symm_local is None:
                    Y_symm_local = np.zeros_like(Y_part)
                    Y_anti_local = np.zeros_like(Y_part)
                Y_symm_local += Y_part
                if is_anti_total:
                    Y_anti_local += Y_part

            if use_out:
                Y_symm_i_local = out_imag
                Y_symm_i_local[:] = Y_symm_local
            else:
                Y_symm_i_local = Y_symm_local.copy()
            Y_symm_i_local -= Y_anti_local
            Y_symm_i_local -= Y_anti_local
            Y_symm_i_local *= 1j

            needs_herm_real = not np.allclose(Y_symm_local, Y_symm_local.conjugate().T)
            needs_herm_imag = not np.allclose(Y_symm_i_local, Y_symm_i_local.conjugate().T)

            # 记录是否触发过 Hermitian 化：用于后续更快的“直接组装”路径（不影响默认计算结果）
            if isinstance(term, ContinuumTerm):
                if not hasattr(term, "_moire_needs_hermitize_real"):
                    term._moire_needs_hermitize_real = bool(needs_herm_real)
                elif bool(getattr(term, "_moire_needs_hermitize_real")) != bool(needs_herm_real):
                    term._moire_hermitize_flags_inconsistent = True
                if not hasattr(term, "_moire_needs_hermitize_imag"):
                    term._moire_needs_hermitize_imag = bool(needs_herm_imag)
                elif bool(getattr(term, "_moire_needs_hermitize_imag")) != bool(needs_herm_imag):
                    term._moire_hermitize_flags_inconsistent = True

            if needs_herm_real:
                if use_out:
                    Y_symm_local[:] = (Y_symm_local + Y_symm_local.conjugate().T)
                else:
                    Y_symm_local = (Y_symm_local + Y_symm_local.conjugate().T)
            if needs_herm_imag:
                if use_out:
                    Y_symm_i_local[:] = (Y_symm_i_local + Y_symm_i_local.conjugate().T)
                else:
                    Y_symm_i_local = (Y_symm_i_local + Y_symm_i_local.conjugate().T)

            return Y_symm_local, Y_symm_i_local

        if use_sparse and ContinuumModelBuilder._SYMM_VALIDATE_SPARSE and not ContinuumModelBuilder._SYMM_SPARSE_VALIDATED and sym_ops:
            Y_sparse, Y_sparse_i = compute_pair(True, allow_out=False)
            Y_dense, Y_dense_i = compute_pair(False, allow_out=False)
            ok = np.allclose(Y_sparse, Y_dense, atol=1e-10, rtol=0.0) and np.allclose(Y_sparse_i, Y_dense_i, atol=1e-10, rtol=0.0)
            if not ok:
                ContinuumModelBuilder._SYMM_USE_SPARSE_BASIS = False
                ContinuumModelBuilder._SYMM_SPARSE_VALIDATED = True
                Y_symm, Y_symm_i = Y_dense, Y_dense_i
            else:
                ContinuumModelBuilder._SYMM_SPARSE_VALIDATED = True
                Y_symm, Y_symm_i = Y_sparse, Y_sparse_i
        else:
            Y_symm, Y_symm_i = compute_pair(use_sparse, allow_out=True)

        if use_cache and cache_key is not None:
            ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE[cache_key] = (Y_symm, Y_symm_i)

        return Y_symm, Y_symm_i

    @staticmethod
    def add_symmetrized_term_to_matrix_static(
        H_out: np.ndarray,
        Y_basis: Callable[[np.ndarray], np.ndarray],
        k: np.ndarray,
        sym_ops: List[Dict[str, Any]],
        r_value_real: float,
        r_value_imag: float,
        symmetry_gen: Any = None,
        term=ContinuumTerm,
    ) -> None:
        """
        将某个 term 在 k 点的贡献直接累加到 H_out，避免先构造 (Y_symm, Y_symm_i) 两个稠密矩阵。

        严格等价于：
          Y_symm, Y_symm_i = symmetrize_Y_and_iY_basis_static(...)
          H_out += r_real * Y_symm + r_imag * Y_symm_i

        推导：对每个 orbit 元素 g，
          unitary:   g(iY)= i g(Y)
          antiunitary: g(iY)= -i g(Y)
        因而每个 g 的权重为：
          w_g = r_real + i * s(g) * r_imag,  s(g)=+1(unitary), -1(antiunitary).
        """
        if r_value_real == 0.0 and r_value_imag == 0.0:
            return
        if sym_ops and symmetry_gen is None:
            raise ValueError("symmetry_gen must be provided when sym_ops is non-empty.")

        # Hermitian 化语义必须与 symmetrize_Y_and_iY_basis_static 完全一致：
        # - 对 Y_symm 与 Y_symm_i 分别做一次 “若非 Hermitian 则 Y<-Y+Y†” 的条件修正。
        #   这一步的触发与否通常只取决于 term/key（与 k 无关）；若检测到不同 k 下不一致，
        #   则该 term 回退到稠密路径（保证严格等价）。
        needs_herm_real = False
        needs_herm_imag = False
        inconsistent = False
        if isinstance(term, ContinuumTerm):
            inconsistent = bool(getattr(term, "_moire_hermitize_flags_inconsistent", False))
            if hasattr(term, "_moire_needs_hermitize_real"):
                needs_herm_real = bool(getattr(term, "_moire_needs_hermitize_real"))
            if hasattr(term, "_moire_needs_hermitize_imag"):
                needs_herm_imag = bool(getattr(term, "_moire_needs_hermitize_imag"))

        if inconsistent or (isinstance(term, ContinuumTerm) and (not hasattr(term, "_moire_needs_hermitize_real") or not hasattr(term, "_moire_needs_hermitize_imag"))):
            # 保守回退：直接用原函数构造 (Y_symm, Y_symm_i) 再加到 H_out。
            # 该分支只在开发/异常情况下触发，不影响默认正确性。
            Y_symm, Y_symm_i = ContinuumModelBuilder.symmetrize_Y_and_iY_basis_static(
                Y_basis, k, sym_ops, symmetry_gen=symmetry_gen, term=term, use_cache=False
            )
            if r_value_real != 0.0:
                H_out += r_value_real * Y_symm
            if r_value_imag != 0.0:
                H_out += r_value_imag * Y_symm_i
            return

        use_sparse = (
            ContinuumModelBuilder._SYMM_USE_SPARSE_BASIS
            and hasattr(Y_basis, "eval_sparse")
            and symmetry_gen is not None
        )

        orbit_actions = (
            ContinuumModelBuilder._get_symmetry_orbit_actions_cached(k, sym_ops, symmetry_gen)
            if sym_ops
            else [(k, None, tuple(), False)]
        )

        for kk, action, op_seq_applied, is_anti in orbit_actions:
            if use_sparse and op_seq_applied == tuple():
                use_add_at = not getattr(Y_basis, "_moire_sparse_unique", True)
                rows, cols, vals0 = Y_basis.eval_sparse(kk)
                if action is not None:
                    perm, inv_perm, vals, inv_vals, is_anti_total = action
                    rr = inv_perm[rows]
                    cc = inv_perm[cols]
                    vv = np.conjugate(vals0) if is_anti_total else vals0
                    vv = vv * vals[rr] * inv_vals[cc]
                else:
                    rr, cc, vv = rows, cols, vals0
                    is_anti_total = False

                # base: r_real * S[Y] + r_imag * S[iY] 的逐 orbit 累加
                s = -1.0 if is_anti_total else 1.0
                weight = r_value_real + (1j * s) * r_value_imag
                if use_add_at:
                    np.add.at(H_out, (rr, cc), weight * vv)
                else:
                    H_out[rr, cc] += weight * vv

                # conditional hermitianization: add transpose contributions if needed
                if (needs_herm_real or needs_herm_imag) and (r_value_real != 0.0 or r_value_imag != 0.0):
                    wt = (r_value_real if needs_herm_real else 0.0) + ((-1j * s) * r_value_imag if needs_herm_imag else 0.0)
                    if wt != 0.0:
                        vvH = np.conjugate(vv)
                        if use_add_at:
                            np.add.at(H_out, (cc, rr), wt * vvH)
                        else:
                            H_out[cc, rr] += wt * vvH
                continue

            # 稠密回退路径（保证正确性；通常不会触发）
            Y_part = Y_basis(kk)
            is_anti_total = is_anti
            if action is not None:
                perm, _inv_perm, vals, inv_vals, is_anti_total = action
                Y_part = Y_part[np.ix_(perm, perm)]
                if is_anti_total:
                    np.conjugate(Y_part, out=Y_part)
                Y_part *= vals[:, None]
                Y_part *= inv_vals[None, :]
            elif op_seq_applied:
                for op_name, param in op_seq_applied:
                    Y_part = ContinuumModelBuilder._apply_symmetry_op_to_matrix(Y_part, op_name, param, symmetry_gen)

            s = -1.0 if is_anti_total else 1.0
            weight = r_value_real + (1j * s) * r_value_imag
            H_out += weight * Y_part

            if needs_herm_real or needs_herm_imag:
                wt = (r_value_real if needs_herm_real else 0.0) + ((-1j * s) * r_value_imag if needs_herm_imag else 0.0)
                if wt != 0.0:
                    H_out += wt * Y_part.conjugate().T

        # 与原逻辑一致：不在此处强制 Hermitian 化（原实现是在 Y_symm/Y_symm_i 层面做 condition-allclose 再修正）

    @staticmethod
    def symmetrize_Y_basis_static_(Y_basis: Callable[[np.ndarray], np.ndarray],
                                k: np.ndarray,
                                sym_ops: List[Dict[str, Any]],
                                symmetry_gen: Any = None,
                                term=ContinuumTerm) -> np.ndarray:
        """
        静态版本的对称化函数（当不需要调用 symmetry_gen 时可传 None）
        如果 symmetry_gen 不为 None，则调用 symmetry_gen.get_operator()。
        对称平均后 Hermitian 化。
        
        这里所有矩阵操作均使用 scipy.sparse 中的稀疏矩阵，最后转换为 np.array 返回。
        """
        
        def rot(k, theta):
            """二维旋转"""
            theta = np.deg2rad(theta)
            return np.array([np.cos(theta)*k[0] - np.sin(theta)*k[1],
                            np.sin(theta)*k[0] + np.cos(theta)*k[1]])
        
        def gen_symm_k(base_k: np.ndarray) -> Tuple[List[np.ndarray], List[List[Tuple[str, Any]]]]:
            """生成对称操作后的 k 点及对应的操作序列"""
            points = [base_k.copy()]
            op_seqs = [[]]
            
            # 定义各对称操作对应的 k 变换
            op_actions = {
                'C3z': lambda k, n: rot(k, -120*n),
                'C2yT': lambda k: np.array([k[0], -k[1]]),
                'C2x': lambda k: np.array([-k[0], k[1]]),
                'C2zT': lambda k: -k,
                'TR': lambda k: -k
            }
            
            for op in sym_ops:
                op_name = op["name"]
                new_pts, new_ops = [], []
                for pt, seq in zip(points, op_seqs):
                    # 避免重复应用相同操作
                    if any(s[0] == op_name for s in seq):
                        continue
                    if op_name == 'C3z':
                        # 对于 C3z，生成两个旋转点（120°, 240°）
                        for n in [1, 2]:
                            new_pt = op_actions[op_name](pt, n)
                            new_pts.append(new_pt)
                            new_ops.append(seq + [(op_name, n)])
                    else:
                        new_pt = op_actions[op_name](pt)
                        new_pts.append(new_pt)
                        new_ops.append(seq + [(op_name, None)])
                points += new_pts
                op_seqs += new_ops
            
            return points, op_seqs
        
        def apply_symm(YY: Callable[[np.ndarray], np.ndarray], kk: np.ndarray, op_seq: List[Tuple[str, Any]]):
            """
            应用对称操作序列的逆操作：
            Y_symm = (1/N)Σ_g D(g) Y(g^{-1}k) D(g)^†
            这里所有操作均使用稀疏矩阵，D(g)^† 使用 getH()（即共轭转置）。
            """
            # 将 Y_basis 的结果转换为稀疏矩阵
            Y = sparse.csr_matrix(YY(kk))
            # 依次对逆序的操作进行变换
            for op in reversed(op_seq):
                op_name, param = op
                if op_name == 'C3z':
                    D = sparse.csr_matrix(symmetry_gen.get_operator(op_name, param))
                    Y = D @ Y @ D.getH()
                elif op_name in ['C2yT', 'C2zT','TR']:
                    D = sparse.csr_matrix(symmetry_gen.get_operator(op_name, param))
                    Y = D @ Y.conjugate() @ D.getH()
                else:
                    raise ValueError(f"Unknown symmetry operation: {op_name}")
            return Y
        
        # 生成对称操作下的 k 点与对应的操作序列
        symm_points, op_seqs = gen_symm_k(k)
        # 累加所有对称化后的矩阵
        Y_symm = None
        for kk, op_seq in zip(symm_points, op_seqs):
            Y_part = apply_symm(Y_basis, kk, op_seq)
            if Y_symm is None:
                Y_symm = Y_part
            else:
                Y_symm = Y_symm + Y_part
        # Hermitian 化：取矩阵与其共轭转置的平均
        Y_symm = (Y_symm + Y_symm.getH()) / 2
        # 转换为密集数组返回
        return Y_symm.toarray()

    # @timing_decorator_factory(0)
    def stack_Y_for_term(self, term: ContinuumTerm, k_points: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        对于 term，在各个 k 点下分别计算对称化后的两部分矩阵：
          - real 部分：直接 symmetrize_Y_basis
          - imag 部分：先乘 i，再 symmetrize_Y_basis
        最后以 block_diag 拼接后返回。
        """
        # Y_real_list = [self.symmetrize_Y_basis(term.Y_basis, k) for k in k_points]
        # Y_imag_list = [self.symmetrize_Y_basis(lambda k: 1j * term.Y_basis(k), k) for k in k_points]
        Y_pairs = [
            ContinuumModelBuilder.symmetrize_Y_and_iY_basis_static(
                term.Y_basis, k, term.symmetry_ops, symmetry_gen=self.symmetry_gen, term=term
            )
            for k in k_points
        ]
        Y_real_list = [pair[0] for pair in Y_pairs]
        Y_imag_list = [pair[1] for pair in Y_pairs]
        # if term.tag == "Kinect":
        #     print(f"Y_real_list for term {term.key}:\n{np.sum(np.abs(np.array(Y_real_list)))}")
        #     print(f"Y_imag_list for term {term.key}:\n{np.sum(np.abs(np.array(Y_imag_list)))}")
            # print(f"Y_imag_list for term {term.key}:\n{Y_imag_list[0]}")
        return (scipy.linalg.block_diag(*Y_real_list),
                scipy.linalg.block_diag(*Y_imag_list))

    def symmetrize_Y_basis(self, Y_basis: Callable[[np.ndarray], np.ndarray], k: np.ndarray) -> np.ndarray:
        """
        对给定 Y_basis 进行对称化，调用内部的 symmetry_gen
        """
        return ContinuumModelBuilder.symmetrize_Y_basis_static(Y_basis, k, 
                                                                sym_ops=[],  # 外部调用时每个 term 自带对称操作
                                                                symmetry_gen=self.symmetry_gen)

    # 以下正交化与系数求解函数与之前类似，仅不再暴露为全局函数

    @timing_decorator_factory(0)
    def orthogonalize_hermitian_matrices_(self, matlist: List[np.ndarray], tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        print(f"orthogonalize_hermitian_matrices num of matlist old: {len(matlist)} dim: {matlist[0].shape}")
        orthogonallist = []
        includinglist = []
        # mat_traceless_list = [mat - np.trace(mat)/len(mat)*np.eye(len(mat)) for mat in matlist]
        mat_traceless_list = matlist
        for i, M in tqdm(enumerate(mat_traceless_list)):
            U = M.copy()
            for Q in orthogonallist:
                Q_dagger = np.conj(Q).T
                projection = np.trace(Q_dagger @ M) / np.trace(Q_dagger @ Q)
                U -= projection * Q
            norm = np.linalg.norm(U)
            if norm > tol:
                orthogonallist.append(U / norm)
                includinglist.append(i)
        return np.array(orthogonallist), np.array(includinglist, dtype=int)
    
    @timing_decorator_factory(0)
    def orthogonalize_hermitian_matrices(self,matlist: List[np.ndarray], tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        print(f"orthogonalize_hermitian_matrices num of matlist new: {len(matlist)} dim: {matlist[0].shape}")
        orthonormal_list = []
        including_list = []
        flattened = [mat.flatten() for mat in matlist]
        for i, v in tqdm(enumerate(flattened)):
            U = v.copy()
            for w in orthonormal_list:
                # 利用 np.vdot 计算内积（假设 w 已归一化）
                projection = np.vdot(w, v)
                U -= projection * w
            norm = np.linalg.norm(U)
            if norm > tol:
                orthonormal_list.append(U / norm)
                including_list.append(i)
        # 还原形状
        n = matlist[0].shape[0]
        orthonormal_matrices = [v.reshape(n, n) for v in orthonormal_list]
        return np.array(orthonormal_matrices), np.array(including_list, dtype=int)

    @timing_decorator_factory(0)
    def get_orthogonalized_terms_subset_by_part_(self, keys: List[ContinuumTermKey], k_points: List[np.ndarray],
                                                 tol: float = 1e-8, part: str = "real", tag: str = None) -> Tuple[List[ContinuumTermKey], List[np.ndarray], np.ndarray, np.ndarray]:
        initialterms = []
        for key in keys:
            mat_real, mat_imag = self.stack_Y_for_term(self.model.terms[key], k_points)
            initialterms.append(mat_real if part=="real" else mat_imag)
        initialterms = np.array(initialterms)
        initalterms_copy = initialterms
        print("sum abs of initialterms", [np.sum(np.abs(initialterm)) for initialterm in initialterms])
        onsite_energy_list = []
        if tag == "inter":
            print("sum abs of initialterms", [np.trace(initialterm) for initialterm in initialterms])
        if tag == "Onsite" or tag == "Kinect":
            # print("tag", tag)
            # print("trace of initialterms", [np.trace(initialterm) for initialterm in initialterms])
            for i, key in enumerate(keys):
                l1, l2 = key.layer_from, key.layer_to
                orb1, orb2 = key.orbital_from, key.orbital_to
                # index_start = self.get_global_index(l1, 0, orb1-1, self.Q_set1, self.Q_set2, self.n_orb1, self.n_orb2)
                # index_end = self.get_global_index(l2, len(self.Q_set2)-1, orb2-1, self.Q_set1, self.Q_set2, self.n_orb1, self.n_orb2)
                for idx, term in enumerate(initialterms):
                    for ikx, kx in enumerate(k_points):
                        index_start = self.get_global_index(l1, 0, orb1-1, self.Q_set1, self.Q_set2, self.n_orb1, self.n_orb2) + ikx*(len(self.Q_set1)*self.n_orb1 + len(self.Q_set2)*self.n_orb2)
                        index_end = self.get_global_index(l2, len(self.Q_set2), orb2-1, self.Q_set1, self.Q_set2, self.n_orb1, self.n_orb2) + ikx*(len(self.Q_set1)*self.n_orb1 + len(self.Q_set2)*self.n_orb2)
                        # print("index_start", index_start)
                        # print("index_end", index_end)
                        index_start = 0 + ikx*60
                        index_end = 60 + ikx*60
                        onsite_energy = np.trace(term[index_start:index_end, index_start:index_end])/(index_end-index_start)
                        initialterms[idx][index_start:index_end, index_start:index_end] -= onsite_energy*np.eye(index_end-index_start)
                        onsite_energy_list.append(onsite_energy)
        # for idx, term in enumerate(initialterms):
        #     for ikx, kx in enumerate(k_points):
        #         initialterms[idx][0+ikx*60:60+ikx*60, 0+ikx*60:60+ikx*60] -= np.trace(initialterms[idx][0+ikx*60:60+ikx*60, 0+ikx*60:60+ikx*60])/(60)*np.eye(60)
        finalterms, includinglist = self.orthogonalize_hermitian_matrices(initialterms, tol=tol)
        if tag == "Onsite" and part == "real":
            includinglist = np.arange(len(keys))
            finalterms = initalterms_copy
            print("trace of finalterms", [np.trace(finalterm) for finalterm in finalterms])
        for i, key in enumerate(keys):
            if not self.model.terms[key].active:
                self.model.terms[key].active = (i in includinglist)
        return keys, initialterms, finalterms, includinglist

    def compute_coefficients_by_tag_(self, heff: np.ndarray, k_points: List[np.ndarray], tol: float = 1e-8) -> Dict[str, Dict[str, np.ndarray]]:
        tag_groups: Dict[str, List[ContinuumTermKey]] = {}
        for key, term in self.model.terms.items():
            tag_groups.setdefault(term.tag, []).append(key)
        coeffs_by_tag = {}
        for tag, keys in tag_groups.items():
            print(f"Processing tag '{tag}' with {len(keys)} terms. Time: {time.strftime('%H:%M:%S', time.localtime())}")
            group_coeffs = {}
            for part in ["real", "imag"]:
                grp_keys, initialterms, finalterms, includinglist = self.get_orthogonalized_terms_subset_by_part(keys, k_points, tol=tol, part=part, tag=tag)
                print(f"  {len(includinglist)} terms included for {part} part. Time: {time.strftime('%H:%M:%S', time.localtime())}")
                if len(finalterms) == 0:
                    group_coeffs[part] = np.array([])
                    continue
                transfermat = np.array([[np.trace(finalterms[i] @ initialterms[includinglist[j]])
                                          for j in range(len(includinglist))]
                                         for i in range(len(finalterms))])

                if tag != "Onsite":
                    rhs = np.array([np.trace(heff @ finalterm) for finalterm in finalterms])
                    coeffs = np.linalg.inv(transfermat) @ rhs
                    for idx, grp_idx in enumerate(includinglist):
                        key = grp_keys[grp_idx]
                        term = self.model.terms[key]
                        c = coeffs[idx]
                        # 对于 （onsite）项单独处理
                        # if term.tag == "Onsite":
                        #     # onsite = np.trace(heff)/len(heff)
                        #     onsite = onsite_energy_list[idx]
                            # c = onsite
                        if part=="real":
                            term.r_value_real = c
                        else:
                            term.r_value_imag = c
                else:
                    if part == "imag" and len(includinglist) > 0:
                        raise ValueError("Onsite energy should be added to real part.")
                    coeffs = []
                    for idx, grp_idx in enumerate(includinglist):
                        key = grp_keys[grp_idx]
                        l1, l2 = key.layer_from, key.layer_to
                        orb1, orb2 = key.orbital_from, key.orbital_to
                        index_start = self.get_global_index(l1, 0, orb1-1, self.Q_set1, self.Q_set2, self.n_orb1, self.n_orb2) 
                        index_end = self.get_global_index(l2, len(self.Q_set2), orb2-1, self.Q_set1, self.Q_set2, self.n_orb1, self.n_orb2)
                        
                        coeffs_i = np.trace(heff @ finalterms[idx])/((index_end-index_start)*len(k_points))
                        print(f"onsite energy for term {key}: {coeffs_i}",np.trace(finalterms[idx]))
                        term = self.model.terms[key]
                        term.r_value_real = coeffs_i
                        term.r_value_imag = 0
                        coeffs.append(coeffs_i)
                    
                group_coeffs[part] = coeffs
                print(f"Updated {part} coefficients for tag '{tag}': {coeffs}")
            coeffs_by_tag[tag] = group_coeffs
        return coeffs_by_tag

    @timing_decorator_factory(0)
    def get_orthogonalized_terms_subset_old(self, keys: List[ContinuumTermKey], k_points: List[np.ndarray],
                                        tol: float = 1e-8, tag: str = None) -> Tuple[List[ContinuumTermKey], List[np.ndarray], np.ndarray, np.ndarray]:
        """
        对模型中指定 keys 的项，在 k_points 下采样后，
        对于每个 term同时采样 real 与 imag 两部分（分别由 stack_Y_for_term 返回），
        将这两部分都添加到 initialterms 中（顺序为 term1_real, term1_imag, term2_real, term2_imag, ...）。
        
        对于 onsite 或 Kinect 类项（tag=="Onsite"或tag=="Kinect"），还会在每个 k 点下对该部分做去迹处理，
        并将对应子矩阵的 trace 用于 onsite 能量的校正。
        
        返回：
        keys: 原 keys 列表（顺序不变）
        initialterms: 每个 term 得到的 block_diag 拼接矩阵（总数为 2*N）
        finalterms: 经过正交化后的矩阵数组
        includinglist: 正交化中被认为是线性独立的矩阵的原始索引数组
        """
        initialterms = []
        initialterms_copy = []
        for key in tqdm(keys):
            mat_real, mat_imag = self.stack_Y_for_term(self.model.terms[key], k_points)
            l1, l2 = key.layer_from, key.layer_to
            orb1, orb2 = key.orbital_from, key.orbital_to
            n_orb1, n_orb2 = self.n_orb1, self.n_orb2
            Q_set1 = self.Q_set1
            Q_set2 = self.Q_set2
            H_dim = len(Q_set1)*self.n_orb1 + len(Q_set2)*self.n_orb2
            
            Qlayer = Q_set1 if l1 == 1 else Q_set2
            # 若是 onsite 或 Kinect 项，则对每个 k 点对应的子块进行校正
            if tag in ("Kinect"): # l1 = l2 and orb1 = orb2
                # C2yT_flag = "C2yT" in self.symmetry_map[tag]
                # 判断是否有 C2yT 对称性
                C2yT_flag = any(sym["name"] == "C2yT" for sym in self.symmetry_map[tag])

                # print(f"tag: {tag}, C2yT_flag: {C2yT_flag}, {self.symmetry_map[tag]}")
                
                for ik, _ in enumerate(k_points):
                    # 计算子块索引（这里假设每个 k 点 block 的尺寸为 block_dim）
                    # block_dim = self.Q_set1.shape[0]*self.n_orb1 + self.Q_set2.shape[0]*self.n_orb2
                    # idx_start = ik * block_dim
                    # idx_end = (ik+1) * block_dim
                    if C2yT_flag:
                        idx_start = ik * H_dim
                        idx_end = (ik+1) * H_dim
                    else:
                        idx_start = ik * H_dim + self.get_global_index(l1, 0, orb1-1, Q_set1, Q_set2, n_orb1, n_orb2)
                        idx_end = ik * H_dim + self.get_global_index(l1, len(Qlayer), orb1-1, Q_set1, Q_set2, n_orb1, n_orb2)
                    block_dim = np.abs(idx_end - idx_start)
                    if idx_end < idx_start:
                        raise ValueError(f"Invalid index range: {idx_start} to {idx_end}")
                    onsite_energy = np.trace(mat_real[idx_start:idx_end, idx_start:idx_end]) / block_dim
                    # 去除子块的平均值
                    mat_real[idx_start:idx_end, idx_start:idx_end] -= onsite_energy * np.eye(block_dim)
                    # 同时记录 onsite 能量（这里可扩展存入 term 中，此处仅作示例）
            initialterms.append(mat_real)
            initialterms.append(mat_imag)
        initialterms = np.array(initialterms)
        initialterms_copy = initialterms.copy()
        finalterms, includinglist = self.orthogonalize_hermitian_matrices(initialterms, tol=tol)
        # 更新每个 term 的 active 标志：如果 term 对应的两个矩阵中至少有一个被保留，则该 term 保持 active
        if tag == "Onsite":
            finalterms = initialterms_copy
            includinglist = np.arange(len(finalterms))
        for i, key in enumerate(keys):
            idx1, idx2 = 2*i, 2*i+1
            self.model.terms[key].active = (idx1 in includinglist or idx2 in includinglist)
        return keys, initialterms, finalterms, includinglist
    
    @timing_decorator_factory(0)
    def get_orthogonalized_terms_subset(self, keys: List[ContinuumTermKey], k_points: List[np.ndarray],
                                        tol: float = 1e-8, tag: str = None) -> Tuple[List[ContinuumTermKey], List[np.ndarray], np.ndarray, np.ndarray]:
        """
        对模型中指定 keys 的项，在 k_points 下采样后，
        对于每个 term同时采样 real 与 imag 两部分（分别由 stack_Y_for_term 返回），
        将这两部分都添加到 initialterms 中（顺序为 term1_real, term1_imag, term2_real, term2_imag, ...）。
        
        对于 onsite 或 Kinect 类项（tag=="Onsite"或tag=="Kinect"），还会在每个 k 点下对该部分做去迹处理，
        并将对应子矩阵的 trace 用于 onsite 能量的校正。
        
        返回：
        keys: 原 keys 列表（顺序不变）
        initialterms: 每个 term 得到的 block_diag 拼接矩阵（总数为 2*N）
        finalterms: 经过正交化后的矩阵数组
        includinglist: 正交化中被认为是线性独立的矩阵的原始索引数组
        """
        
        # 内部定义处理单个 key 的函数
        def _process_single_term(key):
            # 采样获得实部与虚部矩阵
            mat_real, mat_imag = self.stack_Y_for_term(self.model.terms[key], k_points)
            l1, l2 = key.layer_from, key.layer_to
            orb1, orb2 = key.orbital_from, key.orbital_to
            n_orb1, n_orb2 = self.n_orb1, self.n_orb2
            Q_set1 = self.Q_set1
            Q_set2 = self.Q_set2
            H_dim = len(Q_set1) * self.n_orb1 + len(Q_set2) * self.n_orb2
            

            Qlayer = Q_set1 if l1 == 1 else Q_set2
            
            # 若是 Kinect 项，则对每个 k 点对应的子块进行校正
            if tag in ("Kinect",):
                # 判断是否有 C2yT 对称性
                C2yT_flag = any(sym["name"] == "C2yT" for sym in self.symmetry_map[tag])
                for ik, _ in enumerate(k_points):
                    # if C2yT_flag:
                        # idx_start = ik * H_dim
                        # idx_end = (ik + 1) * H_dim
                    # else:
                        # idx_start = ik * H_dim + self.get_global_index(l1, 0, orb1 - 1, Q_set1, Q_set2, n_orb1, n_orb2)
                        # idx_end = ik * H_dim + self.get_global_index(l1, len(Qlayer), orb1 - 1, Q_set1, Q_set2, n_orb1, n_orb2)
                    # block_dim = np.abs(idx_end - idx_start)
                    # if idx_end < idx_start:
                    #     raise ValueError(f"Invalid index range: {idx_start} to {idx_end}")
                    # onsite_energy = np.trace(mat_real[idx_start:idx_end, idx_start:idx_end]) / block_dim
                    # mat_real[idx_start:idx_end, idx_start:idx_end] -= onsite_energy * np.eye(block_dim)
                    sub_block = self.get_mat_blocks([mat_real], key, len(k_points))[0]
                    block_dim = sub_block.shape[0]
                    onsite_energy = np.trace(sub_block) / block_dim
                    mat_real -= onsite_energy * np.eye(mat_real.shape[0])
            # 返回该 key 对应的两个矩阵
            return mat_real, mat_imag

        # 并行处理 keys，使用所有 CPU 核心
        time_start = time.time()
        results = Parallel(n_jobs=1)(
            delayed(_process_single_term)(key) for key in tqdm(keys, desc="Processing terms")
        )
        time_end = time.time()
        print(f"Time elapsed for processing terms: {time_end - time_start:.2f} s")
        # 组合结果：每个 key 返回的两个矩阵依次放入 initialterms 列表
        initialterms = []
        for mat_real, mat_imag in results:
            initialterms.append(mat_real)
            initialterms.append(mat_imag)
            # print(f"shape of mat_real: {mat_real.shape}, mat_imag: {mat_imag.shape}")
        initialterms = np.array(initialterms)
        initialterms_copy = initialterms.copy()
        
        subgroup_0 = (keys[0].layer_from, keys[0].layer_to, keys[0].orbital_from, keys[0].orbital_to)
        for key in keys:
            subgroup = (key.layer_from, key.layer_to, key.orbital_from, key.orbital_to)
            if subgroup != subgroup_0:
                raise ValueError("Different subgroups in keys.")

        initialterms = np.array(self.get_mat_blocks(initialterms, keys[0], len(k_points)))
        initialterms_copy = initialterms.copy()
        
        # 正交化
        finalterms, includinglist = self.orthogonalize_hermitian_matrices(initialterms, tol=tol)

        # 如果 tag 为 "Onsite"，则不进行正交化，直接保留所有初始矩阵
        if tag == "Onsite":
            finalterms = initialterms_copy
            includinglist = np.arange(len(finalterms))
        
        # 更新每个 term 的 active 标志：如果该 term 对应的两个矩阵中至少有一个被保留，则 active 为 True
        for i, key in enumerate(keys):
            idx1, idx2 = 2 * i, 2 * i + 1
            self.model.terms[key].active = (idx1 in includinglist or idx2 in includinglist)
        
        return keys, initialterms, finalterms, includinglist
    
    
    def compute_coeffs_extreme(self,finalterms, initialterms, includinglist, heff):
        """
        极致向量化实现：
        transfermat[i, j] = trace(finalterms[i] @ initialterms[includinglist[j]])
        rhs[i] = trace(heff @ finalterms[i])
        
        具体方法：
        - 将 finalterms 重塑为 (m, n*n)
        - 选取 initialterms[includinglist] 后对每个矩阵先取转置，再重塑为 (k, n*n)
        - 利用 F @ I_T.T 计算 transfermat
        - 利用 F @ (heff.T).ravel() 计算 rhs
        
        返回求解出的系数 coeffs。
        """
        m, n, _ = finalterms.shape
        # 选取包含项
        init_included = initialterms[includinglist]  # shape (k, n, n)
        # 将 finalterms 重塑为 (m, n*n)
        F = finalterms.reshape(m, -1)
        # 对初始矩阵先取转置，再重塑为 (k, n*n)
        I_T = init_included.transpose(0, 2, 1).reshape(len(includinglist), -1)
        # 利用矩阵乘法计算 transfermat (m x k)
        transfermat = F @ I_T.T
        # 计算 rhs：heff 部分先转置后展平
        H_T = heff.T.ravel()
        rhs = F @ H_T
        # 求解线性方程组
        coeffs = np.linalg.solve(transfermat, rhs)
        return coeffs
    
    def compute_coeffs_extreme_(self, finalterms, initialterms, includinglist, heff):
        """
        极致向量化实现：利用 np.einsum 一次性计算 transfermat 与 rhs。
        
        transfermat[i, j] = trace( finalterms[i] @ initialterms[includinglist[j]] )
                        = sum_{p,q} finalterms[i, p, q] * initialterms[includinglist[j], q, p]
        
        rhs[i] = trace( heff @ finalterms[i] )
            = sum_{p,q} heff[p,q] * finalterms[i, q, p]
        """
        init_terms_included = initialterms[includinglist]  # shape (k, n, n)
        # 计算 transfermat：使用 einsum 直接得到形状 (m, k)
        transfermat = np.einsum('ipq,jqp->ij', finalterms, init_terms_included)
        # 计算 rhs：这里 finalterms[i] 的转置后乘以 heff
        rhs = np.einsum('pq,iqp->i', heff, finalterms)
        # 利用线性求解（比 np.linalg.inv 更稳定）
        coeffs = np.linalg.solve(transfermat, rhs)
        return coeffs
    
    
    def compute_coeffs_diag_only_(self,finalterms, initialterms, includinglist, heff,
                           reweight_diag=True, only_diag=False, use_lstsq=False):
        """
        reweight_diag=True: 对角权重设为 w_diag（其中 (0,0)=0.5，其余平分0.5）；
        only_diag=False:    非对角权重=1（不改变非对角的贡献）
        use_lstsq=False:    若方程非方阵或病态，建议设 True 用最小二乘
        """
        m, n, _ = finalterms.shape
        k = len(includinglist)
        init_included = initialterms[includinglist]          # (k,n,n)

        # 1) 构造权重矩阵 W
        # w_diag = np.full(n, 0.5/(n-1), dtype=float); w_diag[0] = 0.5
        nq = 19
        nk = (n//nq)//2
        w_diag = np.eye(nq, nq)
        w_diag[0,0] = 1e10
        w_diag[range(1,7),range(1,7)] = 1e10
        w_diag[range(1,4),range(1,4)] = 1e10
        w_diag[range(7,13),range(7,13)] = 1e10
        w_diag[range(13,19),range(13,19)] = 1e10
        w_diag[range(13,16),range(13,16)] = 1e10
        D_mat = np.zeros((nq,nq))
        D_mat[range(nq),index_new]=1
        w_diag = D_mat.T @ w_diag @ D_mat
        # w_diag = np.kron(np.eye(2), w_diag)
        # w_diag = np.tile(w_diag, (2,2))
        w_diag = np.block([[w_diag,w_diag],
                           [w_diag,w_diag]])
        W = np.tile(w_diag, (nk,nk))          # 若只想看对角，关掉非对角
        # print("shape of W:", W.shape, "shape of init_included:", np.array(init_included).shape,"nk = ",nk)
        # print("nonezero:", np.count_nonzero(W),np.nonzero(W),np.count_nonzero(init_included[0]),np.nonzero(init_included[0]))

        # 2) 展平
        F   = finalterms.reshape(m, -1)                      # vec(F_i) 行堆
        I_T = init_included.transpose(0, 2, 1).reshape(k, -1)# vec(I_j^T) 行堆
        H_T = heff.T.ravel()                                 # vec(H^T)

        # 3) 逐元素加权（关键一步）
        w_flat = W.reshape(-1)
        Fw = F * w_flat[None, :]                             # 对 F 的每个元素乘 w_ab

        # 4) 组装并求解
        transfermat = Fw @ I_T.T                             # (m×k)
        rhs        = Fw @ H_T                                # (m,)

        # if use_lstsq or (m != k):
        #     coeffs, *_ = np.linalg.lstsq(transfermat, rhs, rcond=None)
        # else:
        coeffs = np.linalg.solve(transfermat, rhs)
        
        H_rec = sum(coeffs[j] * init_included[j] for j in range(k))
        temp = np.zeros_like(heff)
        temp[H_rec.nonzero()] = heff[H_rec.nonzero()]
        diag = (H_rec-temp)/heff[H_rec.nonzero()]
        resid = np.linalg.norm(diag)
        print(np.sort(np.diag(diag).real)[:10],np.argsort(np.diag(diag).real)[:10])
        print(f'residual: {resid}')
        print("======compute_coeffs_diag_only_======")
        return coeffs
    
    
    
    @timing_decorator_factory(0)
    def compute_coefficients_by_tag_(self, heff: np.ndarray, k_points: List[np.ndarray], tol: float = 1e-8) -> Dict[str, Dict[str, np.ndarray]]:
        """
        对模型中不同 tag（例如 "Onsite", "Kinect", "intra", "inter"）的项分组求解系数，
        对于每组项，采用 get_orthogonalized_terms_subset 得到初始采样矩阵和正交化后的矩阵。
        注意：每个 term 对应两个矩阵（real 与 imag），
        最终解出的系数向量 x 为长度为 2*N 的向量，
        每个 term 的系数组合为 r = x[2*i] + i*x[2*i+1].
        
        对于 onsite 项（tag=="Onsite"）单独处理：直接使用 onsite 能量（例如取 heff 的 trace 平均）。
        
        返回一个字典，键为 tag，值为一个字典，包含 key "coeffs" 对应各项复数系数（按 keys 顺序）。
        """
        tag_groups: Dict[str, List[ContinuumTermKey]] = {}
        for key, term in self.model.terms.items():
            tag_groups.setdefault(term.tag, []).append(key)
        coeffs_by_tag = {}
        for tag, keys in tag_groups.items():
            # print("="*100)
            print("\n"+"="*100)
            print(f"Processing tag '{tag}' with {len(keys)} terms. Time: {time.strftime('%H:%M:%S', time.localtime())}")
            # 对于每组项，获取正交化结果（同时处理 real 和 imag 部分）
            grp_keys, initialterms, finalterms, includinglist = self.get_orthogonalized_terms_subset(keys, k_points, tol=tol, tag=tag)
            print(f"  {len(includinglist)} terms included. Time: {time.strftime('%H:%M:%S', time.localtime())}")
            # total = len(finalterms)  # 此处应为2*N
            if len(finalterms) == 0:
                coeffs_by_tag[tag] = {"coeffs": np.array([])}
                continue
            coeffs_print = []
            if tag == "Onsite":
                C2yT_flag = any(sym["name"] == "C2yT" for sym in self.symmetry_map[tag])
                H_dim = len(Q_set1)*self.n_orb1 + len(Q_set2)*self.n_orb2
                # 对于 onsite 项，直接使用 onsite 能量作为系数
                coeffs = []
                Qlayer = self.Q_set1 if grp_keys[0].layer_from == 1 else self.Q_set2
                
                for i in tqdm(range(len(keys))):
                    idx_real = includinglist.tolist().index(2*i) if 2*i in includinglist.tolist() else None
                    idx_imag = includinglist.tolist().index(2*i+1) if (2*i+1) in includinglist.tolist() else None
                    # l1, l2 = grp_keys[grp_idx].layer_from, grp_keys[grp_idx].layer_to
                    # orb1, orb2 = grp_keys[grp_idx].orbital_from, grp_keys[grp_idx].orbital_to
                    # n_orb1, n_orb2 = self.n_orb1, self.n_orb2
                    # Q_set1, Q_set2 = self.Q_set1, self.Q_set2
                    # if C2yT_flag:
                    #     idx_start = 0
                    #     idx_end = H_dim
                    # else:
                    #     idx_start = self.get_global_index(l1, 0, orb1-1, Q_set1, Q_set2, n_orb1, n_orb2)
                    #     idx_end = self.get_global_index(l1, len(Q_set1), orb1-1, Q_set1, Q_set2, n_orb1, n_orb2)
                    block_dim = H_dim if C2yT_flag else len(Qlayer)
                    # if idx_end < idx_start:
                    #     raise ValueError(f"Invalid index range: {idx_start} to {idx_end}")
                    print("sum abs of finalterms", [np.sum(np.abs(finalterm)) for finalterm in finalterms])
                    coeffs_i = np.trace(heff @ finalterms[idx_real]) / (block_dim*len(k_points))
                    coeffs.append(coeffs_i)
                    self.model.terms[keys[i]].r_value_real = coeffs_i
                    self.model.terms[keys[i]].r_value_imag = 0
                    coeffs_print.append(coeffs_i)
            else:
                # 构造 transfer matrix T: T[i,j] = trace( finalterms[i] @ initialterms[includinglist[j]] )
                print(f'before transfermat time: {time.strftime("%H:%M:%S", time.localtime())}')
                # transfermat = np.array([[np.trace(finalterms[i] @ initialterms[includinglist[j]]) 
                #                 for j in range(len(includinglist))] 
                #             for i in range(len(finalterms))])
                # # 构造右侧向量 b: b[i] = trace(heff @ finalterms[i])
                # rhs = np.array([np.trace(heff @ finalterm) for finalterm in finalterms])
                # coeffs = np.linalg.inv(transfermat) @ rhs
                coeffs = self.compute_coeffs_extreme(finalterms, initialterms, includinglist, heff)
                print(f'after transfermat time: {time.strftime("%H:%M:%S", time.localtime())}')
                # 对每个 term，组合其两个系数
                
                for i in tqdm(range(len(keys))):
                    idx_real = includinglist.tolist().index(2*i) if 2*i in includinglist.tolist() else None
                    idx_imag = includinglist.tolist().index(2*i+1) if (2*i+1) in includinglist.tolist() else None
                    r_real = coeffs[idx_real] if idx_real is not None else 0
                    r_imag = coeffs[idx_imag] if idx_imag is not None else 0
                    r = r_real + 1j*r_imag
                    coeffs_print.append(r)
                    self.model.terms[keys[i]].r_value_real = r_real
                    self.model.terms[keys[i]].r_value_imag = r_imag
            print(f"Updated coefficients for tag '{tag}': {coeffs_print}")
            print("="*100)
            coeffs_by_tag[tag] = {"coeffs": np.array(coeffs)}
        return coeffs_by_tag

    # def get_mat_blocks(self, mat_list: List[np.ndarray], subgroup: Tuple[int, int, int, int], num_kpoints: int = 1) -> List[np.ndarray]:
    def get_mat_blocks(self, mat_list: List[np.ndarray], key: ContinuumTermKey, num_kpoints: int = 1) -> List[np.ndarray]:
        """
        从 mat_list 中提取子块，子块的索引由 subgroup 指定。
        subgroup 为 (layer_from, layer_to, orbital_from, orbital_to) 的一个 tuple。

        如果是对角块（layer_from == layer_to 且 orbital_from == orbital_to），直接提取子块。
        如果不是，则认为是非对角块，此时构造一个 2×2 的块矩阵：
        [[0, A],
        [B, 0]]
        其中 A 是从 (layer_from, orbital_from) 到 (layer_to, orbital_to) 的子块，
        而 B 则是 (layer_to, orbital_to) 到 (layer_from, orbital_from) 的子块。
        """
        # 解包 subgroup 信息
        l1, l2, orb1, orb2 = key.layer_from, key.layer_to, key.orbital_from, key.orbital_to
        # l1, l2, orb1, orb2 = subgroup
        n_orb1, n_orb2 = self.n_orb1, self.n_orb2
        Q_set1, Q_set2 = self.Q_set1, self.Q_set2
        H_dim = len(Q_set1)*n_orb1 + len(Q_set2)*n_orb2
        
        tag, symm = self.model.terms[key].tag, self.model.terms[key].symmetry_ops

        
        if H_dim * num_kpoints != mat_list[0].shape[0]:
            print(f"mat_list[0].shape[0]: {mat_list[0].shape[0]}, H_dim: {H_dim}, num_kpoints: {num_kpoints}")
            raise ValueError("Mismatched matrix shape and H_dim.")

        # 根据层号确定 Q 集合
        Qlayer1 = Q_set1 if l1 == 1 else Q_set2
        Qlayer2 = Q_set1 if l2 == 1 else Q_set2

        # 计算全局索引范围
        idx_start = self.get_global_index(l1, 0, orb1-1, Q_set1, Q_set2, n_orb1, n_orb2)
        idx_end   = self.get_global_index(l1, len(Qlayer1), orb1-1, Q_set1, Q_set2, n_orb1, n_orb2)
        idy_start = self.get_global_index(l2, 0, orb2-1, Q_set1, Q_set2, n_orb1, n_orb2)
        idy_end   = self.get_global_index(l2, len(Qlayer2), orb2-1, Q_set1, Q_set2, n_orb1, n_orb2)
        idx_inc = np.arange(idx_start, idx_end, dtype=int)
        idy_inc = np.arange(idy_start, idy_end, dtype=int)
        # print("="*100)
        # print(f"idx_inc: {idx_inc}, idy_inc: {idy_inc}")
        symm_ops = [sym["name"] for sym in symm]
        if 'TR' in symm_ops:
            # print(f"TR symmetry detected for tag '{tag}'")
            l1_prime, l2_prime = l1, l2
            orb1_prime, orb2_prime = 2-(orb1-1)%2+(orb1-1)//2*2, 2-(orb2-1)%2+(orb2-1)//2*2
            idx_start_prime = self.get_global_index(l1_prime, 0, orb1_prime-1, Q_set1, Q_set2, n_orb1, n_orb2)
            idx_end_prime   = self.get_global_index(l1_prime, len(Qlayer1), orb1_prime-1, Q_set1, Q_set2, n_orb1, n_orb2)
            idy_start_prime = self.get_global_index(l2_prime, 0, orb2_prime-1, Q_set1, Q_set2, n_orb1, n_orb2)
            idy_end_prime   = self.get_global_index(l2_prime, len(Qlayer2), orb2_prime-1, Q_set1, Q_set2, n_orb1, n_orb2)
            idx_inc_prime = np.arange(idx_start_prime, idx_end_prime, dtype=int)
            idy_inc_prime = np.arange(idy_start_prime, idy_end_prime, dtype=int)
            idx_inc = np.concatenate((idx_inc, idx_inc_prime))
            idy_inc = np.concatenate((idy_inc, idy_inc_prime))
        idx_inc = np.sort(idx_inc)
        idy_inc = np.sort(idy_inc)
        if not np.array_equal(np.sort(idx_inc), np.sort(idy_inc)):
            # 1) 拼起来
            combined = np.concatenate((idx_inc, idy_inc))
            # 2) 去重并排序（也可以用 np.unique，它本身就会排序并去重）
            combined = np.unique(combined)
            # 3) 赋回
            idx_inc = combined.copy()
            idy_inc = combined.copy()
        # if orb1 != orb2:
        #     print("-"*100)
        #     print(f"idx_inc: {idx_inc}, idy_inc: {idy_inc}")
        #     print(f"symm: {symm}")
        mat_blocks = []
        for mat in mat_list:
            block_list = []
            for k in range(num_kpoints):
                # idx_start_k = idx_start + k*H_dim
                # idx_end_k = idx_end + k*H_dim
                # idy_start_k = idy_start + k*H_dim
                # idy_end_k = idy_end + k*H_dim
                # if l1 == l2 and orb1 == orb2:
                #     # 对角块：直接截取
                #     block = mat[idx_start_k:idx_end_k, idy_start_k:idy_end_k]
                # else:
                #     # 非对角块：需要组合两个方向的子块
                #     # A：从 (l1,orb1) 到 (l2,orb2)
                #     A = mat[idx_start_k:idx_end_k, idy_start_k:idy_end_k]
                #     # B：从 (l2,orb2) 到 (l1,orb1)
                #     B = mat[idy_start_k:idy_end_k, idx_start_k:idx_end_k]
                #     block = np.block([[np.zeros_like(A), A], 
                #                            [B, np.zeros_like(B)]])
                idx_inc_k = idx_inc + k*H_dim
                idy_inc_k = idy_inc + k*H_dim
                block = mat[np.ix_(idx_inc_k, idy_inc_k)]
                block_list.append(block)
            mat_blocks.append(scipy.linalg.block_diag(*block_list))

        return mat_blocks
        
            
    @timing_decorator_factory(0)
    def compute_coefficients_by_tag(self, heff: np.ndarray, k_points: List[np.ndarray], tol: float = 1e-8) -> Dict[str, Dict[Any, np.ndarray]]:
        """
        先按照 tag 对模型中的 term 进行分组，
        然后在每个 tag 内再按照 (layer_from, layer_to, orbital_from, orbital_to) 进行子分组，
        分别求解每个子组的系数。

        返回的字典结构为：
        {
            tag1: { subgroup1: coeffs_array, subgroup2: coeffs_array, ... },
            tag2: { subgroup1: coeffs_array, ... },
            ...
        }
        """
        # 按 tag 分组
        tag_groups: Dict[str, List[ContinuumTermKey]] = {}
        for key, term in self.model.terms.items():
            tag_groups.setdefault(term.tag, []).append(key)
        
        coeffs_by_tag = {}
        
        # 遍历每个 tag 组
        for tag, keys in tag_groups.items():
            # if tag not in ['Kinect', 'Onsite']:
            #     continue
            symm = self.symmetry_map.get(tag, [])
            print("\n" + "="*100)
            print(f"Processing tag '{tag}' with {len(keys)} terms of symmetry {symm}. Time: {time.strftime('%H:%M:%S', time.localtime())}")
            
            # 在当前 tag 组内按 (layer_from, layer_to, orbital_from, orbital_to) 分组
            subgroup_dict: Dict[Tuple[int, int, int, int], List[ContinuumTermKey]] = {}
            for key in keys:
                subgroup = (key.layer_from, key.layer_to, key.orbital_from, key.orbital_to)
                subgroup_dict.setdefault(subgroup, []).append(key)
            
            coeffs_by_subgroup = {}
            
            # 遍历每个子组
            for subgroup, sub_keys in subgroup_dict.items():
                print(f"  Processing subgroup {subgroup} with {len(sub_keys)} terms. Time: {time.strftime('%H:%M:%S', time.localtime())}")
                # 获取正交化结果（同时处理 real 与 imag 部分）
                grp_keys, initialterms, finalterms, includinglist = self.get_orthogonalized_terms_subset(sub_keys, k_points, tol=tol, tag=tag)
                print(f"    {len(includinglist)} terms included after orthogonalization. Time: {time.strftime('%H:%M:%S', time.localtime())}")
                
                coeffs_print = []
                # heff_block = []
                # for ik in enumerate(k_points):
                #     heff_block.append(self.get_mat_blocks([heff[ik]], subgroup)[0])
                # heff_block = scipy.linalg.block_diag(*heff_block)
                
                heff_block = self.get_mat_blocks([heff], sub_keys[0], len(k_points))[0]
                print("rank of initialterms[0]", np.linalg.matrix_rank(initialterms[0]),"shape of initialterms[0]", initialterms[0].shape)
                if np.linalg.matrix_rank(initialterms[0]) < initialterms[0].shape[0]:
                    print(f"Warning: initialterms[0] is not full rank. Rank: {np.linalg.matrix_rank(initialterms[0])}, Shape: {initialterms[0].shape}")
                    # print(f"initialterms[0]: {initialterms[0]}")
                    
                # 对于 onsite 类型单独处理
                if tag == "Onsite":
                    # C2yT_flag = any(sym["name"] == "C2yT" for sym in self.symmetry_map[tag])
                    H_dim = len(self.Q_set1)*self.n_orb1 + len(self.Q_set2)*self.n_orb2
                    coeffs = []
                    Qlayer = self.Q_set1 if grp_keys[0].layer_from == 1 else self.Q_set2
                    for i in tqdm(range(len(sub_keys))):
                        # 找到对应的 real 部分在 includinglist 中的索引（若不存在，则返回 None）
                        try:
                            idx_real = includinglist.tolist().index(2*i)
                        except ValueError:
                            idx_real = None
                        try:
                            idx_imag = includinglist.tolist().index(2*i+1)
                        except ValueError:
                            idx_imag = None
                        # block_dim = H_dim if C2yT_flag else len(Qlayer)
                        block_dim = np.shape(heff_block)[0]
                        H_Kinect_list = []
                        for k in k_points:
                            H_Kinect=0
                            for key, term in self.model.terms.items():
                                if term.tag == "Kinect":
                                    H_Kinect += term.r_value_real*self.symmetrize_Y_basis_static(term.Y_basis, k, term.symmetry_ops, self.symmetry_gen, term)
                            H_Kinect_list.append(H_Kinect)
                        H_Kinect_list = scipy.linalg.block_diag(*H_Kinect_list)
                        H_Kinect_list = self.get_mat_blocks([H_Kinect_list], sub_keys[0], len(k_points))[0]
                        heff_block = heff_block - H_Kinect_list
                        # if block_dim != len(finalterms[idx_real]):
                        #     raise ValueError(f"Block dimension mismatch: {block_dim} vs {len(finalterms[idx_real])}")
                        coeffs_i = np.trace(heff_block @ finalterms[idx_real]) / (block_dim) if idx_real is not None else 0
                        coeffs_i = np.real(coeffs_i)
                        coeffs.append(coeffs_i)
                        # 更新每个 term 的系数
                        self.model.terms[sub_keys[i]].r_value_real = coeffs_i
                        self.model.terms[sub_keys[i]].r_value_imag = 0
                        coeffs_print.append(coeffs_i)
                else:
                    # 对非 Onsite 项，构造 transfer matrix 并求解
                    print(f'    before transfer matrix computation time: {time.strftime("%H:%M:%S", time.localtime())}')
                    if tag == "Kinect":
                        heff_block = heff_block - np.eye(heff_block.shape[0]) * np.trace(heff_block) / heff_block.shape[0]
                        # initialterms = np.array([initialterms[i] - np.eye(initialterms[i].shape[0]) * np.trace(initialterms[i]) / initialterms[i].shape[0] for i in range(len(initialterms))])
                    # if tag ==  "inter":
                    #     coeffs = self.compute_coeffs_diag_only_(finalterms, initialterms, includinglist, heff_block)
                    # else:
                    coeffs = self.compute_coeffs_extreme(finalterms, initialterms, includinglist, heff_block)
                    
                    
                    coeffs = np.real(coeffs)
                    # print(f"diag of heff_block", np.diag(heff_block))
                    # print(f"diag of initialterms", np.diag(initialterms[0]))
                    # print(f"diag of finalterms", np.diag(finalterms[0]))
                    print(f'    after transfer matrix computation time: {time.strftime("%H:%M:%S", time.localtime())}')
                    for i in tqdm(range(len(sub_keys))):
                        try:
                            idx_real = includinglist.tolist().index(2*i)
                        except ValueError:
                            idx_real = None
                        try:
                            idx_imag = includinglist.tolist().index(2*i+1)
                        except ValueError:
                            idx_imag = None
                        r_real = coeffs[idx_real] if idx_real is not None else 0.0
                        r_imag = coeffs[idx_imag] if idx_imag is not None else 0.0
                        r = r_real + 1j*r_imag
                        coeffs_print.append(r)
                        self.model.terms[sub_keys[i]].r_value_real = r_real
                        self.model.terms[sub_keys[i]].r_value_imag = r_imag
                print(f"  Updated coefficients for subgroup {subgroup}: {coeffs_print}")
                coeffs_by_subgroup[subgroup] = np.array(coeffs)
            print("="*100)
            coeffs_by_tag[tag] = coeffs_by_subgroup
        return coeffs_by_tag


    def build_terms(self):
        """
        根据输入规则生成所有 term，并允许针对不同类别指定对称操作。
        """
        # 获取各类别对称操作设置
        sym_ops_onsite = self.symmetry_map.get("Onsite", [])
        sym_ops_Kinect = self.symmetry_map.get("Kinect", [])
        sym_ops_intra  = self.symmetry_map.get("intra",  [{"name": "C3z", "params": 1}, {"name": "C3z", "params": 2},
                                                        {"name": "C2yT"}, {"name": "C2zT"}])
        sym_ops_inter  = self.symmetry_map.get("inter",  [{"name": "C3z", "params": 1}, {"name": "C3z", "params": 2},
                                                        {"name": "C2yT"}, {"name": "C2zT"}])
        
        # Onsite 项：l1=l2, orbital相同, p=(0,0), Mz=Mz_star=0

        
        # Kinect 项：l1=l2, orbital相同, p=(0,0)
        for M_sum in range(0, self.max_order["Kinect"]+1):
            for Mz in range(0, M_sum+1):
                Mz_star = M_sum - Mz
                if Mz + Mz_star == 0 or (Mz - Mz_star) % 3 != 0 or Mz_star>Mz:
                    continue
                
                for l in [1]:
                    n_orb = self.n_orb1 if l==1 else self.n_orb2
                    for a in range(1, n_orb+1):
                        if a%2 == 0:
                            continue
                        key = ContinuumTermKey(Mz, Mz_star, l, l, a, a, (0.0, 0.0))
                        Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                                                                            self.n_orb1, self.n_orb2)
                        self.model.add_term(key, Y_func, tag="Kinect", symmetry_ops=sym_ops_Kinect)
                        # print(f"Kinect term {key} generated. symmetry_ops: {sym_ops_Kinect}")
        print("Kinect terms generated.")
        
        for l in [1]:
            n_orb = self.n_orb1 if l==1 else self.n_orb2
            for a in range(1, n_orb+1):
                if a%2 == 0:
                    continue
                key = ContinuumTermKey(0, 0, l, l, a, a, (0.0, 0.0))
                Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                                                                    self.n_orb1, self.n_orb2)
                self.model.add_term(key, Y_func, tag="Onsite", symmetry_ops=sym_ops_onsite)
        print("Onsite term generated.")
        
        # return
        # intra 项：l1=l2, p由intra_harmonics_map给出，且可考虑轨道不同
        for harmonic_order, p_val in self.intra_harmonics_map.items():
            # continue
            for M_sum in range(0, self.max_order["intra"]+1):
                for Mz in range(0, M_sum+1):
                    Mz_star = M_sum - Mz
                    for l in [1]:
                        n_orb = self.n_orb1 if l==1 else self.n_orb2
                        if l == 10:
                            for a in range(1, n_orb+1):
                                for b in range(1, n_orb+1):
                                    if a == b and harmonic_order==1:
                                        continue
                                    key = ContinuumTermKey(Mz, Mz_star, l, l, a, b, tuple(p_val))
                                    Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                                                                                        self.n_orb1, self.n_orb2)
                                    self.model.add_term(key, Y_func, tag="intra", symmetry_ops=sym_ops_intra)
                        elif l == 2 or l == 1:
                            for a in range(1, n_orb+1):
                                for b in range(1, n_orb+1):
                            # for a in [4,1]:
                            #     for b in [1]:
                                    # b = 5-a
                                    if a%2==0 and b%2==0:
                                        continue

                                    # if a%2!=1 and b%2==1:
                                    #     continue
                                    # if (a==b and a == 10 and M_sum > 1) or harmonic_order > 10:
                                    #     continue
                                    if a <= b and harmonic_order == 1:# and (a-b)%2!=0:
                                        continue
                                    
                                    if b == 2 and a in [2,4]:
                                        continue
                                    if (a==2 and b==3) or (a==1 and b==4):
                                        continue
                                    if a==3 and b==2 and harmonic_order== 1:
                                        continue
                                    
                                    
                                    # if (a-b)%2 == 1 and ( harmonic_order > 3 or (M_sum>3 and harmonic_order>1)):
                                    #     continue
                                    # if  harmonic_order >1 and M_sum>6:
                                    #     continue
                                    
                                    key = ContinuumTermKey(Mz, Mz_star, l, l, a, b, tuple(p_val))
                                    Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                                                                                        self.n_orb1, self.n_orb2)
                                    self.model.add_term(key, Y_func, tag="intra", symmetry_ops=sym_ops_intra)
                                    # if harmonic_order >1:
                                    #     if (a==4 and b==1) or (a==2 and b==1) or (a==4 and b==3)
                                    #         key = ContinuumTermKey(Mz, Mz_star, l, l, a, b, tuple(-p_val))
                                    #         Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                                    #                                                             self.n_orb1, self.n_orb2)
                                    #         self.model.add_term(key, Y_func, tag="intra", symmetry_ops=sym_ops_intra)
                                    # if a!=b:
                                    #     key = ContinuumTermKey(Mz, Mz_star, l, l, b, a, tuple(p_val))
                                    #     Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                                    #                                                         self.n_orb1, self.n_orb2)
                                    #     self.model.add_term(key, Y_func, tag="intra", symmetry_ops=sym_ops_intra)
                                # key = ContinuumTermKey(Mz, Mz_star, l, l, a, b, tuple(-p_val))
                                # Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                                #                                                     self.n_orb1, self.n_orb2)
                                # self.model.add_term(key, Y_func, tag="intra", symmetry_ops=sym_ops_intra)
                                    if a!=b and harmonic_order == 100:
                                        key = ContinuumTermKey(Mz, Mz_star, l, l, b, a, (0, 0))
                                        Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                                                                                            self.n_orb1, self.n_orb2)
                                        self.model.add_term(key, Y_func, tag="intra", symmetry_ops=sym_ops_intra)
                        # if n_orb > 1:
                        #     for a in range(1, n_orb):
                        #         for b in range(a+1, n_orb+1):
                        #             key = ContinuumTermKey(Mz, Mz_star, l, l, a, b, (0.0, 0.0))
                        #             Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                        #                                                                 self.n_orb1, self.n_orb2)
                        #             self.model.add_term(key, Y_func, tag="intra", symmetry_ops=sym_ops_intra)
        for l1 in [1,2]:
            continue
            l2 = l1
            n_orb = self.n_orb1 if l1==1 else self.n_orb2
            if n_orb == 0:
                continue
            same_spin_diag_key_list =       generate_orb("same_spin_diag",    l1=l1, l2=l2, max_M_sum=10, max_p_order=len(self.intra_harmonics_map.items()), orb_list=[self.n_orb1,self.n_orb2], intra_harmonics_map=self.intra_harmonics_map, symm=sym_ops_intra)
            same_spin_offdiag_key_list =    generate_orb("same_spin_offdiag", l1=l1, l2=l2, max_M_sum=8, max_p_order=len(self.intra_harmonics_map.items()), orb_list=[self.n_orb1,self.n_orb2], intra_harmonics_map=self.intra_harmonics_map, symm=sym_ops_intra)
            diff_spin_diag_key_list =       generate_orb("diff_spin_diag",    l1=l1, l2=l2, max_M_sum=10, max_p_order=len(self.intra_harmonics_map.items()), orb_list=[self.n_orb1,self.n_orb2], intra_harmonics_map=self.intra_harmonics_map, symm=sym_ops_intra)
            diff_spin_offdiag_key_list =    generate_orb("diff_spin_offdiag", l1=l1, l2=l2, max_M_sum=8, max_p_order=len(self.intra_harmonics_map.items()), orb_list=[self.n_orb1,self.n_orb2], intra_harmonics_map=self.intra_harmonics_map, symm=sym_ops_intra)
            key_list = np.concatenate((same_spin_diag_key_list, same_spin_offdiag_key_list, diff_spin_diag_key_list, diff_spin_offdiag_key_list))
            # key_list = []
            for key in key_list:
                Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                                                                    self.n_orb1, self.n_orb2)
                self.model.add_term(key, Y_func, tag="intra", symmetry_ops=sym_ops_intra)
        print("Intra terms generated.")
        
        # inter 项：l1≠l2, p由 inter_harmonics_map给出
        for harmonic_order, p_val in self.inter_harmonics_map.items():
            for M_sum in range(0, self.max_order["inter"]+1):
                # if harmonic_order > 1 and M_sum > 6:
                #     continue
                for Mz in range(0, M_sum+1):
                    Mz_star = M_sum - Mz
                    n_orb1, n_orb2 = self.n_orb1, self.n_orb2
                    # for a in range(1, n_orb1+1):
                    #     for b in range(1, n_orb2+1):
                    
                    # for a in [1]:
                    #     for b in [1]:
                    #         key = ContinuumTermKey(Mz, Mz_star, 2, 1, b, a, tuple(p_val))
                    #         Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                    #                                                             self.n_orb1, self.n_orb2)
                    #         self.model.add_term(key, Y_func, tag="inter", symmetry_ops=sym_ops_inter)   
                    if M_sum > 6 and harmonic_order > 1:
                        continue
                    key = ContinuumTermKey(Mz, Mz_star, 2, 1, 1, 1, tuple(p_val))
                    Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                                                                        self.n_orb1, self.n_orb2)
                    self.model.add_term(key, Y_func, tag="inter", symmetry_ops=sym_ops_inter)
                    
                    key = ContinuumTermKey(Mz, Mz_star, 2, 1, 2, 1, tuple(p_val))
                    Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                                                                        self.n_orb1, self.n_orb2)
                    self.model.add_term(key, Y_func, tag="inter", symmetry_ops=sym_ops_inter)
                    
                    # print(f"Inter terms generated: Mz={Mz}, Mz_star={Mz_star}, l1=2, l2=1, p={p_val}, M_sum={M_sum}, harmonic_order={harmonic_order}")
                    
                    if harmonic_order > 1:
                        key = ContinuumTermKey(Mz, Mz_star, 2, 1, 1, 1, tuple(-p_val))
                        Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                                                                            self.n_orb1, self.n_orb2)
                        self.model.add_term(key, Y_func, tag="inter", symmetry_ops=sym_ops_inter)
                        
                    #     key = ContinuumTermKey(Mz, Mz_star, 2, 1, 1, 2, tuple(p_val))
                    #     Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                    #                                                         self.n_orb1, self.n_orb2)
                    #     self.model.add_term(key, Y_func, tag="inter", symmetry_ops=sym_ops_inter)
                        
                    
                    
                        # key = ContinuumTermKey(Mz, Mz_star, 1, 2, 1, 1, tuple(p_val))
                        # Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                        #                                                     self.n_orb1, self.n_orb2)
                        # self.model.add_term(key, Y_func, tag="inter", symmetry_ops=sym_ops_inter)
                    
                    # key = ContinuumTermKey(Mz, Mz_star, 1, 2, 2, 1, tuple(-p_val))
                    # Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2,
                    #                                                     self.n_orb1, self.n_orb2)
                    # self.model.add_term(key, Y_func, tag="inter", symmetry_ops=sym_ops_inter)
                                                   
        print("Inter terms generated.")
        print("All terms generated.")

    def get_term_metadata(self, key):
        """
        根据给定的 ContinuumTermKey，返回对应 term 的 tag 和 symmetry_ops。
        """
        try:
            term = self.model.terms[key]
        except KeyError:
            raise KeyError(f"No term found for key {key!r}")
        # # 如果 term 是字典
        # if isinstance(term, dict):
        #     return term["tag"], term["symmetry_ops"]
        # 如果 term 是对象
        return term.tag, term.symmetry_ops
    
    def get_model(self) -> ContinuumModel:
        return self.model

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        # 将键转换为字符串，防止循环引用
        return {str(key): convert_to_serializable(value) for key, value in obj.items()}
    else:
        return str(obj)


# h_dft_low = np.load("/data/work/zy/software/TAPW_tmdc/tGa2Se2/2.13/2soc/Q_shell_6_cbm/heff_list.npy")
# h_dft_low = np.load("/data/work/zy/software/1.tapw_code/moirekp/kp/configs/plots_mgi2_5_2.65/heff_list.npy")
# nlow_state = [2,2]
# symmetry_gen = SymmetryGenerator(Q_set1, Q_set2, nlow_state)
# TR = symmetry_gen.get_operator(name='C2x',params=1)
# print(TR.shape)
# TR_inv = np.linalg.inv(TR)
# H = h_dft_low[0].copy()
# # H = temp
# # H = H_cont_new[0]
# fig,ax = plt.subplots(figsize=(5, 5))
# cbar = ax.matshow(np.abs(TR@H@TR_inv - H),cmap='viridis',vmin=0)
# # cbar = ax.matshow(np.abs(TR@change_orb@H@change_orb@TR_inv - change_orb@H@change_orb),cmap='viridis',vmin=0,vmax=0.08)
# # ax.matshow(np.abs(H.conj().T - H*0)/(np.abs(H)*0+1.00),cmap='viridis',vmin=0)
# plt.colorbar(cbar)



Tmat = np.array([[ 39.4387956162, 0, 0],
                [-19.7193978074, 34.1549988985, 0],
                [0, 0, 50]])      
reciprocal_Tmat = np.linalg.inv(Tmat).T*2*np.pi
print(reciprocal_Tmat)

phase = 210
# kpath = '/data/home/zy/software/TAPW_tmdc/from_work/source_02/KPATH_GMKG.in'
# kpath_out = "/data/home/zy/software/TAPW_tmdc/from_work/source_02/kpath.out"
kpath = '/Users/xtz/code/TAPW_tmdc/tMgI2/moirekp/kp/configs/mgi2_G/plots_mgi2_5_Gamma/KPATH_GMKG.in'
kpath_out = "/Users/xtz/code/TAPW_tmdc/tMgI2/moirekp/kp/configs/mgi2_G/plots_mgi2_5_Gamma/kpath.out"
kpath_config = KPathGenerator(Tmat)
kpath_config.read_and_generate_kpath(kpath,kpath_out)
kpath = np.array(kpath_config.kpoints)
kpath_dft = np.zeros_like(kpath)
for i in range(len(kpath)):
    kpath_dft[i, :2] = (np.array([reciprocal_Tmat[0][:2], reciprocal_Tmat[1][:2]]).T @ kpath[i, :2]).T
    kpath[i, :2] = rot(kpath_dft[i, :2], phase)
kpath = kpath[:, :2]

# Qlayer1 = np.load("/data/work/zy/software/1.tapw_code/tapw/examples/1.triangular_lattice/1.homo/19.MgI2/AA/3.openmx_qk/5_6.01//tapw/Q_shell_5/g_vec_list_5_Gamma_1layer.npy")
# Qlayer2 = np.load("/data/work/zy/software/1.tapw_code/tapw/examples/1.triangular_lattice/1.homo/19.MgI2/AA/3.openmx_qk/5_6.01//tapw/Q_shell_5/g_vec_list_5_Gamma_2layer.npy")

Qlayer1 = np.load("/Users/xtz/code/TAPW_tmdc/tMgI2/moirekp/kp/configs/mgi2_G/plots_mgi2_5_Gamma/g_vec_list_5_Gamma_1layer.npy")
Qlayer2 = np.load("/Users/xtz/code/TAPW_tmdc/tMgI2/moirekp/kp/configs/mgi2_G/plots_mgi2_5_Gamma/g_vec_list_5_Gamma_2layer.npy")


Q_set1 = np.array([rot(np.mean(Qlayer1,axis=0) - Qlayer1[i],210) for i in range(len(Qlayer1))])
Q_set2 = np.array([rot(np.mean(Qlayer2,axis=0) - Qlayer2[i],210) for i in range(len(Qlayer2))])

# Q_set1 = Q_set1_new
# Q_set2 = Q_set2_new

index_1 = np.argsort(np.linalg.norm(Q_set1,axis=1))
index_2 = np.argsort(np.linalg.norm(Q_set2,axis=1))
Qlayer_list = [[Q_set1], [Q_set2]]


# =============================================================================
# 4. 使用示例
# =============================================================================

# Qlayer1 = np.load("/data/work/zy/software/TAPW_tmdc/struct_1_3/BA-A_openmx/8_3.89/1_morestep/2soc/Q_shell_8_rigid_nsymm/g_vec_list_8_Gamma_1layer.npy")
# Qlayer2 = np.load("/data/work/zy/software/TAPW_tmdc/struct_1_3/BA-A_openmx/8_3.89/1_morestep/2soc/Q_shell_8_rigid_nsymm/g_vec_list_8_Gamma_2layer.npy")
# Qlayer1 = np.load("/data/work/zy/software/TAPW_tmdc/struct_1_3/BA-A_openmx/8_3.89/1_morestep/2soc/Q_shell_7_rigid_nsymm/g_vec_list_7_Gamma_1layer.npy")
# Qlayer2 = np.load("/data/work/zy/software/TAPW_tmdc/struct_1_3/BA-A_openmx/8_3.89/1_morestep/2soc/Q_shell_7_rigid_nsymm/g_vec_list_7_Gamma_2layer.npy")

# Q_set1 = np.array([rot(np.mean(Qlayer1,axis=0) - Qlayer1[i],210) for i in range(len(Qlayer1))])
# Q_set2 = np.array([rot(np.mean(Qlayer2,axis=0) - Qlayer2[i],210) for i in range(len(Qlayer2))])


index_1 = np.argsort(np.linalg.norm(Q_set1,axis=1))
index_2 = np.argsort(np.linalg.norm(Q_set2,axis=1))

bM1 = np.array([1.0, 0.0])
bM1 = np.array([np.min(np.linalg.norm(Q_set1[1:], axis=1)), 0.0])
bM1 = rot(bM1, 0)
bM2 = rot(bM1, 60)

# harmonic map（intra与inter）

intra_harmonics_map = {
    1: bM1*0,
    2: -bM1,
    # 3: bM1 + bM2,
    # 4: 2*bM1,
    # 5: 2*bM1 + bM2,
    # 5: 3*bM1
}

# 构造 inter 中所需 q 向量
q1 = np.linalg.norm(bM1)*np.array([0,1/np.sqrt(3)])
q3 = rot(q1,240)

# inter_harmonics_map = {
#     1: q1,
#     2: -2*q1,
#     3: q1 + bM2,
#     4: 2*bM2 + q3,
#     # 5: 4*q1,
#     # 6: q1 + 2*bM2
# }

inter_harmonics_map = {
    1: bM1*0,
    2: -bM1,
    # 3: bM1 + bM2,
    # 4: 2*bM1,
    # 5: 2*bM1 + bM2,
    # 5: 3*bM1
}

max_order = {"Kinect": 10, "intra": 4, "inter": 4}

# nlow_state：每层低能轨道数（例如第一层1个，第二层2个）
nlow_state = [2,2]
# 轨道数：例如第一层1个轨道，第二层2个轨道
n_orb1 = nlow_state[0]
n_orb2 = nlow_state[1]

# 构造对称操作生成器（注意：Q_set 此处取全体 Q_set1, Q_set2 可根据实际情况调整）

symmetry_gen = SymmetryGenerator(Q_set1, Q_set2, nlow_state)

# 可选：自定义 symmetry_map

# -----------------------------------------------------------------------------
# 可选：对称算符正确性审计（只跑矩阵检查，不跑后续系数/能带）
# 用法：MOIRE_AUDIT_SYMM=1 python kp/configs/mgi2_G/src/moire.py
# -----------------------------------------------------------------------------
if os.environ.get("MOIRE_AUDIT_SYMM", "0") == "1":
    def _nnz_stats(D: np.ndarray, tol: float = 1e-12) -> dict:
        mask = np.abs(D) > tol
        row_nnz = np.sum(mask, axis=1)
        col_nnz = np.sum(mask, axis=0)
        return {
            "n": int(D.shape[0]),
            "row_min": int(row_nnz.min()) if row_nnz.size else 0,
            "row_max": int(row_nnz.max()) if row_nnz.size else 0,
            "col_min": int(col_nnz.min()) if col_nnz.size else 0,
            "col_max": int(col_nnz.max()) if col_nnz.size else 0,
            "is_monomial": bool(np.all(row_nnz == 1) and np.all(col_nnz == 1)),
        }

    def _unitary_err(D: np.ndarray) -> float:
        I = np.eye(D.shape[0], dtype=complex)
        return float(np.max(np.abs(D.conj().T @ D - I)))

    def _print_op(name: str, D: np.ndarray, extra: dict | None = None) -> None:
        st = _nnz_stats(D)
        mono = ContinuumModelBuilder._extract_monomial_matrix(D, tol=1e-12)
        print(f"[AUDIT] {name}: shape={D.shape} nnz(row)=[{st['row_min']},{st['row_max']}] nnz(col)=[{st['col_min']},{st['col_max']}] monomial={st['is_monomial']} extracted={mono is not None} unitary_err={_unitary_err(D):.3e}")
        if extra:
            for k, v in extra.items():
                print(f"        {k}: {v}")
        if mono is not None:
            # 与稠密乘法做一次性对拍（随机矩阵），确保 fast-path 的公式正确
            op_name = name.split("(", 1)[0].strip()
            # name 中可能包含 params 信息，尽量从缓存里取 param
            validate_key = None
            if "C3z" in name:
                # 形如 C3z(params=1)
                try:
                    p = int(name.split("params=")[1].split(")")[0])
                    validate_key = ("C3z", p)
                except Exception:
                    validate_key = ("C3z", 1)
            elif op_name.startswith("TR"):
                validate_key = ("TR", None)
            elif op_name.startswith("C2x"):
                validate_key = ("C2x", None)
            elif op_name.startswith("C2zT"):
                validate_key = ("C2zT", None)
            elif op_name.startswith("C2yT"):
                validate_key = ("C2yT", None)
            if validate_key is not None:
                _op, _param = validate_key
                mono2 = ContinuumModelBuilder._get_monomial_op(symmetry_gen, _op, _param)
                ok = ContinuumModelBuilder._validate_monomial_op_once(symmetry_gen, _op, _param, mono2)
                print(f"        validate_monomial_fastpath: {ok}")

    print("==== Symmetry operator audit (MOIRE_AUDIT_SYMM=1) ====")
    D_TR = symmetry_gen.get_operator("TR", None)
    n = D_TR.shape[0]
    I = np.eye(n, dtype=complex)
    T2 = D_TR @ D_TR.conj()
    _print_op("TR (unitary part)", D_TR, extra={
        "max|T2 + I|": f"{np.max(np.abs(T2 + I)):.3e}",
        "max|T2 - I|": f"{np.max(np.abs(T2 - I)):.3e}",
    })

    D_C2x = symmetry_gen.get_operator("C2x", None)
    _print_op("C2x", D_C2x, extra={
        "max|C2x^2 - I|": f"{np.max(np.abs(D_C2x @ D_C2x - np.eye(D_C2x.shape[0], dtype=complex))):.3e}",
    })

    for p in (1, 2, -1, -2):
        D_C3 = symmetry_gen.get_operator("C3z", p)
        extra = None
        if p == 1:
            extra = {
                "max|C3^3 - I|": f"{np.max(np.abs(D_C3 @ D_C3 @ D_C3 - np.eye(D_C3.shape[0], dtype=complex))):.3e}",
            }
        _print_op(f"C3z(params={p})", D_C3, extra=extra)

    # 其它算符（可能未在当前配置使用）
    try:
        D_C2yT = symmetry_gen.get_operator("C2yT", None)
        _print_op("C2yT", D_C2yT)
    except Exception as e:
        print(f"[AUDIT] C2yT: not available ({e})")
    try:
        D_C2zT = symmetry_gen.get_operator("C2zT", None)
        _print_op("C2zT", D_C2zT)
    except Exception as e:
        print(f"[AUDIT] C2zT: not available ({e})")

    raise SystemExit(0)

symmetry_map = {
    "Kinect": [{"name":"TR"},{"name":"C2x"}],
    "Onsite": [{"name":"TR"},{"name":"C2x"}],
    "intra":  [{"name": "C3z"}, {"name": "TR"},{"name":"C2x"}],
    "inter":  [ {"name": "C3z"},{"name": "TR"},{"name":"C2x"}],
}

# symmetry_map = {
#     "Onsite": [],
#     "Kinect": [],
#     "intra":  [{"name": "C3z"}],
#     "inter":  [{"name": "C3z"}]
# }

# 构造模型生成器
builder = ContinuumModelBuilder(Q_set1, Q_set2, n_orb1, n_orb2,
                                bM1, bM2,
                                intra_harmonics_map, inter_harmonics_map,
                                max_order, symmetry_gen, symmetry_map)

builder.build_terms()
model = builder.get_model()

# -----------------------------------------------------------------------------
# 可选：等价性自测（不跑系数拟合/能带），用于开发时对拍
# 用法：MOIRE_SELFTEST=1 python kp/configs/mgi2_G/src/moire.py
# -----------------------------------------------------------------------------
if os.environ.get("MOIRE_SELFTEST", "0") == "1":
    rng = np.random.default_rng(0)

    # 选取若干 term 与 k 点进行对拍
    terms = list(model.terms.values())
    if not terms:
        raise RuntimeError("No terms found for self-test.")
    k_samples = kpath[rng.choice(len(kpath), size=min(5, len(kpath)), replace=False)]
    term_samples = rng.choice(terms, size=min(10, len(terms)), replace=False)
    dim_full = len(Q_set1) * n_orb1 + len(Q_set2) * n_orb2

    print("==== Self-test (MOIRE_SELFTEST=1) ====")
    print(f"Sampled {len(term_samples)} terms, {len(k_samples)} k points.")

    # 1) Y_basis(k) vs 参考实现（make_Y_basis_function_）
    max_diff_y = 0.0
    for term in term_samples:
        Y_ref_fn = ContinuumModelBuilder.make_Y_basis_function_(
            term.key, Q_set1, Q_set2, n_orb1, n_orb2
        )
        for k in k_samples:
            Y_new = term.Y_basis(k)
            Y_ref = Y_ref_fn(k)
            d = float(np.max(np.abs(Y_new - Y_ref)))
            max_diff_y = max(max_diff_y, d)
            if d > 1e-10:
                raise AssertionError(f"Y_basis mismatch: term={term.key} max|Δ|={d:.3e}")
    print(f"[PASS] Y_basis (optimized) == reference, max|Δ|={max_diff_y:.3e}")

    # 2) eval_sparse(k) -> densify 对拍（若可用）
    max_diff_sparse = 0.0
    for term in term_samples:
        if not hasattr(term.Y_basis, "eval_sparse"):
            continue
        dim = getattr(term.Y_basis, "_moire_sparse_dim", None)
        if not isinstance(dim, int):
            continue
        for k in k_samples:
            rows, cols, vals = term.Y_basis.eval_sparse(k)
            Y_sp = np.zeros((dim, dim), dtype=complex)
            np.add.at(Y_sp, (rows, cols), vals)
            Y_dn = term.Y_basis(k)
            d = float(np.max(np.abs(Y_sp - Y_dn)))
            max_diff_sparse = max(max_diff_sparse, d)
            if d > 1e-10:
                raise AssertionError(f"eval_sparse densify mismatch: term={term.key} max|Δ|={d:.3e}")
    print(f"[PASS] eval_sparse densify matches dense Y_basis, max|Δ|={max_diff_sparse:.3e}")

    # 3) symmetrize pair vs 两次单独 symmetrize（Y 与 iY）
    max_diff_symm = 0.0
    for term in term_samples:
        sym_ops = term.symmetry_ops
        for k in k_samples:
            Y_pair, Yi_pair = ContinuumModelBuilder.symmetrize_Y_and_iY_basis_static(
                term.Y_basis, k, sym_ops, symmetry_gen=symmetry_gen, term=term, use_cache=False
            )
            Y_single = ContinuumModelBuilder.symmetrize_Y_basis_static(
                term.Y_basis, k, sym_ops, symmetry_gen=symmetry_gen, term=term, use_cache=False
            )
            Yi_single = ContinuumModelBuilder.symmetrize_Y_basis_static(
                lambda kk, _Y=term.Y_basis: 1j * _Y(kk), k, sym_ops, symmetry_gen=symmetry_gen, term=term, use_cache=False
            )
            d1 = float(np.max(np.abs(Y_pair - Y_single)))
            d2 = float(np.max(np.abs(Yi_pair - Yi_single)))
            max_diff_symm = max(max_diff_symm, d1, d2)
            if d1 > 1e-10 or d2 > 1e-10:
                raise AssertionError(f"symmetrize mismatch: term={term.key} dY={d1:.3e} d(iY)={d2:.3e}")
    print(f"[PASS] symmetrize_Y_and_iY == separate symmetrize, max|Δ|={max_diff_symm:.3e}")

    # 4) composed action vs stepwise action（在真实 Y_basis(kk) 上对拍）
    max_diff_comp = 0.0
    for term in term_samples:
        sym_ops = term.symmetry_ops
        for k in k_samples:
            symm_points, op_seqs = ContinuumModelBuilder._generate_symmetry_orbit(k, sym_ops)
            for kk, op_seq in zip(symm_points, op_seqs):
                if not op_seq:
                    continue
                op_seq_applied = tuple(reversed(op_seq))
                Y0 = term.Y_basis(kk)
                # stepwise
                Ys = Y0
                for op_name, param in op_seq_applied:
                    Ys = ContinuumModelBuilder._apply_symmetry_op_to_matrix(Ys, op_name, param, symmetry_gen)
                # composed
                comp = ContinuumModelBuilder._get_composed_symmetry_action(symmetry_gen, op_seq_applied)
                if comp is None:
                    continue
                perm, vals, inv_vals, is_anti_total = comp
                Yc = Y0.conj() if is_anti_total else Y0
                Yc = vals[:, None] * Yc[perm, :]
                Yc = Yc[:, perm] * inv_vals[None, :]
                d = float(np.max(np.abs(Yc - Ys)))
                max_diff_comp = max(max_diff_comp, d)
                if d > 1e-10:
                    raise AssertionError(f"composed action mismatch: term={term.key} seq={op_seq_applied} max|Δ|={d:.3e}")
    print(f"[PASS] composed action matches stepwise on real Y_basis, max|Δ|={max_diff_comp:.3e}")

    # 5) cache-key 构造成本粗测（仅评估 get_function_hash 本身）
    t0 = time.perf_counter()
    for _ in range(2000):
        _ = get_function_hash(term_samples[0].Y_basis)
    dt = (time.perf_counter() - t0) / 2000.0
    print(f"[INFO] get_function_hash(avg) ~ {dt*1e6:.1f} µs/call (note: use_cache=False in k-loop)")

    print("==== Self-test done ====")
    raise SystemExit(0)

# 设定一组 k 点进行采样（多个 k 点后直和）
k_proj = [35, 37, 40, 43]
# k_proj = [58,0,2,18,20,22,38,40,42]
k_proj = [0,40]
# k_proj = [58,0,2]
k_points = kpath[k_proj]
# 构造数值哈密顿量 heff，其尺寸须与 block_diag 后 Y_basis 尺寸匹配：

# h_dft_low = np.load("/data/work/zy/software/1.tapw_code/moirekp/kp/configs/mgi2_G/plots_mgi2_5_Gamma/heff_list.npy")
h_dft_low = np.load("/Users/xtz/code/TAPW_tmdc/tMgI2/moirekp/kp/configs/mgi2_G/plots_mgi2_5_Gamma/heff_list.npy")
# h_dft_low = h_dft_low_modify
# h_dft_low = np.load("/data/work/zy/software/1.tapw_code/moirekp/kp/configs/plots_ga2se2/heff_list.npy")

heff = scipy.linalg.block_diag(*(h_dft_low[k_proj]))
# heff = scipy.linalg.block_diag(*(h_dft_low_reduced[k_proj]))
# 分组计算各 tag 的系数（分别对 real 与 imag 部分）
coeffs_by_tag = builder.compute_coefficients_by_tag(heff, k_points, tol=1e-6)

# print("Coefficients by tag:")
# print(json.dumps(coeffs_by_tag, indent=2, default=convert_to_serializable))

# # 保存所有 term 的系数，保存格式为字典（便于直接修改和调用）
# coefficients_storage = {}
# for key, term in model.terms.items():
#     coefficients_storage[str(key)] = {
#         "tag": term.tag,
#         "r_value_real": term.r_value_real,
#         "r_value_imag": term.r_value_imag,
#         "symmetry_ops": term.symmetry_ops
#     }
# print("\n\nFinal coefficients:")
# print(json.dumps(coefficients_storage, indent=2, default=convert_to_serializable))

# 组装连续模型的哈密顿量，取某个 k 点计算
# k_sample = kpath[0]
# H_cont = model.assemble_hamilt
# onian(k_sample, symmetry_gen=builder.symmetry_gen)
# print("Assembled Hamiltonian at k =", k_sample, ":\n", H_cont)


sym_ops_intra = [{"name":"TR"}]
l1 = 2
l2 = 2
# onsite_key_list =               generate_orb("Onsite",            l1=l1, l2=l2, max_M_sum=0, max_p_order=1, orb_list=[0, 4], intra_harmonics_map=intra_harmonics_map, symm=sym_ops_intra)
# Kinect_key_list =               generate_orb("Kinect",            l1=l1, l2=l2, max_M_sum=8, max_p_order=1, orb_list=[0, 4], intra_harmonics_map=intra_harmonics_map, symm=sym_ops_intra)
# same_spin_diag_key_list =       generate_orb("same_spin_diag",    l1=l1, l2=l2, max_M_sum=6, max_p_order=1, orb_list=[0, 4], intra_harmonics_map=intra_harmonics_map, symm=sym_ops_intra)
# same_spin_offdiag_key_list =    generate_orb("same_spin_offdiag", l1=l1, l2=l2, max_M_sum=2, max_p_order=4, orb_list=[0, 4], intra_harmonics_map=intra_harmonics_map, symm=sym_ops_intra)
# diff_spin_diag_key_list =       generate_orb("diff_spin_diag",    l1=l1, l2=l2, max_M_sum=2, max_p_order=1, orb_list=[0, 4], intra_harmonics_map=intra_harmonics_map, symm=sym_ops_intra)
# diff_spin_offdiag_key_list =    generate_orb("diff_spin_offdiag", l1=l1, l2=l2, max_M_sum=3, max_p_order=2, orb_list=[0, 4], intra_harmonics_map=intra_harmonics_map, symm=sym_ops_intra)
# print(f"len of onsite: {len(onsite_key_list)}, len of Kinect: {len(Kinect_key_list)}, len of same_spin_diag: {len(same_spin_diag_key_list)}, len of same_spin_offdiag: {len(same_spin_offdiag_key_list)}, len of diff_spin_diag: {len(diff_spin_diag_key_list)}, len of diff_spin_offdiag: {len(diff_spin_offdiag_key_list)}")
# key_list = np.concatenate((onsite_key_list, Kinect_key_list, same_spin_diag_key_list, same_spin_offdiag_key_list, diff_spin_diag_key_list, diff_spin_offdiag_key_list))
# keep, remove = truncate_indices(Q_set1, Q_set2, n_orb1, n_orb2, cutoff_shells=30, spin=False)


keep1_Q, remove1_Q = truncate_Q_indices(Q_set1, cutoff_shells=30)
keep2_Q, remove2_Q = truncate_Q_indices(Q_set2, cutoff_shells=30)
print(f"len of Q_set1: {len(Q_set1)}, len of Q_set2: {len(Q_set2)}")
print(f"len of keep1_Q: {len(keep1_Q)}, len of keep2_Q: {len(keep2_Q)}")
Q_set1_reduced = Q_set1[keep1_Q]
Q_set2_reduced = Q_set2[keep2_Q]
keep, remove = truncate_indices(Q_set1_reduced, Q_set2_reduced, n_orb1, n_orb2, cutoff_shells=3, spin=False)
print(len(keep),len(remove))

# 预先筛掉系数几乎为 0 的 term，避免每个 k 重复扫描全表
COEFF_EPS = 1.001 * 1e-33
ACTIVE_TERMS: List[Tuple[ContinuumTerm, Callable[[np.ndarray], np.ndarray], List[Dict[str, Any]], float, float]] = []
for _term in model.terms.values():
    if not getattr(_term, "active", True):
        continue
    if _term.r_value_real is None or _term.r_value_imag is None:
        raise ValueError(f"Term {_term.key} coefficients not assigned!")
    _r_real = float(_term.r_value_real)
    _r_imag = float(_term.r_value_imag)
    if abs(_r_real) > COEFF_EPS or abs(_r_imag) > COEFF_EPS:
        ACTIVE_TERMS.append((_term, _term.Y_basis, _term.symmetry_ops, _r_real, _r_imag))
N_TERMS_USED = len(ACTIVE_TERMS)
N_TERMS_REAL = sum(1 for (_, _, _, r, _) in ACTIVE_TERMS if abs(r) > 1.001 * 1e-3)
N_TERMS_IMAG = sum(1 for (_, _, _, _, r) in ACTIVE_TERMS if abs(r) > 1.001 * 1e-3)

# 复用对称化输出缓冲区（n_jobs=1 时安全），避免每个 term 重复分配 76×76 复矩阵
DIM_FULL = len(Q_set1) * n_orb1 + len(Q_set2) * n_orb2
SYMM_OUT_REAL = np.zeros((DIM_FULL, DIM_FULL), dtype=complex)
SYMM_OUT_IMAG = np.zeros((DIM_FULL, DIM_FULL), dtype=complex)
SYMM_OUT_ANTI = np.zeros((DIM_FULL, DIM_FULL), dtype=complex)

# -----------------------------------------------------------------------------
# 可选：H(k) 组装等价性对拍（验证“直接组装”与“先构造 Y_symm 再累加”严格一致）
# 用法：MOIRE_COMPARE_ASSEMBLY=1 python kp/configs/mgi2_G/src/moire.py
# -----------------------------------------------------------------------------
if os.environ.get("MOIRE_COMPARE_ASSEMBLY", "0") == "1":
    if len(kpath) == 0:
        raise ValueError("kpath is empty; cannot run MOIRE_COMPARE_ASSEMBLY.")

    test_indices = sorted(set([0, len(kpath) // 2, len(kpath) - 1]))
    print(f"==== Assembly compare (MOIRE_COMPARE_ASSEMBLY=1): test k indices {test_indices} ====")
    for idx in test_indices:
        kk = np.asarray(kpath[idx], dtype=float)

        H_dense = np.zeros((DIM_FULL, DIM_FULL), dtype=complex)
        for term, Y_basis, sym_ops, r_value_real, r_value_imag in ACTIVE_TERMS:
            Y_symm, Y_symm_i = ContinuumModelBuilder.symmetrize_Y_and_iY_basis_static(
                Y_basis, kk, sym_ops, symmetry_gen=symmetry_gen, term=term, use_cache=False
            )
            if r_value_real != 0.0:
                H_dense += r_value_real * Y_symm
            if r_value_imag != 0.0:
                H_dense += r_value_imag * Y_symm_i

        H_fast = np.zeros((DIM_FULL, DIM_FULL), dtype=complex)
        for term, Y_basis, sym_ops, r_value_real, r_value_imag in ACTIVE_TERMS:
            ContinuumModelBuilder.add_symmetrized_term_to_matrix_static(
                H_fast,
                Y_basis,
                kk,
                sym_ops,
                r_value_real,
                r_value_imag,
                symmetry_gen=symmetry_gen,
                term=term,
            )

        max_abs = float(np.max(np.abs(H_dense - H_fast)))
        print(f"[COMPARE] k_index={idx} max|ΔH|={max_abs:.3e}")
        if max_abs > 1e-10:
            raise AssertionError(f"Assembly mismatch at k_index={idx}: max|ΔH|={max_abs:.3e} > 1e-10")

    print("==== Assembly compare PASS ====")
    raise SystemExit(0)

PROFILE_LIGHT = True
def compute_eigenvalues(i,k):
    t_total_start = time.perf_counter()
    t_loop = 0.0
    t_symm = 0.0
    t_schur = 0.0
    t_eig = 0.0
    use_cache = False
    H_cont = np.zeros((DIM_FULL, DIM_FULL), dtype=complex)
        # 注意：这里组装时调用的是 builder 中封装的对称化方法（由 builder 实例调用）
        # 本方法仅简单地遍历各 term

    # for term in model.terms.values():
    # for key in key_list:
    t_loop_start = time.perf_counter()
    for term, Y_basis, sym_ops, r_value_real, r_value_imag in ACTIVE_TERMS:
        t_symm_start = time.perf_counter()
        ContinuumModelBuilder.add_symmetrized_term_to_matrix_static(
            H_cont,
            Y_basis,
            k,
            sym_ops,
            r_value_real,
            r_value_imag,
            symmetry_gen=symmetry_gen,
            term=term,
        )
        t_symm += time.perf_counter() - t_symm_start
            # contribution = term.r_value_real * Y_symm + term.r_value_imag * Y_symm_imag
            # H_cont += contribution

    t_loop = time.perf_counter() - t_loop_start
    t_schur_start = time.perf_counter()
    H00 = H_cont[np.ix_(keep, keep)]
    if len(remove) == 0:
        H = H00
    else:
        H01 = H_cont[np.ix_(keep, remove)]
        H10 = H_cont[np.ix_(remove, keep)]
        H11 = H_cont[np.ix_(remove, remove)]
        energy = np.max(np.linalg.eigvalsh(H00))
        H = H00 + H01 @ np.linalg.inv(energy*np.eye(len(H11)) - H11) @ H10
    t_schur = time.perf_counter() - t_schur_start
    
    t_eig_start = time.perf_counter()
    eigvals, vec = scipy.linalg.eigh(H, check_finite=False)
    t_eig = time.perf_counter() - t_eig_start
    # print("count = ",count,count_real,count_imag)
    
    t_total = time.perf_counter() - t_total_start
    if PROFILE_LIGHT:
        profile = {
            "t_total": t_total,
            "t_loop": t_loop,
            "t_symm": t_symm,
            "t_schur": t_schur,
            "t_eig": t_eig,
            "n_terms_used": N_TERMS_USED,
        }
        return H,eigvals,vec,[N_TERMS_USED,N_TERMS_REAL,N_TERMS_IMAG],profile
    return H,eigvals,vec,[N_TERMS_USED,N_TERMS_REAL,N_TERMS_IMAG]

N_JOBS = 1
if N_JOBS == 1:
    results_new_ = [compute_eigenvalues(i, k) for i, k in tqdm(enumerate(kpath), total=len(kpath))]
else:
    results_new_ = Parallel(n_jobs=N_JOBS)(delayed(compute_eigenvalues)(i, k) for i, k in tqdm(enumerate(kpath), total=len(kpath)))
H_cont_new = np.array([result[0] for result in results_new_])
eigvals_list_new = np.array([result[1] for result in results_new_])
eig_vec = np.array([result[2] for result in results_new_])
count = np.array([result[3] for result in results_new_])
if PROFILE_LIGHT and len(results_new_) > 0:
    profiles = [result[4] for result in results_new_]
    total_k = len(profiles)
    sums = {
        "t_total": 0.0,
        "t_loop": 0.0,
        "t_symm": 0.0,
        "t_schur": 0.0,
        "t_eig": 0.0,
    }
    for p in profiles:
        for key in sums:
            sums[key] += p[key]
    avg = {key: sums[key] / total_k for key in sums}
    denom = sums["t_total"] if sums["t_total"] > 0 else 1.0
    print("Light profiling (avg per k):")
    print(
        "  total={:.4f}s loop={:.4f}s symm={:.4f}s schur={:.4f}s eig={:.4f}s".format(
            avg["t_total"], avg["t_loop"], avg["t_symm"], avg["t_schur"], avg["t_eig"]
        )
    )
    print(
        "Light profiling (share of total):"
        " loop={:.1%} symm={:.1%} schur={:.1%} eig={:.1%}".format(
            sums["t_loop"] / denom,
            sums["t_symm"] / denom,
            sums["t_schur"] / denom,
            sums["t_eig"] / denom,
        )
    )
H_cont_new_ori = H_cont_new.copy()
count[0],len(model.terms.values()),np.shape(H_cont_new[0])
