#!/usr/bin/env python
# plot_bands.py

import argparse
import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt
import yaml
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

plt.rc('font',family='Times New Roman')
#公式也是Times New Roman
plt.rc('mathtext',fontset='stix')

@dataclass
class BandConfig:
    file: str
    label: Optional[str] = None
    color: Optional[str] = None
    linestyle: str = '-'
    linewidth: float = 1.5
    plot_type: str = 'line'  # 'line' 或 'scatter'
    marker_size: float = 1  # 散点大小
    marker: str = 'o'      # 散点形状

@dataclass
class PlotConfig:
    kpath_in: str  # KPATH.in file
    kpath_out: str # KPATH.out file
    bands: List[BandConfig]
    title: str = ""
    ymin: Optional[float] = None 
    ymax: Optional[float] = None
    output: Optional[str] = None
    legend_loc: str = "upper right"
    legend_show: bool = False  # 添加是否显示图例的配置
    legend_fontsize: int = 9   # 添加图例字体大小的配置
    global_plot_type: Optional[str] = None  # 全局设置，会覆盖单个band的设置
    marker_size: float = 1
    marker: str = 'o'
    fermi_energy: Optional[float] = None  # 添加费米能级
    fermi_line: bool = True  # 是否画费米能级的水平线
    
DEFAULT_COLORS = ['#DE6E66', 'dodgerblue', 'g', 'purple', 'orange', 'brown']

def format_kpoint_label(label: str) -> str:
    """格式化k点标签，处理特殊情况如G_M -> Γ_M或'Gamma Valley'"""
    # 处理常见的k点标签
    label_map = {
        'Gamma': r'{\Gamma}',
        'G': r'{\Gamma}',
        'K': r'{\mathrm{K}}',
        'M': r'{\mathrm{M}}',
        'X': r'{\mathrm{X}}',
        'Y': r'{\mathrm{Y}}'
    }
    # Sort keys by length, descending, to match 'Gamma' before 'G'
    sorted_keys = sorted(label_map.keys(), key=len, reverse=True)
    
    # 1. 处理带下划线的标签（例如 G_M -> Γ_M）
    if '_' in label:
        main_label, subscript = label.split('_', 1)
        main_label = label_map.get(main_label, r'{\mathrm{' + main_label + r'}}')
        return fr'${main_label}_\mathrm{{{subscript}}}$'

    # 2. 处理带空格的复杂标签 (例如 'Gamma Valley', "K' point")
    parts = label.split(' ')
    formatted_parts = []
    for part in parts:
        if not part: continue # Skip empty parts that might result from multiple spaces

        # 检查部分是否以已知的k点符号开头
        found_match = False
        for key in sorted_keys:
            if part.startswith(key):
                rest_of_part = part[len(key):]
                # '代表prime，在LaTeX中用'表示
                if rest_of_part == "'":
                    formatted_parts.append(f"{label_map[key]}'")
                else: # 其他情况，用\mathrm包裹
                    formatted_parts.append(f"{label_map[key]}" + (r"\mathrm{" + rest_of_part + "}" if rest_of_part else ""))
                found_match = True
                break
        
        if not found_match:
            # 如果没有匹配，则将整个部分视为普通文本
            formatted_parts.append(r'\mathrm{' + part + '}')
    print(f'${" ".join(formatted_parts)}$'.replace(' ', r'\ '))
    return f'${" ".join(formatted_parts)}$'.replace(' ', r'\ ')

def parse_kpath_labels(lines: List[str]) -> Tuple[List[str], List[int]]:
    """
    解析VASPKIT格式的K路径文件
    
    Args:
        lines: KPATH.in文件的行列表
    
    Returns:
        Tuple[List[str], List[int]]:
            - 不重复的高对称点标签列表（如果同一位置有多个标签，用'|'连接）
            - 对应的tick索引位置
    """
    # 读取每段路径的k点数
    N = int(lines[1].strip())
    
    # 提取所有标签和坐标
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                float(parts[0]), float(parts[1]), float(parts[2])
                labels.append(parts[3].strip())
            except ValueError:
                continue
    
    # 计算每个标签对应的tick位置
    raw_idx = []
    idx = 0
    for i in range(len(labels)):
        if (i + 1) > 1 and (i + 1) % 2 == 0:  # 偶数行，idx增加N
            idx += N
        raw_idx.append(idx)
    
    # 合并同一位置的标签
    ticks = []
    groups = {}
    for lab, t in zip(labels, raw_idx):
        if t not in groups:
            ticks.append(t)
            groups[t] = [lab]
        else:
            if lab not in groups[t]:
                groups[t].append(lab)
    
    # 生成最终标签列表，使用format_kpoint_label处理每个标签
    unique_labels = []
    for t in ticks:
        formatted_labels = [format_kpoint_label(label) for label in groups[t]]
        unique_labels.append('|'.join(formatted_labels))
    
    return unique_labels, ticks

def read_kpath(kpath_in: str, kpath_out: str) -> Tuple[List[str], np.ndarray, List[int]]:
    """
    读取KPATH.in和KPATH.out文件
    
    Args:
        kpath_in: KPATH.in文件路径，用于获取高对称点标签和位置
        kpath_out: KPATH.out文件路径，用于获取实际k点坐标
        
    Returns:
        Tuple[List[str], np.ndarray, List[int]]:
            - 格式化后的标签列表
            - 用于画图的x轴坐标
    """
    # 读取KPATH.in获取高对称点和标签
    with open(kpath_in) as f:
        lines = f.readlines()
    labels, ticks_idx = parse_kpath_labels(lines)
    
    # 读取KPATH.out获取x坐标
    kpoints_data = np.loadtxt(kpath_out)
    x_coords = kpoints_data[:, 3]  # 第4列是用于画图的x坐标
    
    return labels, x_coords, ticks_idx

def read_band_data(filename: str) -> np.ndarray:
    """读取能带数据文件"""
    return np.loadtxt(filename)

def plot_bands(config: PlotConfig):
    """主要的画图函数"""
    # 读取k点路径
    labels, x_coords, ticks_idx = read_kpath(config.kpath_in, config.kpath_out)
    # 创建图形
    fig, ax = plt.subplots(figsize=(3, 5))
    
    # 画每个文件的能带
    band_all = []
    for band in config.bands:
        data = read_band_data(band.file)
        if config.fermi_energy is not None:
            data = data - config.fermi_energy
        band_all.append(data)
    ymin = None
    ymax = None
    efermi_shift = 0
    band_max = np.max([np.max(band) for band in band_all])
    band_min = np.min([np.min(band) for band in band_all])
    print(f"efermi_shift: {config.fermi_energy}")
    print(f"band_max + efermi_shift: {band_max + config.fermi_energy}, band_min + efermi_shift: {band_min + config.fermi_energy}")
    if abs(band_max) < abs(band_min):
        if abs(band_max) < 0.5:
            efermi_shift = band_max
            band_all = [band - band_max for band in band_all]
            ymin = np.min([band[:,-6] for band in band_all])
            ymax = - ymin/6

    else:
        if abs(band_min) < 0.5:
            efermi_shift = band_min
            band_all = [band - band_min for band in band_all]
            ymax = np.max([band[:,6] for band in band_all])
            ymin = -ymax/6
            
    # band_all = np.concatenate(band_all, axis=0)
    # print(band_all.shape)
    # print(f"config.fermi_energy: {config.fermi_energy}")
    # print(f"np.max(band_all): {np.max(band_all)+config.fermi_energy}, np.min(band_all): {np.min(band_all)+config.fermi_energy}")
    # ymin = None
    # ymax = None
    # efermi_shift = 0
    # if np.abs(np.max(band_all)) < np.abs(np.min(band_all)):
    #     if np.abs(np.max(band_all)) < 0.5:
    #         efermi_shift = np.max(band_all)
    #         band_all = band_all - np.max(band_all)
    #         ymin = np.min(band_all[:,-10])
    #         ymax = - ymin/6
    # else:
    #     if np.abs(np.min(band_all)) < 0.5:
    #         efermi_shift = np.min(band_all)
    #         band_all = band_all - np.min(band_all)
    #         ymax = np.max(band_all[:,10])
    #         ymin = -ymax/6
    
    # if np.abs(np.max(band_all)) < 0.3:
    #     efermi_shift = np.max(band_all)
    #     band_all = band_all - np.max(band_all)
    #     ymin = np.min(band_all[:,-6])
    #     ymax = - ymin/6
    # if np.abs(np.min(band_all)) < 0.3:
    #     efermi_shift = np.min(band_all)
    #     band_all = band_all - np.min(band_all)
    #     ymax = np.max(band_all[:,6])
    #     ymin = -ymax/6
    # print(f"efermi_shift: {efermi_shift}, ymin: {ymin}, ymax: {ymax}, np.abs(np.max(band_all)): {np.abs(np.max(band_all))}, np.abs(np.min(band_all)): {np.abs(np.min(band_all))}")
    
    data_list = []
    for i, band in enumerate(config.bands):
            data = read_band_data(band.file)
            data_list.append(data)    
    # if len(config.bands) == 3:
    #     # M valley
        
    #     data_list = np.array(data_list)
    #     print("shape of data_list: ", data_list.shape)
    #     data_min = np.min(data_list, axis=1)
    #     min_index = np.argmin(data_min)
    #     print(f"min_index: {min_index}")
    #     other_index = np.delete(np.arange(len(config.bands)), min_index)
    #     for index in other_index:
    #         shift = data_list[index][0,0] - data_list[min_index][0,0]
    #         print(f"index: {index}, shift: {shift}")
    #         data_list[index] = data_list[index] - shift
    # data_list = np.array(data_list)
    for i, band in enumerate(config.bands):
        # data = read_band_data(band.file)  # data shape: (n_kpoints, n_bands)
        data = data_list[i]
        # 如果设置了费米能级，减去费米能级
        if config.fermi_energy is not None:
            data = data - config.fermi_energy - efermi_shift
        ymin = None
        ymax = None
        if np.abs(np.max(data)) < 0.2:
            data = data - np.max(data)
            ymin = np.min(data[:,-6])
            ymax = - ymin/6
        if np.abs(np.min(data)) < 0.2:
            data = data - np.min(data)
            ymax = np.max(data[:,6])
            ymin = -ymax/6
            
        color = band.color or DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        
        # 确保数据点数匹配
        if len(data) != len(x_coords):
            raise ValueError(f"Band data length ({len(data)}) does not match k-points length ({len(x_coords)})")
        
        # 使用全局设置或单个band的设置
        if config.global_plot_type is not None:
            plot_type = config.global_plot_type
        else:
            plot_type = band.plot_type
        
        if plot_type == 'scatter':
            # 第一条带带标签
            ax.scatter(x_coords, data[:, 0],
                      color=color,
                      label=format_kpoint_label(band.label),
                      s=band.marker_size,
                      marker=band.marker)
            # 其余带不带标签，一次性画出
            if data.shape[1] > 1:
                ax.scatter(x_coords[:, None].repeat(data.shape[1]-1, axis=1),
                          data[:, 1:],
                          color=color,
                          s=band.marker_size,
                          marker=band.marker)
        else:  # 'line'
            # 第一条带带标签
            ax.plot(x_coords, data[:, 0],
                   color=color,
                   label=format_kpoint_label(band.label),
                   linestyle=band.linestyle,
                   linewidth=band.linewidth)
            # 其余带不带标签，一次性画出
            if data.shape[1] > 1:
                ax.plot(x_coords[:, None].repeat(data.shape[1]-1, axis=1),
                       data[:, 1:],
                       color=color,
                       linestyle=band.linestyle,
                       linewidth=band.linewidth)
    
    # 设置图形属性
    ax.set_xlim(x_coords[0], x_coords[-1])
    if config.ymin is not None and config.ymax is not None:
        ax.set_ylim(config.ymin, config.ymax)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    
    # 如果设置了费米能级，画一条水平线表示费米能级
    if config.fermi_energy is not None and config.fermi_line:
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 设置k点标签和垂直线
    ticks = x_coords[ticks_idx]
    for tick in ticks[1:-1]:  # 不包括首尾点
        ax.axvline(x=tick, color='gray', linestyle='-', alpha=0.4, linewidth=0.6)
    
    # 设置x轴标签
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=11*1.5, fontfamily='Times New Roman')
    
    # 设置y轴标签和刻度
    ax.set_ylabel('Energy (eV)', fontsize=12*1.5, fontfamily='Times New Roman')
    # y_ticks = ax.get_yticks()
    # ax.set_yticklabels(y_ticks, fontsize=11, fontfamily='Times New Roman')
    y1_label = ax.get_yticklabels() 
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
    [y1_label_temp.set_fontsize(11*1.5) for y1_label_temp in y1_label]
    [y1_label_temp.set_position((0.01, y1_label_temp.get_position()[1])) for y1_label_temp in y1_label]  # Adjust the x position
    
    
    # 添加标题和图例
    if config.title:
        ax.set_title(config.title, fontsize=12*1.5, fontfamily='Times New Roman')
    
    # 根据配置决定是否显示图例
    if config.legend_show:
        ax.legend(loc=config.legend_loc, fontsize=config.legend_fontsize)
        # 设置图例中的字体
        for text in ax.get_legend().get_texts():
            text.set_fontfamily('Times New Roman')
    
    # 保存或显示
    if config.output:
        plt.savefig(config.output, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot band structure')
    parser.add_argument('--config', type=str, help='YAML config file')
    parser.add_argument('--kpath-in', type=str, help='KPATH.in file')
    parser.add_argument('--kpath-out', type=str, help='KPATH.out file')
    parser.add_argument('--bands', nargs='+', help='Band data files')
    parser.add_argument('--labels', nargs='+', help='Labels for each band')
    parser.add_argument('--colors', nargs='+', help='Colors for each band')
    parser.add_argument('--title', type=str, default='', help='Plot title')
    parser.add_argument('--energy-range', nargs=2, type=float, help='Y-axis range (min max)')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--plot-type', choices=['line', 'scatter'], 
                       help='Global plot type (line or scatter)')
    parser.add_argument('--marker-size', type=float, default=1,
                       help='Marker size for scatter plot')
    parser.add_argument('--marker', default='o',
                       help='Marker style for scatter plot')
    parser.add_argument('--fermi', type=float,
                       help='Fermi energy to subtract from band energies')
    parser.add_argument('--no-fermi-line', action='store_true',
                       help='Do not plot the Fermi level line')
    
    args = parser.parse_args()
    
    if args.config:
        # 从YAML读取配置
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
            
        # 将bands字典列表转换为BandConfig对象列表
        if 'bands' in config_dict:
            config_dict['bands'] = [BandConfig(**band) for band in config_dict['bands']]
            
        config = PlotConfig(**config_dict)
        # 用命令行参数覆盖
        if args.kpath_in: config.kpath_in = args.kpath_in
        if args.kpath_out: config.kpath_out = args.kpath_out
        if args.bands: config.bands = [BandConfig(file=band) for band in args.bands]
        if args.title: config.title = args.title
        if args.energy_range: config.ymin, config.ymax = args.energy_range
        if args.output: config.output = args.output
        if args.plot_type: config.global_plot_type = args.plot_type
        if args.marker_size: config.marker_size = args.marker_size
        if args.marker: config.marker = args.marker
        if args.fermi is not None: config.fermi_energy = args.fermi
        if args.no_fermi_line: config.fermi_line = False
    else:
        # 从命令行参数构建配置
        if not (args.kpath_in and args.kpath_out and args.bands):
            parser.error("Must provide --kpath-in, --kpath-out and --bands arguments")
            
        bands = []
        for i, band_file in enumerate(args.bands):
            bands.append(BandConfig(
                file=band_file,
                label=args.labels[i] if args.labels and i < len(args.labels) else None,
                color=args.colors[i] if args.colors and i < len(args.colors) else None,
                marker_size=args.marker_size,
                marker=args.marker
            ))
            
        config = PlotConfig(
            kpath_in=args.kpath_in,
            kpath_out=args.kpath_out,
            bands=bands,
            title=args.title,
            ymin=args.yrange[0] if args.yrange else None,
            ymax=args.yrange[1] if args.yrange else None,
            output=args.output,
            global_plot_type=args.plot_type,
            fermi_energy=args.fermi,
            fermi_line=not args.no_fermi_line
        )
    
    plot_bands(config)

if __name__ == '__main__':
    main()