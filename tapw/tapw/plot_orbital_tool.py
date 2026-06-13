#!/usr/bin/env python3
"""
TAPW 轨道成分能带绘图工具
支持交互式选择原子、轨道，绘制 fatband 图。

用法示例（建议使用模块方式运行）:

- 基本用法（指定结果目录、谷与能带类型）
  python -m tapw.plot_orbital_tool /path/to/result --valley Gamma --band CBM

- 指定轨道数据目录与输出目录
  python -m tapw.plot_orbital_tool /path/to/result --valley M1 --band vbm \
         --orbital-dir orbital_analysis --output-dir orbital_plots

- 指定 K 路径文件（可选）
  python -m tapw.plot_orbital_tool /path/to/result --valley Gamma --band CBM \
         --kpath-in KPATH.in --kpath-out KPATH.out

文件命名与自动检测:
- 新版文件结构（推荐）: 结果目录包含子目录 band/
  band/band_{CBM|VBM}_{Valley}_valley.txt
  band/vec_{CBM|VBM}_{Valley}_valley.npy

- 轨道成分 JSON 输出来自 orbital_analysis_tool:
  {result_dir}/{orbital_dir}/orbital_data_{CBM|VBM}_{Valley}.json

- 旧版文件名也会尽量兼容自动检测
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re
import yaml

from .artifacts import band_output_filename

# 设置字体为 Times New Roman
plt.rc('font', family='Times New Roman')
# 公式也是 Times New Roman
plt.rc('mathtext', fontset='stix')


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
            - tick索引位置列表
    """
    # 读取KPATH.in获取高对称点和标签
    with open(kpath_in) as f:
        lines = f.readlines()
    labels, ticks_idx = parse_kpath_labels(lines)
    
    # 读取KPATH.out获取x坐标
    kpoints_data = np.loadtxt(kpath_out)
    x_coords = kpoints_data[:, 3]  # 第4列是用于画图的x坐标
    
    return labels, x_coords, ticks_idx


@dataclass
class PlotOrbitalConfig:
    """轨道绘图配置"""
    config_file: str = "config.yaml"
    result_dir: str = "."
    orbital_dir: str = "orbital_analysis"
    output_dir: str = "orbital_plots"
    valley: str = "Gamma"
    band_type: str = "CBM"
    energy_window: Tuple[float, float] = (-2.0, 2.0)  # eV相对费米能级
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 300
    format: str = "png"
    interactive: bool = False
    detailed: bool = False
    # K路径相关配置
    kpath_in: Optional[str] = None  # KPATH.in文件路径
    kpath_out: Optional[str] = None  # KPATH.out文件路径
    use_kpath: bool = False  # 是否使用k路径标签


class OrbitalPlotter:
    """轨道成分能带绘图器"""
    
    def __init__(self, config: PlotOrbitalConfig):
        self.config = config
        self.band_data = None
        self.orbital_data = None
        self.available_atoms = []
        self.available_orbitals = []
        self.available_sublayers = []
        
        # K路径相关数据
        self.kpath_labels = None
        self.kpath_coords = None
        self.kpath_ticks = None
        
        # 创建按valley分类的输出目录
        self.valley_output_dir = os.path.join(self.config.output_dir, self.config.valley)
        os.makedirs(self.valley_output_dir, exist_ok=True)
    
    def load_data(self):
        """加载能带和轨道数据"""
        print("Loading data...")
        
        # Load band structure data (new structure preferred)
        band_dir = os.path.join(self.config.result_dir, "band")
        band_file_candidates = []
        if os.path.isdir(band_dir):
            bt = (self.config.band_type or "CBM").upper()
            band_file_candidates.extend([
                os.path.join(band_dir, band_output_filename(bt, valley_flag=self.config.valley, tapw=True)),
                os.path.join(band_dir, f"band_{bt}_{self.config.valley}_valley.txt"),
                os.path.join(band_dir, f"band_{bt}_{self.config.valley}.txt"),
            ])
        # Fallbacks for old naming
        band_file_candidates.extend([
            os.path.join(self.config.result_dir, "band_data", f"band_data_{self.config.valley}_valley.txt"),
            os.path.join(self.config.result_dir, "band_data", f"band_data_{self.config.valley}.txt"),
            os.path.join(self.config.result_dir, f"band_data_{self.config.valley}.txt"),
        ])
        band_file = next((p for p in band_file_candidates if os.path.exists(p)), None)
        
        if not band_file or not os.path.exists(band_file):
            raise FileNotFoundError(f"Band structure file not found. Tried: {band_file_candidates}")
        
        self.band_data = np.loadtxt(band_file)
        print(f"[OK] Loaded band data: {self.band_data.shape}")
        
        # Load orbital composition data
        bt = (self.config.band_type or "CBM").upper()
        orbital_dir = os.path.join(self.config.result_dir, self.config.orbital_dir)
        orbital_file_candidates = [
            os.path.join(orbital_dir, f"orbital_data_{bt}_{self.config.valley}.json"),
            # legacy fallback without band type
            os.path.join(orbital_dir, f"orbital_data_{self.config.valley}.json"),
        ]
        orbital_file = next((p for p in orbital_file_candidates if os.path.exists(p)), None)
        
        if not orbital_file or not os.path.exists(orbital_file):
            raise FileNotFoundError(f"Orbital composition file not found. Tried: {orbital_file_candidates}")
        
        with open(orbital_file, 'r') as f:
            self.orbital_data = json.load(f)
        
        print(f"[OK] Loaded orbital data: {len(self.orbital_data['data'])} k-points")
        
        # Load k-path data if available
        self._load_kpath_data()
        
        # Analyze available atoms and orbitals
        self._analyze_available_options()
    
    def _load_kpath_data(self):
        """加载K路径数据 (KPATH.in和KPATH.out)"""
        if self.config.kpath_in and self.config.kpath_out:
            if os.path.exists(self.config.kpath_in) and os.path.exists(self.config.kpath_out):
                try:
                    self.kpath_labels, self.kpath_coords, self.kpath_ticks = read_kpath(
                        self.config.kpath_in, self.config.kpath_out)
                    self.config.use_kpath = True
                    print(f"[OK] Loaded K-path data: {len(self.kpath_labels)} high-symmetry points")
                except Exception as e:
                    print(f"[WARN] Failed to load K-path data: {e}")
                    self.config.use_kpath = False
            else:
                print(f"[WARN] K-path files not found: {self.config.kpath_in}, {self.config.kpath_out}")
                self.config.use_kpath = False
        else:
            # Try to load from TAPW config.yaml
            kpath_from_config = self._try_load_kpath_from_tapw_config()
            
            if kpath_from_config:
                try:
                    self.kpath_labels, self.kpath_coords, self.kpath_ticks = read_kpath(
                        kpath_from_config['kpath_in'], kpath_from_config['kpath_out'])
                    self.config.use_kpath = True
                    print(f"[OK] Loaded K-path from TAPW config: {len(self.kpath_labels)} high-symmetry points")
                except Exception as e:
                    print(f"[WARN] Failed to load K-path from TAPW config: {e}")
                    self.config.use_kpath = False
            else:
                # Try to auto-detect K-path files in result directory
                kpath_in = os.path.join(self.config.result_dir, "KPATH.in")
                kpath_out = os.path.join(self.config.result_dir, "KPATH.out")
                
                if os.path.exists(kpath_in) and os.path.exists(kpath_out):
                    try:
                        self.kpath_labels, self.kpath_coords, self.kpath_ticks = read_kpath(kpath_in, kpath_out)
                        self.config.use_kpath = True
                        print(f"[OK] Auto-detected and loaded K-path data: {len(self.kpath_labels)} high-symmetry points")
                    except Exception as e:
                        print(f"[WARN] Failed to auto-load K-path data: {e}")
                        self.config.use_kpath = False
                else:
                    print("[INFO] K-path data not available (KPATH.in/KPATH.out not found)")
                    self.config.use_kpath = False
    
    def _try_load_kpath_from_tapw_config(self):
        """尝试从 TAPW config.yaml 文件中加载 k 路径信息"""
        try:
            import yaml
        except ImportError:
            print("[WARN] PyYAML not installed, cannot read TAPW config.yaml")
            return None
            
        # 尝试在多个位置查找 config.yaml
        config_paths = [
            self.config.config_file,  # 用户指定的配置文件
            os.path.join(self.config.result_dir, "config.yaml"),  # 结果目录中
            os.path.join(os.path.dirname(self.config.result_dir), "config.yaml"),  # 上级目录
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                    
                    # 检查是否有 paths.kpath_in 和 paths.kpath_out
                    if 'paths' in config_data:
                        paths = config_data['paths']
                        kpath_in = paths.get('kpath_in')
                        kpath_out = paths.get('kpath_out')
                        
                        if kpath_in and kpath_out:
                            # 检查文件是否存在
                            if os.path.exists(kpath_in) and os.path.exists(kpath_out):
                                print(f"[OK] Found K-path configuration in {config_path}")
                                return {'kpath_in': kpath_in, 'kpath_out': kpath_out}
                            else:
                                # 尝试相对于配置文件的路径
                                config_dir = os.path.dirname(config_path)
                                kpath_in_rel = os.path.join(config_dir, os.path.basename(kpath_in))
                                kpath_out_rel = os.path.join(config_dir, os.path.basename(kpath_out))
                                
                                if os.path.exists(kpath_in_rel) and os.path.exists(kpath_out_rel):
                                    print(f"[OK] Found K-path files relative to config in {config_path}")
                                    return {'kpath_in': kpath_in_rel, 'kpath_out': kpath_out_rel}
                                
                except Exception as e:
                    print(f"[WARN] Failed to read config file {config_path}: {e}")
                    continue
        
        return None
    
    def _setup_axes(self, ax, title: str = "", xlabel: str = "K-point", ylabel: str = "Energy (eV)", 
                   fontsize: int = 9, title_fontsize: int = 10):
        """设置坐标轴，包括k路径标签、高对称点和字体"""
        nk = len(self.orbital_data['data'])
        
        # 设置x轴
        if self.config.use_kpath and self.kpath_coords is not None:
            # 使用实际的k路径坐标
            ax.set_xlim(self.kpath_coords[0], self.kpath_coords[-1])
            
            # 设置高对称点标记和垂直线
            if self.kpath_ticks and self.kpath_labels:
                ticks = self.kpath_coords[self.kpath_ticks]
                # 画垂直分割线（不包括首尾点）
                for tick in ticks[1:-1]:
                    ax.axvline(x=tick, color='gray', linestyle='-', alpha=0.4, linewidth=0.6)
                
                # 设置x轴标签
                ax.set_xticks(ticks)
                ax.set_xticklabels(self.kpath_labels, fontsize=fontsize*1.5, fontfamily='Times New Roman')
        else:
            # 使用k点索引
            ax.set_xlim(0, nk-1)
            ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman')
        
        # 设置y轴
        ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_ylim(self.config.energy_window)
        
        # 设置标题
        if title:
            ax.set_title(title, fontsize=title_fontsize, fontfamily='Times New Roman')
        
        # 设置费米能级参考线
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.8, linewidth=1)
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        
        # 设置y轴刻度字体
        y_labels = ax.get_yticklabels()
        for label in y_labels:
            label.set_fontname('Times New Roman')
            label.set_fontsize(fontsize)
    
    def _get_kpoint_coordinate(self, k_idx: int) -> float:
        """获取k点坐标，如果有k路径数据则使用实际坐标，否则使用索引"""
        if self.config.use_kpath and self.kpath_coords is not None:
            return self.kpath_coords[k_idx] if k_idx < len(self.kpath_coords) else k_idx
        else:
            return k_idx
    
    def _analyze_available_options(self):
        """分析可用的原子类型、轨道和子层"""
        print("\n" + "="*60)
        print("Analyzing available plotting options...")
        print("="*60)
        
        structure = self.orbital_data['structure']
        atom_types = set()
        orbital_types = set()
        sublayers = []
        
        # Track atom type counts per layer-sublayer for proper naming
        layer_sublayer_atoms = {}
        
        for layer_info in structure:
            layer_num = layer_info['layer']
            layer_name = f"L{layer_num}"
            
            for sublayer_info in layer_info['sublayers']:
                sublayer_num = sublayer_info['sub']
                sublayer_name = f"S{sublayer_num}"
                atom_type = sublayer_info['atom']
                orbitals = sublayer_info['orbs']
                
                # Track atoms in each layer-sublayer
                layer_sublayer_key = (layer_name, sublayer_name)
                if layer_sublayer_key not in layer_sublayer_atoms:
                    layer_sublayer_atoms[layer_sublayer_key] = {}
                
                if atom_type not in layer_sublayer_atoms[layer_sublayer_key]:
                    layer_sublayer_atoms[layer_sublayer_key][atom_type] = 0
                layer_sublayer_atoms[layer_sublayer_key][atom_type] += 1
                
                # Get atom index within this sublayer
                atom_index = layer_sublayer_atoms[layer_sublayer_key][atom_type]
                
                atom_types.add(atom_type)
                orbital_types.update(orbitals)
                sublayers.append({
                    'layer': layer_name,
                    'sublayer': sublayer_name,
                    'atom': atom_type,
                    'atom_index': atom_index,
                    'orbs': orbitals,
                    'label': f"{layer_name}-{sublayer_name}({atom_index}{atom_type})",
                    'filename_part': f"{atom_index}{atom_type}"
                })
        
        self.available_atoms = sorted(list(atom_types))
        self.available_orbitals = sorted(list(orbital_types))
        self.available_sublayers = sublayers
        
        # Print available options
        print(f"\nAvailable atom types ({len(self.available_atoms)} types):")
        for i, atom in enumerate(self.available_atoms, 1):
            print(f"  {i:2d}. {atom}")
        
        print(f"\nAvailable orbital types ({len(self.available_orbitals)} types):")
        for i, orb in enumerate(self.available_orbitals, 1):
            print(f"  {i:2d}. {orb}")
        
        print(f"\nAvailable sublayers ({len(self.available_sublayers)} sublayers):")
        for i, sub in enumerate(self.available_sublayers, 1):
            print(f"  {i:2d}. {sub['label']} - orbitals: {', '.join(sub['orbs'])}")
    
    def interactive_selection(self):
        """交互式选择绘图内容"""
        print("\n" + "="*60)
        print("交互式轨道绘图选择")
        print("="*60)
        
        print("\n请选择绘图模式:")
        print("1. 按原子类型选择 (如: Se, In)")
        print("2. 按轨道类型选择 (如: 1s, 2px, 1dz2)")
        print("3. 按子层选择 (如: L0-S0, L1-S1)")
        print("4. 混合选择 (原子+轨道)")
        print("5. 总结模式 (自动生成主要贡献图)")
        
        while True:
            try:
                mode = int(input("\n请输入模式编号 (1-5): "))
                if 1 <= mode <= 5:
                    break
                print("错误: 请输入1-5之间的数字")
            except ValueError:
                print("错误: 请输入有效数字")
        
        if mode == 1:
            return self._select_atoms()
        elif mode == 2:
            return self._select_orbitals()
        elif mode == 3:
            return self._select_sublayers()
        elif mode == 4:
            return self._select_mixed()
        elif mode == 5:
            return self._select_summary()
    
    def _select_atoms(self):
        """选择原子类型"""
        print(f"\n请选择原子类型 (可用: {', '.join(f'{i+1}' for i in range(len(self.available_atoms)))}):")
        print("格式: 单个数字(1) 或 范围(1-3) 或 列表(1,3,5)")
        
        selection = input("输入选择: ").strip()
        indices = self._parse_selection(selection, len(self.available_atoms))
        
        selected_atoms = [self.available_atoms[i] for i in indices]
        print(f"[OK] 已选择原子: {', '.join(selected_atoms)}")
        
        return {
            'type': 'atoms',
            'atoms': selected_atoms,
            'orbitals': None,
            'sublayers': None
        }
    
    def _select_orbitals(self):
        """选择轨道类型"""
        print(f"\n请选择轨道类型 (可用: {', '.join(f'{i+1}' for i in range(len(self.available_orbitals)))}):")
        print("格式: 单个数字(1) 或 范围(1-5) 或 列表(1,3,5)")
        
        selection = input("输入选择: ").strip()
        indices = self._parse_selection(selection, len(self.available_orbitals))
        
        selected_orbitals = [self.available_orbitals[i] for i in indices]
        print(f"[OK] 已选择轨道: {', '.join(selected_orbitals)}")
        
        return {
            'type': 'orbitals',
            'atoms': None,
            'orbitals': selected_orbitals,
            'sublayers': None
        }
    
    def _select_sublayers(self):
        """选择子层"""
        print(f"\n请选择子层 (可用: {', '.join(f'{i+1}' for i in range(len(self.available_sublayers)))}):")
        print("格式: 单个数字(1) 或 范围(1-4) 或 列表(1,3,5)")
        
        selection = input("输入选择: ").strip()
        indices = self._parse_selection(selection, len(self.available_sublayers))
        
        selected_sublayers = [self.available_sublayers[i] for i in indices]
        print(f"[OK] 已选择子层: {', '.join([s['label'] for s in selected_sublayers])}")
        
        return {
            'type': 'sublayers',
            'atoms': None,
            'orbitals': None,
            'sublayers': selected_sublayers
        }
    
    def _select_mixed(self):
        """混合选择"""
        atoms_sel = self._select_atoms()
        orbitals_sel = self._select_orbitals()
        
        return {
            'type': 'mixed',
            'atoms': atoms_sel['atoms'],
            'orbitals': orbitals_sel['orbitals'],
            'sublayers': None
        }
    
    def _select_summary(self):
        """总结模式"""
        print("\n总结模式将生成以下图表:")
        print("  - 层级总贡献 (L0 vs L1)")
        print("  - 原子类型总贡献 (Se vs In)")
        print("  - 轨道类型总贡献 (s, p, d)")
        
        return {
            'type': 'summary',
            'atoms': None,
            'orbitals': None,
            'sublayers': None
        }
    
    def _parse_selection(self, selection: str, max_val: int) -> List[int]:
        """解析用户选择字符串"""
        indices = []
        
        # 处理逗号分隔的多个选择
        parts = [p.strip() for p in selection.split(',')]
        
        for part in parts:
            if '-' in part:
                # 范围选择 (如 1-5)
                start, end = part.split('-', 1)
                start_idx = int(start.strip()) - 1
                end_idx = int(end.strip()) - 1
                indices.extend(range(start_idx, end_idx + 1))
            else:
                # 单个选择
                idx = int(part) - 1
                indices.append(idx)
        
        # 验证索引范围
        indices = [i for i in indices if 0 <= i < max_val]
        return sorted(list(set(indices)))  # 去重并排序
    
    def plot_detailed(self):
        """详细模式：绘制所有原子和轨道"""
        print("\nDetailed mode: Generating all detailed plots...")
        
        # 1. Plot detailed sublayer plots (all orbitals combined)
        for i, sublayer in enumerate(self.available_sublayers):
            layer_name = sublayer['layer']
            sublayer_name = sublayer['sublayer']
            atom_part = sublayer['filename_part']
            filename = f"detailed_{layer_name}_{sublayer_name}_{atom_part}_all"
            self._plot_sublayer_detail(sublayer, filename)
        
        # 2. Plot individual orbital breakdown for each sublayer
        for i, sublayer in enumerate(self.available_sublayers):
            layer_name = sublayer['layer']
            sublayer_name = sublayer['sublayer'] 
            atom_part = sublayer['filename_part']
            filename = f"detailed_{layer_name}_{sublayer_name}_{atom_part}_breakdown"
            self._plot_sublayer_orbital_breakdown(sublayer, filename)
        
        # 2b. Plot spin-resolved orbital breakdown if spin-polarized
        print("  Generating spin-resolved plots...")
        for i, sublayer in enumerate(self.available_sublayers):
            layer_name = sublayer['layer']
            sublayer_name = sublayer['sublayer']
            atom_part = sublayer['filename_part']
            filename = f"detailed_{layer_name}_{sublayer_name}_{atom_part}_spin"
            self._plot_sublayer_spin_breakdown(sublayer, filename)
        
        # 3. Plot atom type summaries
        for atom in self.available_atoms:
            self._plot_atom_summary(atom, f"summary_atom_{atom}")
        
        # 4. Plot orbital group summaries
        orbital_groups = self._group_orbitals()
        for group_name, orbitals in orbital_groups.items():
            self._plot_orbital_group(group_name, orbitals, f"summary_orbital_{group_name}")
        
        # 5. Plot layer/sublayer/atom contribution summaries
        print("  Generating layer/sublayer/atom contribution plots...")
        self._plot_layer_sublayer_atom_contributions()
        
        # 6. Plot overall summary
        self._plot_overall_summary("summary_overall")
        
        print(f"[OK] Detailed mode completed, plots saved in: {self.valley_output_dir}/")
    
    def _group_orbitals(self) -> Dict[str, List[str]]:
        """将轨道按类型分组"""
        groups = {'s': [], 'p': [], 'd': [], 'f': [], 'other': []}
        
        for orb in self.available_orbitals:
            if 's' in orb.lower():
                groups['s'].append(orb)
            elif 'p' in orb.lower():
                groups['p'].append(orb)
            elif 'd' in orb.lower():
                groups['d'].append(orb)
            elif 'f' in orb.lower():
                groups['f'].append(orb)
            else:
                groups['other'].append(orb)
        
        # 移除空组
        return {k: v for k, v in groups.items() if v}
    
    def _plot_sublayer_detail(self, sublayer: Dict, filename: str):
        """Plot detailed orbital composition for a single sublayer"""
        print(f"  Plotting {sublayer['label']} detailed plot...")
        
        fig, ax = plt.subplots(figsize=(4,6), dpi=self.config.dpi)
        
        # Extract orbital data for this sublayer
        weights_data = self._extract_sublayer_weights(sublayer)
        
        # Draw fatband
        self._draw_fatband(ax, weights_data, f"{sublayer['label']} Orbital Composition")
        
        # 保存图片
        output_path = os.path.join(self.valley_output_dir, f"{filename}.{self.config.format}")
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_sublayer_orbital_breakdown(self, sublayer: Dict, filename_base: str):
        """绘制子层中每个轨道的单独详细图"""
        print(f"  Plotting {sublayer['label']} orbital breakdown...")
        
        # Create subplots with more space for colorbar (make them taller and narrower, 5 per row)
        n_orbitals = len(sublayer['orbs'])
        n_cols = min(4, n_orbitals)  # 每行5个图
        n_rows = (n_orbitals + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 4), dpi=self.config.dpi, sharey=True)
        if n_orbitals == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        # 首先计算所有轨道的权重范围以统一colorbar
        all_weights = []
        orbital_data_list = []
        for orbital in sublayer['orbs']:
            single_orb_data = self._extract_single_orbital_weights(sublayer, orbital)
            orbital_data_list.append((orbital, single_orb_data))
            all_weights.extend(single_orb_data['weights'][single_orb_data['weights'] > 0.0001])
        
        # 确定统一的colorbar范围
        if all_weights:
            vmin, vmax = 0, max(all_weights)
        else:
            vmin, vmax = 0, 1
        
        # 为每个轨道单独绘制
        for orb_idx, (orbital, single_orb_data) in enumerate(orbital_data_list):
            ax = axes[orb_idx] if orb_idx < len(axes) else None
            if ax is None:
                continue
                
            # 绘制fatband with unified colorbar range
            self._draw_single_orbital_fatband(ax, single_orb_data, f"{sublayer['label']} - {orbital}", vmin, vmax)
        
        # 隐藏多余的子图
        for i in range(n_orbitals, len(axes)):
            axes[i].set_visible(False)
        
        # Adjust layout with more space
        plt.tight_layout(pad=2.0, w_pad=3.0, h_pad=2.0)
        
        # 保存图片
        output_path = os.path.join(self.valley_output_dir, f"{filename_base}.{self.config.format}")
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight', pad_inches=0.2)
        plt.close()
    
    def _extract_single_orbital_weights(self, sublayer: Dict, target_orbital: str) -> Dict:
        """提取子层中单个轨道的权重数据"""
        layer_name = sublayer['layer']
        sublayer_name = sublayer['sublayer']
        
        k_points = []
        energies = []
        weights = []
        
        for k_idx, k_data in enumerate(self.orbital_data['data']):
            k_points.append(k_idx)
            
            for band_idx, band_data in enumerate(k_data['bands']):
                energy = self.band_data[k_idx, band_idx]
                energies.append(energy)
                
                # 获取该子层的轨道权重
                composition = band_data['comp']
                weight = 0.0
                
                if layer_name in composition and sublayer_name in composition[layer_name]:
                    sublayer_data = composition[layer_name][sublayer_name]
                    
                    # 找到目标轨道的索引
                    if target_orbital in sublayer['orbs']:
                        orb_idx = sublayer['orbs'].index(target_orbital)
                        
                        if 'up' in sublayer_data and orb_idx < len(sublayer_data['up']):
                            # Handle spin-polarized case: sum up and down contributions
                            up_weight = sublayer_data['up'][orb_idx] if orb_idx < len(sublayer_data['up']) else 0.0
                            dn_weight = sublayer_data['dn'][orb_idx] if orb_idx < len(sublayer_data['dn']) else 0.0
                            weight = up_weight + dn_weight
                        elif 'weights' in sublayer_data and orb_idx < len(sublayer_data['weights']):
                            # Handle non-spin-polarized case
                            weight = sublayer_data['weights'][orb_idx]
                
                weights.append(weight)
        
        return {
            'k_points': np.array(k_points),
            'energies': np.array(energies),
            'weights': np.array(weights),
            'orbital': target_orbital
        }
    
    def _draw_single_orbital_fatband(self, ax, orbital_data, title: str, vmin: float = None, vmax: float = None):
        """绘制单个轨道的fatband图"""
        k_points = orbital_data['k_points']
        energies = orbital_data['energies']
        weights = orbital_data['weights']
        
        # 重构数据用于绘图
        nk = len(self.orbital_data['data'])
        nbands = len(self.orbital_data['data'][0]['bands'])
        
        # 构建k点坐标数组
        if self.config.use_kpath and self.kpath_coords is not None:
            k_mesh = self.kpath_coords[:nk] if len(self.kpath_coords) >= nk else np.arange(nk)
        else:
            k_mesh = np.arange(nk)
            
        energy_mesh = energies.reshape(nk, nbands)
        weight_mesh = weights.reshape(nk, nbands)
        
        # First draw background band lines
        bands_drawn = 0
        for band in range(nbands):
            mask = (energy_mesh[:, band] >= self.config.energy_window[0]) & (energy_mesh[:, band] <= self.config.energy_window[1])
            if np.any(mask):
                ax.plot(k_mesh[mask], energy_mesh[mask, band], 'k-', linewidth=0.5, alpha=0.3)
                bands_drawn += 1
        
        if bands_drawn == 0:
            print(f"Warning: No band lines drawn for {title}, energy window: {self.config.energy_window}")
        
        # Collect valid points for plotting
        k_points_plot = []
        energies_plot = []
        sizes_plot = []
        
        for band in range(nbands):
            for k in range(nk):
                energy = energy_mesh[k, band]
                weight = weight_mesh[k, band]
                
                # 只绘制在能量窗口内且有贡献的点
                if (self.config.energy_window[0] <= energy <= self.config.energy_window[1] and weight > 0.0001):
                    k_points_plot.append(self._get_kpoint_coordinate(k))
                    energies_plot.append(energy)
                    # 将权重转换为散点大小
                    sizes_plot.append(max(5, min(100, weight * 200)))
        
        if len(k_points_plot) > 0:
            # Use actual weights for colors
            weights_for_color = []
            for i in range(len(k_points_plot)):
                k_coord = k_points_plot[i]
                energy = energies_plot[i]
                # Find the corresponding k index and band
                k_idx = -1
                if self.config.use_kpath and self.kpath_coords is not None:
                    # Find closest k index
                    k_idx = np.argmin(np.abs(self.kpath_coords[:nk] - k_coord))
                else:
                    k_idx = int(k_coord)
                
                band_idx = -1
                for b in range(nbands):
                    if abs(energy_mesh[k_idx, b] - energy) < 1e-6:
                        band_idx = b
                        break
                if band_idx >= 0:
                    weights_for_color.append(weight_mesh[k_idx, band_idx])
                else:
                    weights_for_color.append(0)
            
            # Use provided vmin/vmax or calculate from data
            if vmin is None or vmax is None:
                data_vmin, data_vmax = 0, max(weights_for_color) if weights_for_color else 1
            else:
                data_vmin, data_vmax = vmin, vmax
                
            scatter = ax.scatter(k_points_plot, energies_plot, 
                               s=sizes_plot, c=weights_for_color, cmap='Reds', 
                               alpha=0.8, edgecolors='none', vmin=data_vmin, vmax=data_vmax)
            
            # Add colorbar with proper spacing
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15)
            cbar.ax.tick_params(labelsize=7)
        
        # 使用新的坐标轴设置方法
        self._setup_axes(ax, title=title, fontsize=8, title_fontsize=10)
    
    def _plot_sublayer_spin_breakdown(self, sublayer: Dict, filename_base: str):
        """Plot spin-resolved orbital breakdown for a sublayer"""
        print(f"  Plotting {sublayer['label']} spin breakdown...")
        
        # Create figure with subplots for up and down spins (make it taller and narrower)
        fig, (ax_up, ax_down) = plt.subplots(1, 2, figsize=(6, 4), 
                                            dpi=self.config.dpi, sharey=True)
        
        # Extract data for both spins and calculate unified colorbar range
        spin_up_data = self._extract_sublayer_spin_weights(sublayer, 'up')
        spin_down_data = self._extract_sublayer_spin_weights(sublayer, 'down')
        
        # Calculate unified colorbar range
        all_weights = []
        for orb_weights in spin_up_data['weights'].values():
            all_weights.extend([w for w in orb_weights if w > 0.0001])
        for orb_weights in spin_down_data['weights'].values():
            all_weights.extend([w for w in orb_weights if w > 0.0001])
        
        if all_weights:
            vmin, vmax = 0, max(all_weights)
        else:
            vmin, vmax = 0, 1
        
        # Plot spin-up contributions
        self._draw_spin_fatband(ax_up, spin_up_data, f"{sublayer['label']} - Spin Up", 'Blues', vmin, vmax)
        
        # Plot spin-down contributions  
        self._draw_spin_fatband(ax_down, spin_down_data, f"{sublayer['label']} - Spin Down", 'Reds', vmin, vmax)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.valley_output_dir, f"{filename_base}.{self.config.format}")
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
    
    def _extract_sublayer_spin_weights(self, sublayer: Dict, spin: str) -> Dict:
        """Extract spin-resolved weights for a sublayer"""
        layer_name = sublayer['layer']
        sublayer_name = sublayer['sublayer']
        
        k_points = []
        energies = []
        weights = {orb: [] for orb in sublayer['orbs']}
        
        for k_idx, k_data in enumerate(self.orbital_data['data']):
            k_points.append(k_idx)
            
            for band_idx, band_data in enumerate(k_data['bands']):
                energy = self.band_data[k_idx, band_idx]
                energies.append(energy)
                
                # Get orbital weights for this sublayer
                composition = band_data['comp']
                if layer_name in composition and sublayer_name in composition[layer_name]:
                    sublayer_data = composition[layer_name][sublayer_name]
                    
                    for orb_idx, orb in enumerate(sublayer['orbs']):
                        if spin == 'up' and 'up' in sublayer_data and orb_idx < len(sublayer_data['up']):
                            weight = sublayer_data['up'][orb_idx]
                        elif spin == 'down' and 'dn' in sublayer_data and orb_idx < len(sublayer_data['dn']):
                            weight = sublayer_data['dn'][orb_idx]
                        elif 'weights' in sublayer_data and orb_idx < len(sublayer_data['weights']):
                            # For non-spin-polarized, split weight equally
                            weight = sublayer_data['weights'][orb_idx] / 2.0
                        else:
                            weight = 0.0
                        weights[orb].append(weight)
                else:
                    # If no data, fill with zeros
                    for orb in sublayer['orbs']:
                        weights[orb].append(0.0)
        
        return {
            'k_points': np.array(k_points),
            'energies': np.array(energies),
            'weights': weights
        }
    
    def _draw_spin_fatband(self, ax, weights_data, title: str, colormap: str, vmin: float = None, vmax: float = None):
        """Draw fatband for a specific spin channel"""
        k_points = weights_data['k_points']
        energies = weights_data['energies']
        weights = weights_data['weights']
        
        # Restructure data for plotting
        nk = len(self.orbital_data['data'])
        nbands = len(self.orbital_data['data'][0]['bands'])
        
        k_mesh = np.arange(nk)
        energy_mesh = energies.reshape(nk, nbands)
        
        # First draw background band lines
        for band in range(nbands):
            mask = (energy_mesh[:, band] >= self.config.energy_window[0]) & (energy_mesh[:, band] <= self.config.energy_window[1])
            if np.any(mask):
                ax.plot(k_mesh[mask], energy_mesh[mask, band], 'k-', linewidth=0.5, alpha=0.3)
        
        # Draw fatband for each orbital
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
        
        for i, (orb, orb_weights) in enumerate(weights.items()):
            weight_mesh = np.array(orb_weights).reshape(nk, nbands)
            
            # Collect valid points
            k_points_plot = []
            energies_plot = []
            sizes_plot = []
            weights_plot = []
            
            for band in range(nbands):
                for k in range(nk):
                    energy = energy_mesh[k, band]
                    weight = weight_mesh[k, band]
                    
                    if (self.config.energy_window[0] <= energy <= self.config.energy_window[1] and weight > 0.0001):
                        k_points_plot.append(self._get_kpoint_coordinate(k))
                        energies_plot.append(energy)
                        sizes_plot.append(max(5, min(100, weight * 300)))
                        weights_plot.append(weight)
            
            if len(k_points_plot) > 0:
                # Use provided vmin/vmax or calculate from data
                if vmin is None or vmax is None:
                    data_vmin, data_vmax = 0, max(weights_plot) if weights_plot else 1
                else:
                    data_vmin, data_vmax = vmin, vmax
                    
                scatter = ax.scatter(k_points_plot, energies_plot, 
                                   s=sizes_plot, c=weights_plot, cmap=colormap,
                                   alpha=0.7, label=orb, edgecolors='none',
                                   vmin=data_vmin, vmax=data_vmax)
        
        # Set up plot
        # ax.set_ylabel('Energy (eV)')
        # ax.set_title(title)
        # ax.set_ylim(self.config.energy_window)
        # ax.axhline(y=0, color='k', linestyle='--', alpha=0.8, linewidth=1)
        # ax.grid(True, alpha=0.3)
        # ax.set_xlim(0, nk-1)
        # 使用新的坐标轴设置方法
        self._setup_axes(ax, title=title, fontsize=8, title_fontsize=10)
        
        # Add colorbar and legend with proper spacing
        if len(k_points_plot) > 0:
            # Add colorbar on the right
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.15)
            cbar.set_label('Orbital Weight', rotation=270, labelpad=15)
            
            # Add legend below the plot to avoid overlap
            ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=min(len(weights), 7),
                     fontsize='small', frameon=True)
    
    def _extract_sublayer_weights(self, sublayer: Dict) -> Dict:
        """提取子层的权重数据"""
        layer_name = sublayer['layer']
        sublayer_name = sublayer['sublayer']
        
        k_points = []
        energies = []
        weights = {orb: [] for orb in sublayer['orbs']}
        
        for k_idx, k_data in enumerate(self.orbital_data['data']):
            k_points.append(k_idx)
            
            for band_idx, band_data in enumerate(k_data['bands']):
                energy = self.band_data[k_idx, band_idx]  # 直接获取能量
                energies.append(energy)
                
                # 获取该子层的轨道权重
                composition = band_data['comp']
                if layer_name in composition and sublayer_name in composition[layer_name]:
                    sublayer_data = composition[layer_name][sublayer_name]
                    
                    for orb_idx, orb in enumerate(sublayer['orbs']):
                        if 'up' in sublayer_data:
                            weight = sublayer_data['up'][orb_idx] + sublayer_data['dn'][orb_idx]
                        else:
                            weight = sublayer_data['weights'][orb_idx]
                        weights[orb].append(weight)
                else:
                    # 如果没有数据，填充0
                    for orb in sublayer['orbs']:
                        weights[orb].append(0.0)
        
        return {
            'k_points': np.array(k_points),
            'energies': np.array(energies),
            'weights': weights
        }
    
    def _draw_fatband(self, ax, weights_data, title: str):
        """绘制真正的fatband图 - 使用散点大小表示轨道贡献"""
        k_points = weights_data['k_points']
        energies = weights_data['energies']
        weights = weights_data['weights']
        
        # 重构数据用于绘图
        nk = len(self.orbital_data['data'])
        nbands = len(self.orbital_data['data'][0]['bands'])
        
        # 构建k点坐标数组
        if self.config.use_kpath and self.kpath_coords is not None:
            k_mesh = self.kpath_coords[:nk] if len(self.kpath_coords) >= nk else np.arange(nk)
        else:
            k_mesh = np.arange(nk)
            
        energy_mesh = energies.reshape(nk, nbands)
        
        # First draw background band lines
        bands_drawn = 0
        for band in range(nbands):
            mask = (energy_mesh[:, band] >= self.config.energy_window[0]) & (energy_mesh[:, band] <= self.config.energy_window[1])
            if np.any(mask):
                ax.plot(k_mesh[mask], energy_mesh[mask, band], 'k-', linewidth=0.5, alpha=0.3)
                bands_drawn += 1
        
        if bands_drawn == 0:
            print(f"Warning: No band lines drawn for {title}, energy window: {self.config.energy_window}")
        
        # 为每个轨道绘制fatband散点
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
        
        for i, (orb, orb_weights) in enumerate(weights.items()):
            weight_mesh = np.array(orb_weights).reshape(nk, nbands)
            
            # 收集所有有效的点
            k_points_plot = []
            energies_plot = []
            sizes_plot = []
            
            for band in range(nbands):
                for k in range(nk):
                    energy = energy_mesh[k, band]
                    weight = weight_mesh[k, band]
                    
                    # 只绘制在能量窗口内且有贡献的点  
                    if (self.config.energy_window[0] <= energy <= self.config.energy_window[1] and weight > 0.0001):
                        k_points_plot.append(self._get_kpoint_coordinate(k))
                        energies_plot.append(energy)
                        # 将权重转换为散点大小 (0.001-1 -> 1-200)
                        sizes_plot.append(max(1, min(200, weight * 500)))
            
            if len(k_points_plot) > 0:
                scatter = ax.scatter(k_points_plot, energies_plot, 
                                   s=sizes_plot, c=[colors[i]], 
                                   alpha=0.7, label=orb, edgecolors='none')
        
        # 使用新的坐标轴设置方法
        self._setup_axes(ax, title=title, fontsize=9, title_fontsize=10)
        
        # Set up legends including scatter size explanation
        legend1 = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        ax.add_artist(legend1)
        
        # Add scatter size explanation
        size_legend_elements = [
            plt.scatter([], [], s=20, c='gray', alpha=0.7, label='Small'),
            plt.scatter([], [], s=100, c='gray', alpha=0.7, label='Medium'),
            plt.scatter([], [], s=200, c='gray', alpha=0.7, label='Large')
        ]
        legend2 = ax.legend(handles=size_legend_elements, bbox_to_anchor=(1.05, 0.3), 
                           loc='upper left', title='Weight Size', fontsize='small')
    
    def _plot_atom_summary(self, atom: str, filename: str):
        """绘制原子类型总结图"""
        print(f"  绘制 {atom} 原子总结图...")
        return None
    
    def _plot_orbital_group(self, group_name: str, orbitals: List[str], filename: str):
        """绘制轨道组总结图"""
        print(f"  绘制 {group_name} 轨道组总结图...")
        return None
    
    def _plot_layer_sublayer_atom_contributions(self):
        """Plot fatband contributions by layer, sublayer, and atom type"""
        
        # Group sublayers by layer
        layers_data = {}
        for sublayer in self.available_sublayers:
            layer = sublayer['layer']
            if layer not in layers_data:
                layers_data[layer] = []
            layers_data[layer].append(sublayer)
        
        # Only plot the combined fatband showing all sublayer-atom contributions
        # (Layer-specific plots are not useful)
        self._plot_combined_sublayer_fatband()
    
    def _plot_layer_fatband(self, layer_name: str, layer_sublayers: list):
        """Plot fatband for a specific layer showing all its sublayer-atom contributions"""
        print(f"    Plotting {layer_name} fatband with all sublayer contributions...")
        
        fig, ax = plt.subplots(figsize=(8, 10), dpi=self.config.dpi)
        
        # Extract weights data for all sublayers in this layer
        weights_data = self._extract_layer_weights(layer_sublayers)
        
        # Draw fatband
        self._draw_fatband(ax, weights_data, f"{layer_name} Sublayer-Atom Contributions")
        
        # Save plot
        output_path = os.path.join(self.valley_output_dir, f"summary_{layer_name}_sublayer_contributions.{self.config.format}")
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        
        print(f"      Saved {layer_name} sublayer fatband: {self.valley_output_dir}/summary_{layer_name}_sublayer_contributions.{self.config.format}")
    
    def _plot_combined_sublayer_fatband(self):
        """Plot combined fatband with subplots for each sublayer-atom contribution"""
        print(f"    Plotting combined fatband with subplots for each sublayer-atom...")
        
        # Check if data is spin-polarized
        spin_polarized = self._check_if_spin_polarized()
        
        n_sublayers = len(self.available_sublayers)
        
        if spin_polarized:
            # 自旋极化情况：自旋上的图全部排完后，换行开始排自旋下的图
            n_cols = min(4, n_sublayers)  # 每行最多4个图
            n_rows_up = (n_sublayers + n_cols - 1) // n_cols  # 自旋上需要的行数
            n_rows_dn = (n_sublayers + n_cols - 1) // n_cols  # 自旋下需要的行数
            n_rows = n_rows_up + n_rows_dn  # 总行数
            n_total_plots = n_sublayers * 2
        else:
            n_total_plots = n_sublayers
            n_cols = min(4, n_sublayers)
            n_rows = (n_sublayers + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 4), dpi=self.config.dpi)
        
        # Handle axes array structure
        if n_total_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        # Calculate unified colorbar range for all sublayers
        all_weights = []
        
        for sublayer in self.available_sublayers:
            if spin_polarized:
                up_data = self._extract_sublayer_spin_weights(sublayer, 'up')
                dn_data = self._extract_sublayer_spin_weights(sublayer, 'down')
                
                # Collect weights for unified range
                for orb_weights in up_data['weights'].values():
                    all_weights.extend([w for w in orb_weights if w > 0.0001])
                for orb_weights in dn_data['weights'].values():
                    all_weights.extend([w for w in orb_weights if w > 0.0001])
            else:
                weights_data = self._extract_single_sublayer_weights(sublayer)
                for orb_weights in weights_data['weights'].values():
                    all_weights.extend([w for w in orb_weights if w > 0.0001])
        
        # Determine unified colorbar range
        if all_weights:
            vmin, vmax = 0, max(all_weights)
        else:
            vmin, vmax = 0, 1
        
        # Plot sublayers
        for idx, sublayer in enumerate(self.available_sublayers):
            if spin_polarized:
                # Extract spin data
                up_data = self._extract_sublayer_spin_weights(sublayer, 'up')
                dn_data = self._extract_sublayer_spin_weights(sublayer, 'down')
                
                # Plot spin up: 在自旋上的行区域内按行列排列
                up_row = idx // n_cols
                up_col = idx % n_cols
                up_ax_idx = up_row * n_cols + up_col
                if up_ax_idx < len(axes):
                    self._draw_sublayer_fatband_single_spin(axes[up_ax_idx], up_data, 
                                                           f"{sublayer['label']} (↑)", 
                                                           vmin, vmax, 'Blues')
                
                # Plot spin down: 从自旋上区域之后开始按行列排列
                dn_row = idx // n_cols + n_rows_up  # 从自旋上行数之后开始
                dn_col = idx % n_cols
                dn_ax_idx = dn_row * n_cols + dn_col
                if dn_ax_idx < len(axes):
                    self._draw_sublayer_fatband_single_spin(axes[dn_ax_idx], dn_data, 
                                                           f"{sublayer['label']} (↓)", 
                                                           vmin, vmax, 'Reds')
            else:
                # Non-spin-polarized case
                weights_data = self._extract_single_sublayer_weights(sublayer)
                if idx < len(axes):
                    self._draw_sublayer_fatband_single(axes[idx], weights_data, sublayer['label'], vmin, vmax)
        
        # Hide unused subplots (如果有的话)
        used_indices = set()
        if spin_polarized:
            # 记录实际使用的子图索引
            for idx in range(n_sublayers):
                # 自旋上索引
                up_row = idx // n_cols
                up_col = idx % n_cols
                up_ax_idx = up_row * n_cols + up_col
                used_indices.add(up_ax_idx)
                
                # 自旋下索引
                dn_row = idx // n_cols + n_rows_up
                dn_col = idx % n_cols
                dn_ax_idx = dn_row * n_cols + dn_col
                used_indices.add(dn_ax_idx)
        else:
            for idx in range(n_sublayers):
                used_indices.add(idx)
        
        # 隐藏未使用的子图
        for i in range(len(axes)):
            if i not in used_indices:
                axes[i].set_visible(False)
        
        # Adjust layout
        plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)
        
        # Save plot
        output_path = os.path.join(self.valley_output_dir, f"summary_all_sublayer_contributions.{self.config.format}")
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        
        print(f"      Saved combined sublayer fatband: {self.valley_output_dir}/summary_all_sublayer_contributions.{self.config.format}")
    
    def _extract_layer_weights(self, layer_sublayers: list) -> Dict:
        """Extract orbital weights for all sublayers in a layer"""
        k_points = []
        energies = []
        weights = {}
        
        # Initialize weights for each sublayer-atom
        for sublayer in layer_sublayers:
            key = sublayer['label']
            weights[key] = []
        
        for k_idx, k_data in enumerate(self.orbital_data['data']):
            k_points.append(k_idx)
            
            for band_idx, band_data in enumerate(k_data['bands']):
                energy = self.band_data[k_idx, band_idx]
                energies.append(energy)
                
                # Get weights for each sublayer-atom in this layer
                composition = band_data['comp']
                
                for sublayer in layer_sublayers:
                    layer_name = sublayer['layer']
                    sublayer_name = sublayer['sublayer']
                    key = sublayer['label']
                    
                    weight = 0.0
                    if layer_name in composition and sublayer_name in composition[layer_name]:
                        sublayer_data = composition[layer_name][sublayer_name]
                        if 'tot' in sublayer_data:
                            weight = sublayer_data['tot']
                    
                    weights[key].append(weight)
        
        return {
            'k_points': np.array(k_points),
            'energies': np.array(energies),
            'weights': weights
        }
    
    def _extract_all_sublayers_weights(self) -> Dict:
        """Extract orbital weights for all sublayers"""
        k_points = []
        energies = []
        weights = {}
        
        # Initialize weights for each sublayer-atom
        for sublayer in self.available_sublayers:
            key = sublayer['label']
            weights[key] = []
        
        for k_idx, k_data in enumerate(self.orbital_data['data']):
            k_points.append(k_idx)
            
            for band_idx, band_data in enumerate(k_data['bands']):
                energy = self.band_data[k_idx, band_idx]
                energies.append(energy)
                
                # Get weights for each sublayer-atom
                composition = band_data['comp']
                
                for sublayer in self.available_sublayers:
                    layer_name = sublayer['layer']
                    sublayer_name = sublayer['sublayer']
                    key = sublayer['label']
                    
                    weight = 0.0
                    if layer_name in composition and sublayer_name in composition[layer_name]:
                        sublayer_data = composition[layer_name][sublayer_name]
                        if 'tot' in sublayer_data:
                            weight = sublayer_data['tot']
                    
                    weights[key].append(weight)
        
        return {
            'k_points': np.array(k_points),
            'energies': np.array(energies),
            'weights': weights
        }
    
    def _extract_single_sublayer_weights(self, sublayer: Dict) -> Dict:
        """Extract total weights for a single sublayer (all orbitals combined)"""
        layer_name = sublayer['layer']
        sublayer_name = sublayer['sublayer']
        
        k_points = []
        energies = []
        weights = {'total': []}
        
        for k_idx, k_data in enumerate(self.orbital_data['data']):
            k_points.append(k_idx)
            
            for band_idx, band_data in enumerate(k_data['bands']):
                energy = self.band_data[k_idx, band_idx]
                energies.append(energy)
                
                # Get total weight for this sublayer
                composition = band_data['comp']
                weight = 0.0
                
                if layer_name in composition and sublayer_name in composition[layer_name]:
                    sublayer_data = composition[layer_name][sublayer_name]
                    if 'tot' in sublayer_data:
                        weight = sublayer_data['tot']
                
                weights['total'].append(weight)
        
        return {
            'k_points': np.array(k_points),
            'energies': np.array(energies),
            'weights': weights
        }
    
    def _draw_sublayer_fatband_with_spin(self, ax, up_data: Dict, dn_data: Dict, title: str, vmin: float, vmax: float):
        """Draw fatband for a sublayer with spin resolution showing both up and down"""
        k_points = up_data['k_points']
        energies = up_data['energies']
        
        # Restructure data for plotting
        nk = len(self.orbital_data['data'])
        nbands = len(self.orbital_data['data'][0]['bands'])
        
        k_mesh = np.arange(nk)
        energy_mesh = energies.reshape(nk, nbands)
        
        # First draw background band lines
        for band in range(nbands):
            mask = (energy_mesh[:, band] >= self.config.energy_window[0]) & (energy_mesh[:, band] <= self.config.energy_window[1])
            if np.any(mask):
                ax.plot(k_mesh[mask], energy_mesh[mask, band], 'k-', linewidth=0.5, alpha=0.3)
        
        # Extract up and down weights
        up_weights = []
        dn_weights = []
        for i in range(len(energies)):
            up_total = sum([orb_weights[i] for orb_weights in up_data['weights'].values()])
            dn_total = sum([orb_weights[i] for orb_weights in dn_data['weights'].values()])
            up_weights.append(up_total)
            dn_weights.append(dn_total)
        
        up_weight_mesh = np.array(up_weights).reshape(nk, nbands)
        dn_weight_mesh = np.array(dn_weights).reshape(nk, nbands)
        
        # Plot up-spin (positive y-shift)
        k_up, e_up, s_up, w_up = [], [], [], []
        for band in range(nbands):
            for k in range(nk):
                energy = energy_mesh[k, band]
                weight = up_weight_mesh[k, band]
                
                if (self.config.energy_window[0] <= energy <= self.config.energy_window[1] and weight > 0.0001):
                    k_up.append(k)
                    e_up.append(energy)
                    s_up.append(max(3, min(60, weight * 150)))
                    w_up.append(weight)
        
        # Plot down-spin (negative y-shift) 
        k_dn, e_dn, s_dn, w_dn = [], [], [], []
        for band in range(nbands):
            for k in range(nk):
                energy = energy_mesh[k, band]
                weight = dn_weight_mesh[k, band]
                
                if (self.config.energy_window[0] <= energy <= self.config.energy_window[1] and weight > 0.0001):
                    k_dn.append(k)
                    e_dn.append(energy)
                    s_dn.append(max(3, min(60, weight * 150)))
                    w_dn.append(weight)
        
        # Plot both spins - use red/blue colormaps with circles
        scatter = None
        
        if len(k_up) > 0:
            scatter_up = ax.scatter(k_up, e_up, s=s_up, c=w_up, cmap='Blues', 
                                   alpha=0.8, edgecolors='none',
                                   marker='o', label='Spin Up',
                                   vmin=vmin, vmax=vmax)
            scatter = scatter_up
        
        if len(k_dn) > 0:
            scatter_dn = ax.scatter(k_dn, e_dn, s=s_dn, c=w_dn, cmap='Reds',
                                   alpha=0.8, edgecolors='none',
                                   marker='o', label='Spin Down',
                                   vmin=vmin, vmax=vmax)
            if scatter is None:
                scatter = scatter_dn
        
        # Add colorbar if we have data
        if scatter is not None:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15)
            cbar.set_label('Weight', rotation=270, labelpad=10, fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        
        # Set up plot
        ax.set_xlabel('K-point', fontsize=8)
        ax.set_ylabel('Energy (eV)', fontsize=8)
        ax.set_title(f"{title} (Spin-resolved)", fontsize=9)
        ax.set_ylim(self.config.energy_window)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.8, linewidth=1)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, nk-1)
        ax.tick_params(labelsize=7)
        
        # Add legend for spin channels
        if len(k_up) > 0 or len(k_dn) > 0:
            ax.legend(fontsize=6, loc='upper right')
    
    def _draw_sublayer_fatband_single(self, ax, weights_data: Dict, title: str, vmin: float, vmax: float):
        """Draw fatband for a sublayer without spin resolution"""
        k_points = weights_data['k_points']
        energies = weights_data['energies']
        weights = weights_data['weights']['total']
        
        # Restructure data for plotting
        nk = len(self.orbital_data['data'])
        nbands = len(self.orbital_data['data'][0]['bands'])
        
        k_mesh = np.arange(nk)
        energy_mesh = energies.reshape(nk, nbands)
        weight_mesh = np.array(weights).reshape(nk, nbands)
        
        # First draw background band lines
        for band in range(nbands):
            mask = (energy_mesh[:, band] >= self.config.energy_window[0]) & (energy_mesh[:, band] <= self.config.energy_window[1])
            if np.any(mask):
                ax.plot(k_mesh[mask], energy_mesh[mask, band], 'k-', linewidth=0.5, alpha=0.3)
        
        # Collect valid points for plotting
        k_points_plot = []
        energies_plot = []
        sizes_plot = []
        weights_plot = []
        
        for band in range(nbands):
            for k in range(nk):
                energy = energy_mesh[k, band]
                weight = weight_mesh[k, band]
                
                if (self.config.energy_window[0] <= energy <= self.config.energy_window[1] and weight > 0.0001):
                    k_points_plot.append(k)
                    energies_plot.append(energy)
                    sizes_plot.append(max(5, min(80, weight * 200)))
                    weights_plot.append(weight)
        
        if len(k_points_plot) > 0:
            scatter = ax.scatter(k_points_plot, energies_plot, 
                               s=sizes_plot, c=weights_plot, cmap='viridis',
                               alpha=0.8, edgecolors='none', vmin=vmin, vmax=vmax)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15)
            cbar.set_label('Weight', rotation=270, labelpad=10, fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        
        # Set up plot
        ax.set_xlabel('K-point', fontsize=8)
        ax.set_ylabel('Energy (eV)', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(self.config.energy_window)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.8, linewidth=1)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, nk-1)
        ax.tick_params(labelsize=7)
    
    def _draw_sublayer_fatband_single_spin(self, ax, spin_data: Dict, title: str, vmin: float, vmax: float, colormap: str):
        """Draw fatband for a single spin channel of a sublayer"""
        k_points = spin_data['k_points']
        energies = spin_data['energies']
        
        # Sum all orbital weights for this spin channel
        total_weights = []
        for i in range(len(energies)):
            total = sum([orb_weights[i] for orb_weights in spin_data['weights'].values()])
            total_weights.append(total)
        
        # Restructure data for plotting
        nk = len(self.orbital_data['data'])
        nbands = len(self.orbital_data['data'][0]['bands'])
        
        # 构建k点坐标数组
        if self.config.use_kpath and self.kpath_coords is not None:
            k_mesh = self.kpath_coords[:nk] if len(self.kpath_coords) >= nk else np.arange(nk)
        else:
            k_mesh = np.arange(nk)
            
        energy_mesh = energies.reshape(nk, nbands)
        weight_mesh = np.array(total_weights).reshape(nk, nbands)
        
        # First draw background band lines
        bands_drawn = 0
        for band in range(nbands):
            mask = (energy_mesh[:, band] >= self.config.energy_window[0]) & (energy_mesh[:, band] <= self.config.energy_window[1])
            if np.any(mask):
                ax.plot(k_mesh[mask], energy_mesh[mask, band], 'k-', linewidth=0.5, alpha=0.3)
                bands_drawn += 1
        
        if bands_drawn == 0:
            print(f"Warning: No band lines drawn for {title}, energy window: {self.config.energy_window}")
        
        # Collect valid points for plotting
        k_points_plot = []
        energies_plot = []
        sizes_plot = []
        weights_plot = []
        
        for band in range(nbands):
            for k in range(nk):
                energy = energy_mesh[k, band]
                weight = weight_mesh[k, band]
                
                if (self.config.energy_window[0] <= energy <= self.config.energy_window[1] and weight > 0.0001):
                    k_points_plot.append(self._get_kpoint_coordinate(k))
                    energies_plot.append(energy)
                    sizes_plot.append(max(5, min(80, weight * 200)))
                    weights_plot.append(weight)
        
        if len(k_points_plot) > 0:
            scatter = ax.scatter(k_points_plot, energies_plot, 
                               s=sizes_plot, c=weights_plot, cmap=colormap,
                               alpha=0.8, edgecolors='none', vmin=vmin, vmax=vmax)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15)
            cbar.set_label('Weight', rotation=270, labelpad=10, fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        
        # 使用新的坐标轴设置方法（去掉xlabel，因为可能有多个子图）
        self._setup_axes(ax, title=title, xlabel="", fontsize=8, title_fontsize=9)
    
    def _check_if_spin_polarized(self) -> bool:
        """Check if the data is spin-polarized by looking at the data structure"""
        # Check first k-point, first band to see if 'up' and 'dn' keys exist
        if len(self.orbital_data['data']) > 0 and len(self.orbital_data['data'][0]['bands']) > 0:
            first_band = self.orbital_data['data'][0]['bands'][0]
            if 'comp' in first_band:
                comp = first_band['comp']
                for layer_name, layer_data in comp.items():
                    for sublayer_name, sublayer_data in layer_data.items():
                        if 'up' in sublayer_data and 'dn' in sublayer_data:
                            return True
        return False
    
    def _calculate_sublayer_total_contribution(self, sublayer: Dict) -> float:
        """Calculate total orbital contribution for a sublayer across all k-points and bands"""
        layer_name = sublayer['layer']
        sublayer_name = sublayer['sublayer']
        
        total_contribution = 0.0
        count = 0
        
        for k_idx, k_data in enumerate(self.orbital_data['data']):
            for band_idx, band_data in enumerate(k_data['bands']):
                energy = self.band_data[k_idx, band_idx]
                
                # Only consider states in energy window
                if not (self.config.energy_window[0] <= energy <= self.config.energy_window[1]):
                    continue
                
                composition = band_data['comp']
                if layer_name in composition and sublayer_name in composition[layer_name]:
                    sublayer_data = composition[layer_name][sublayer_name]
                    
                    if 'tot' in sublayer_data:
                        total_contribution += sublayer_data['tot']
                        count += 1
        
        return total_contribution / count if count > 0 else 0.0
    
    def _plot_overall_summary(self, filename: str):
        """绘制总体总结图"""
        print("  绘制总体总结图...")
        return None
    
    def run(self):
        """运行绘图工具"""
        try:
            self.load_data()
            
            if self.config.detailed:
                self.plot_detailed()
            elif self.config.interactive:
                selection = self.interactive_selection()
                # 根据selection绘制对应图表
            else:
                print("请指定 --interactive 或 --detailed 模式")
                
        except Exception as e:
            print(f"错误: {e}")
            return 1
        
        return 0


def main(argv=None, *, prog=None):
    """主函数"""
    parser = argparse.ArgumentParser(prog=prog, description="TAPW轨道成分能带绘图工具")
    parser.add_argument("result_dir", nargs="?", default=".", help="结果目录路径")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--valley", default="Gamma", help="谷标识 (default: Gamma)")
    parser.add_argument("--band", "--band-type", dest="band", default="CBM", choices=["CBM","VBM","cbm","vbm"],
                       help="能带类型 (CBM 或 VBM，default: CBM)")
    parser.add_argument("--orbital-dir", default="orbital_analysis", help="轨道分析结果目录")
    parser.add_argument("--output-dir", default="orbital_plots", help="输出目录")
    parser.add_argument("--energy-window", nargs=2, type=float, default=[-2.0, 2.0], 
                       help="能量窗口范围 eV，格式: min max (default: -2 2)")
    parser.add_argument("--figsize", nargs=2, type=int, default=[12, 8], 
                       help="图片尺寸，格式: width height (default: 12 8)")
    parser.add_argument("--dpi", type=int, default=300, help="图片分辨率 (default: 300)")
    parser.add_argument("--format", default="pdf", choices=["png", "pdf", "svg"], 
                       help="输出格式 (default: pdf)")
    parser.add_argument("--interactive", action="store_true", help="交互式选择模式")
    parser.add_argument("--detailed", action="store_true", help="详细模式：生成所有图表")
    
    # K路径相关选项
    parser.add_argument("--kpath-in", type=str, help="KPATH.in文件路径（用于高对称点标签）")
    parser.add_argument("--kpath-out", type=str, help="KPATH.out文件路径（用于k点坐标）")
    parser.add_argument("--no-kpath", action="store_true", help="禁用自动检测k路径文件")
    
    args = parser.parse_args(argv)
    
    # 解析参数
    energy_window = tuple(args.energy_window)
    figsize = tuple(args.figsize)
    
    config = PlotOrbitalConfig(
        config_file=args.config,
        result_dir=args.result_dir,
        orbital_dir=args.orbital_dir,
        output_dir=args.output_dir,
        valley=args.valley,
        band_type=(args.band or 'CBM').upper(),
        energy_window=energy_window,
        figsize=figsize,
        dpi=args.dpi,
        format=args.format,
        interactive=args.interactive,
        detailed=args.detailed,
        # K路径相关配置
        kpath_in=args.kpath_in if not args.no_kpath else None,
        kpath_out=args.kpath_out if not args.no_kpath else None
    )
    
    plotter = OrbitalPlotter(config)
    return plotter.run()


if __name__ == "__main__":
    exit(main()) 
