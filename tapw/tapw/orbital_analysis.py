"""
TAPW轨道成分分析模块
用于分析能带的轨道成分，包括不同层、子层、原子类型和轨道的贡献
"""
import numpy as np
import json
import time
import os
import re
from typing import Dict, List, Tuple, Any, Optional
from .config import ComputeConfig
from .read_pos_01 import StructureProcessor


class OrbitalAnalyzer:
    """TAPW轨道成分分析器"""
    
    def __init__(self, structure: StructureProcessor, config: ComputeConfig, valley_flag: str):
        """初始化轨道分析器
        
        Args:
            structure: 结构处理器
            config: 计算配置
            valley_flag: 谷标识
        """
        self.structure = structure
        self.config = config
        self.valley_flag = valley_flag
        
        if not self.config.TAPW:
            raise ValueError("轨道成分分析只在TAPW模式下可用")
    
    def parse_orbital_name(self, orb_name: str) -> List[str]:
        """解析轨道名称，返回详细的轨道列表
        
        Args:
            orb_name: 轨道名称字符串，如 "Mo7.0-s3p2d1"
            
        Returns:
            List[str]: 详细轨道名称列表，如 ["1s", "2s", "3s", "1px", "1py", "1pz", "2px", "2py", "2pz", "1dz2", "1dx2-y2", "1dxy", "1dxz", "1dyz"]
        """
        parts = orb_name.split("-")
        if len(parts) < 2:
            return []
        
        # 获取轨道部分
        suffix = parts[1]
        
        # 解析每个字符和数字对
        orb_counts = []
        i = 0
        while i < len(suffix):
            # 匹配字母
            char = suffix[i]
            # 匹配数字（如果存在）
            if i + 1 < len(suffix) and suffix[i + 1].isdigit():
                count = int(suffix[i + 1])
                orb_counts.append((char, count))
                i += 1
            else:
                orb_counts.append((char, 1))
            i += 1
        
        # 展开为详细轨道名称
        final_result = []
        for orb_type, count in orb_counts:
            if orb_type == "s":
                for i in range(count):
                    final_result.append(f"{i+1}s")
            elif orb_type == "p":
                for i in range(count):
                    final_result.extend([f"{i+1}px", f"{i+1}py", f"{i+1}pz"])
            elif orb_type == "d":
                for i in range(count):
                    final_result.extend([f"{i+1}dz2", f"{i+1}dx2-y2", f"{i+1}dxy", f"{i+1}dxz", f"{i+1}dyz"])
            elif orb_type == "f":
                for i in range(count):
                    final_result.extend([f"{i+1}fz3", f"{i+1}fxz2", f"{i+1}fyz2", 
                                       f"{i+1}fzx2", f"{i+1}fxyz", f"{i+1}fx3", f"{i+1}fy3x2"])
        
        return final_result
    
    def get_structure_info(self) -> Dict[str, Any]:
        """获取结构信息，包括层、子层、原子类型和轨道信息
        
        Returns:
            Dict: 包含完整结构信息的字典
        """
        df = self.structure.df
        structure_info = {"layers": []}
        
        # 按层组织信息
        for layer_idx in sorted(df['layer'].unique()):
            layer_data = {"layer": int(layer_idx + 1), "sublayers": []}
            
            # 按子层组织信息
            layer_df = df[df['layer'] == layer_idx]
            for sublayer_idx in sorted(layer_df['sublayer'].unique()):
                sublayer_df = layer_df[layer_df['sublayer'] == sublayer_idx]
                
                # 获取原子类型（假设每个子层只有一种原子类型）
                atom_type = sublayer_df['atom_type'].iloc[0]
                orb_name = sublayer_df['orb_name'].iloc[0]
                atom_count = len(sublayer_df)
                
                # 解析轨道信息
                orb_list = self.parse_orbital_name(orb_name)
                
                sublayer_data = {
                    "sub": int(sublayer_idx + 1),
                    "atom": atom_type,
                    "count": atom_count,
                    "orbs": orb_list
                }
                
                layer_data["sublayers"].append(sublayer_data)
            
            structure_info["layers"].append(layer_data)
        
        return structure_info
    
    def get_orbital_layout(self) -> Tuple[List[Dict], Dict[str, int]]:
        """获取轨道布局信息
        
        Returns:
            Tuple[List[Dict], Dict]: (轨道映射列表, 维度信息)
        """
        df = self.structure.df
        orbital_layout = []
        
        # 按层和子层顺序构建布局
        for layer_idx in sorted(df['layer'].unique()):
            layer_df = df[df['layer'] == layer_idx]
            
            for sublayer_idx in sorted(layer_df['sublayer'].unique()):
                sublayer_df = layer_df[layer_df['sublayer'] == sublayer_idx]
                
                atom_type = sublayer_df['atom_type'].iloc[0]
                orb_name = sublayer_df['orb_name'].iloc[0]
                atom_count = len(sublayer_df)
                orb_list = self.parse_orbital_name(orb_name)
                orb_per_atom = len(orb_list)
                
                layout_info = {
                    "layer": int(layer_idx + 1),
                    "sublayer": int(sublayer_idx + 1),
                    "atom_type": atom_type,
                    "atom_count": atom_count,
                    "orb_per_atom": orb_per_atom,
                    "orb_names": orb_list,
                    "total_orbs": atom_count * orb_per_atom
                }
                
                orbital_layout.append(layout_info)
        
        # 计算维度信息
        total_orbs = sum(layout["total_orbs"] for layout in orbital_layout)
        dims = {
            "total_orbs": total_orbs,
            "total_orbs_with_spin": total_orbs * 2 if self.structure.spin else total_orbs,
            "num_layers": len(df['layer'].unique()),
            "num_sublayers": sum(len(df[df['layer'] == layer]['sublayer'].unique()) 
                               for layer in df['layer'].unique())
        }
        
        return orbital_layout, dims
    
    def analyze_single_band(self, eigenvector: np.ndarray) -> Dict[str, Any]:
        """分析单个能带的轨道成分
        
        Args:
            eigenvector: 本征向量
            
        Returns:
            Dict: 轨道成分分析结果
        """
        # 获取轨道布局
        orbital_layout, dims = self.get_orbital_layout()
        
        # 计算轨道权重（波函数系数的模方）
        orbital_weights = np.abs(eigenvector) ** 2
        
        # 初始化结果字典
        composition = {}
        current_idx = 0
        
        for layout in orbital_layout:
            layer_key = f"L{layout['layer']}"
            sublayer_key = f"S{layout['sublayer']}"
            
            if layer_key not in composition:
                composition[layer_key] = {}
            
            atom_count = layout["atom_count"]
            orb_per_atom = layout["orb_per_atom"]
            total_orbs = layout["total_orbs"]
            
            if self.structure.spin:
                # 自旋极化情况
                # 自旋向上
                up_weights = orbital_weights[current_idx:current_idx + total_orbs]
                # 自旋向下
                dn_weights = orbital_weights[current_idx + dims["total_orbs"]:current_idx + dims["total_orbs"] + total_orbs]
                
                # 按轨道类型汇总
                up_orb_weights = []
                dn_orb_weights = []
                
                for orb_idx in range(orb_per_atom):
                    # 收集同一轨道类型在所有原子上的权重
                    up_orb_weight = 0
                    dn_orb_weight = 0
                    
                    for atom_idx in range(atom_count):
                        idx = atom_idx * orb_per_atom + orb_idx
                        if idx < len(up_weights):
                            up_orb_weight += up_weights[idx]
                        if idx < len(dn_weights):
                            dn_orb_weight += dn_weights[idx]
                    
                    up_orb_weights.append(round(float(up_orb_weight * 100), 2))
                    dn_orb_weights.append(round(float(dn_orb_weight * 100), 2))
                
                sublayer_data = {
                    "atom": layout["atom_type"],
                    "up": up_orb_weights,
                    "dn": dn_orb_weights,
                    "tot": round(sum(up_orb_weights) + sum(dn_orb_weights), 2)
                }
                
                current_idx += total_orbs
                
            else:
                # 非自旋极化情况
                weights = orbital_weights[current_idx:current_idx + total_orbs]
                
                # 按轨道类型汇总
                orb_weights = []
                for orb_idx in range(orb_per_atom):
                    orb_weight = 0
                    for atom_idx in range(atom_count):
                        idx = atom_idx * orb_per_atom + orb_idx
                        if idx < len(weights):
                            orb_weight += weights[idx]
                    orb_weights.append(round(float(orb_weight * 100), 2))
                
                sublayer_data = {
                    "atom": layout["atom_type"],
                    "weights": orb_weights,
                    "tot": round(sum(orb_weights), 2)
                }
                
                current_idx += total_orbs
            
            composition[layer_key][sublayer_key] = sublayer_data
        
        return composition
    
    def save_analysis(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray, 
                     kpoints: np.ndarray, output_path: str):
        """保存完整的轨道成分分析结果
        
        Args:
            eigenvalues: 本征值数组 (nk, nbands)
            eigenvectors: 本征向量数组 (nk, norb, nbands) 或 (nk, norb)
            kpoints: k点坐标数组 (nk, 3)
            output_path: 输出路径
        """
        print("开始轨道成分分析...")
        
        # 创建输出目录
        orbital_dir = os.path.join(output_path, "orbital_analysis")
        os.makedirs(orbital_dir, exist_ok=True)
        
        # 获取结构信息
        structure_info = self.get_structure_info()
        
        # 构建JSON数据
        json_data = {
            "meta": {
                "valley": self.valley_flag,
                "n_g": self.config.n_g,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "spin": self.structure.spin
            },
            "structure": structure_info["layers"],
            "data": []
        }
        
        # 确定数据维度
        nk = len(kpoints)
        if len(eigenvectors.shape) == 3:
            # (nk, norb, nbands)
            _, norb, nbands = eigenvectors.shape
            nbands = min(nbands, eigenvalues.shape[1])
        else:
            # (nk, norb) - 只有一个能带
            norb = eigenvectors.shape[1]
            nbands = 1
        
        print(f"分析 {nk} 个k点，每个k点 {nbands} 个能带")
        
        # 分析每个k点
        for k_idx in range(nk):
            if k_idx >= len(eigenvalues):
                break
                
            kpoint_data = {
                "k": k_idx,
                "coord": [round(float(kpoints[k_idx][0]), 6), 
                         round(float(kpoints[k_idx][1]), 6), 
                         round(float(kpoints[k_idx][2]), 6)],
                "bands": []
            }
            
            # 分析每个能带
            for band_idx in range(min(nbands, len(eigenvalues[k_idx]))):
                # 获取本征向量
                if len(eigenvectors.shape) == 3:
                    eigenvector = eigenvectors[k_idx, :, band_idx]
                else:
                    eigenvector = eigenvectors[k_idx, :]
                
                # 分析轨道成分
                composition = self.analyze_single_band(eigenvector)
                
                band_data = {
                    "b": band_idx,
                    "e": round(float(eigenvalues[k_idx][band_idx]), 6),
                    "comp": composition
                }
                
                kpoint_data["bands"].append(band_data)
            
            json_data["data"].append(kpoint_data)
            
            # 显示进度
            if (k_idx + 1) % 10 == 0 or k_idx == nk - 1:
                print(f"已处理 {k_idx + 1}/{nk} 个k点")
        
        # 保存JSON文件
        suffix = "_2d" if getattr(self.config, "mode", None) == "chern" else ""
        if getattr(self.config, "mode", None) == "chern" and hasattr(self.config, "num_chern"):
            suffix += f"_{self.config.num_chern}"
        
        json_filename = os.path.join(orbital_dir, f"orbital_data_{self.config.band_type}_{self.valley_flag}_valley{suffix}.json")
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"轨道成分分析结果已保存到: {json_filename}")
        
        # 生成汇总报告
        self._generate_summary_report(json_data, orbital_dir, suffix)
    
    def _generate_summary_report(self, json_data: Dict, output_dir: str, suffix: str):
        """生成汇总报告
        
        Args:
            json_data: 完整的轨道分析数据
            output_dir: 输出目录
            suffix: 文件后缀
        """
        summary_filename = os.path.join(output_dir, f"orbital_summary_{self.config.band_type}_{self.valley_flag}_valley{suffix}.txt")
        
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write("# TAPW轨道成分汇总统计\n")
            f.write(f"# Valley: {self.valley_flag}, n_g: {self.config.n_g}\n")
            f.write(f"# 时间: {json_data['meta']['time']}\n")
            f.write(f"# 自旋极化: {json_data['meta']['spin']}\n\n")
            
            # 结构信息
            f.write("## 结构信息\n")
            for layer in json_data['structure']:
                f.write(f"Layer {layer['layer']}:\n")
                for sublayer in layer['sublayers']:
                    orb_str = " ".join(sublayer['orbs'])
                    f.write(f"  Sub{sublayer['sub']}({sublayer['atom']}): {len(sublayer['orbs'])}orb [{orb_str}]\n")
                f.write("\n")
            
            # 统计信息
            f.write("## 子层贡献统计 (平均值)\n")
            f.write("Layer Sub Atom     Total(%)  主要轨道贡献\n")
            f.write("-" * 60 + "\n")
            
            # 计算每个子层的平均贡献
            sublayer_stats = {}
            for kpoint_data in json_data['data']:
                for band_data in kpoint_data['bands']:
                    comp = band_data['comp']
                    for layer_key, layer_data in comp.items():
                        for sub_key, sub_data in layer_data.items():
                            key = f"{layer_key}{sub_key}"
                            if key not in sublayer_stats:
                                sublayer_stats[key] = {
                                    'atom': sub_data['atom'],
                                    'totals': [],
                                    'orb_contributions': []
                                }
                            sublayer_stats[key]['totals'].append(sub_data['tot'])
                            if self.structure.spin:
                                sublayer_stats[key]['orb_contributions'].append(
                                    [u + d for u, d in zip(sub_data['up'], sub_data['dn'])]
                                )
                            else:
                                sublayer_stats[key]['orb_contributions'].append(sub_data['weights'])
            
            # 输出统计结果
            for key, stats in sublayer_stats.items():
                layer_num = key[1]
                sub_num = key[3]
                atom = stats['atom']
                avg_total = np.mean(stats['totals'])
                
                # 计算主要轨道贡献
                avg_orb_contrib = np.mean(stats['orb_contributions'], axis=0)
                # 找到前3个最大贡献的轨道
                top_indices = np.argsort(avg_orb_contrib)[-3:][::-1]
                
                # 获取轨道名称
                orb_names = None
                for layer in json_data['structure']:
                    if layer['layer'] == int(layer_num):
                        for sublayer in layer['sublayers']:
                            if sublayer['sub'] == int(sub_num):
                                orb_names = sublayer['orbs']
                                break
                        break
                
                if orb_names:
                    top_orbs = [f"{orb_names[i]}({avg_orb_contrib[i]:.1f}%)" for i in top_indices if avg_orb_contrib[i] > 0.1]
                    top_orbs_str = " ".join(top_orbs[:3])
                else:
                    top_orbs_str = "N/A"
                
                f.write(f"{layer_num:>5} {sub_num:>3} {atom:>6} {avg_total:>10.1f}  {top_orbs_str}\n")
        
        print(f"汇总报告已保存到: {summary_filename}")