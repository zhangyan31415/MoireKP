#!/usr/bin/env python3
"""
TAPW 轨道成分分析工具

用法示例（建议使用模块方式运行）:

- 基本用法（指定结果目录与配置文件）
  python -m tapw.orbital_analysis_tool /path/to/result --config /path/to/config.yaml

- 指定谷与能带类型（CBM 或 VBM）
  python -m tapw.orbital_analysis_tool /path/to/result --config /path/to/config.yaml --valley Gamma --band cbm
  python -m tapw.orbital_analysis_tool /path/to/result --config /path/to/config.yaml --valley M1 --band vbm

- 静默模式与输出目录
  python -m tapw.orbital_analysis_tool /path/to/result --config config.yaml --band CBM --output out_dir -q

文件命名与自动检测:
- 新版文件结构（推荐）: 结果目录包含子目录 band/
  band/band_{CBM|VBM}_{Valley}_valley.txt
  band/vec_{CBM|VBM}_{Valley}_valley.npy
  例如: band/band_CBM_Gamma_valley.txt, band/vec_VBM_M1_valley.npy

- 旧版兼容: 也会尝试在 result_dir 或 result_dir/band_data 下匹配旧式文件名

输出:
- 结果保存在 --output 目录下，命名为 orbital_data_{CBM|VBM}_{Valley}.json
"""
import argparse
import os
import sys
import numpy as np
import json
import time
import re
from typing import Dict, List, Tuple, Any, Optional, cast
from dataclasses import dataclass, field
from json import JSONEncoder
from tqdm import tqdm

# 导入TAPW模块
from .config import ComputeConfig, Config
from .io.structure import OpenMXFile, StructureProcessorSpglib


@dataclass
class OrbitalAnalysisConfig:
    """轨道分析配置"""
    config_file: str  # TAPW配置文件路径
    result_dir: str  # TAPW计算结果目录
    output_dir: str = "orbital_analysis"  # 输出目录
    valley: str = "auto"  # 谷标识，auto表示自动检测
    band_type: str = "CBM"  # 能带类型: CBM 或 VBM
    spin_polarized: bool = True  # 是否为自旋极化计算
    verbose: bool = True  # 详细输出


def compact_json_arrays(json_str):
    """
    将JSON字符串中的数组压缩到单行
    """
    # 匹配数字数组的正则表达式
    number_pattern = r'\[\s*\n\s*((?:(?:\d+(?:\.\d+)?),?\s*\n?\s*)+)\s*\n\s*\]'
    
    def replace_number_array(match):
        content = match.group(1)
        numbers = re.findall(r'\d+(?:\.\d+)?', content)
        return '[' + ', '.join(numbers) + ']'
    
    # 匹配字符串数组的正则表达式 (包含换行的字符串数组)
    string_pattern = r'\[\s*\n\s*((?:(?:"[^"]*"),?\s*\n?\s*)+)\s*\n\s*\]'
    
    def replace_string_array(match):
        content = match.group(1)
        # 提取所有带引号的字符串
        strings = re.findall(r'"[^"]*"', content)
        return '[' + ', '.join(strings) + ']'
    
    # 先处理数字数组，再处理字符串数组
    result = re.sub(number_pattern, replace_number_array, json_str)
    result = re.sub(string_pattern, replace_string_array, result)
    
    return result

def convert_numpy_types(obj):
    """
    递归转换numpy类型为Python原生类型
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class OrbitalAnalyzer:
    """TAPW轨道成分分析器"""
    
    def __init__(self, config: OrbitalAnalysisConfig):
        """初始化轨道分析器"""
        self.config = config
        self.structure = None
        self.compute_config = None
        self.valley_flag = None
        self._first_analysis = True  # 用于控制详细信息的输出
        # 规范化 band_type
        self.band_type = (config.band_type or "CBM").upper()

        # 关键：将 result_dir 规范化为绝对路径。
        # 否则当用户在 band/ 目录里用默认 result_dir="." 运行时，
        # os.path.dirname(".") == ""，会导致上级目录（例如 Q_shell_4/）中的 g_vec_list*.npy 无法被找到。
        self.config.result_dir = os.path.abspath(self.config.result_dir)
        
        # 检查输入目录
        if not os.path.exists(self.config.result_dir):
            raise FileNotFoundError(f"结果目录不存在: {self.config.result_dir}")
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        if self.config.verbose:
            print(f"轨道分析器初始化完成")
            print(f"输入目录: {self.config.result_dir}")
            print(f"输出目录: {config.output_dir}")
    
    def detect_valleys_and_files(self) -> Dict[str, Dict[str, str]]:
        """检测可用的谷和相关文件（支持新版 band/ 目录和旧版命名）"""
        valleys: Dict[str, Dict[str, str]] = {}

        # 优先扫描新版 band/ 目录
        band_dir = os.path.join(self.config.result_dir, "band")
        if os.path.isdir(band_dir):
            for file in os.listdir(band_dir):
                if not (file.startswith("band_") and file.endswith(".txt")):
                    continue
                # 期望格式: band_{CBM|VBM}_{Valley}_valley.txt
                m = re.match(rf"band_({self.band_type})_([A-Za-z0-9']+)(?:_valley)?\.txt", file)
                if not m:
                    # 若 band_type 不匹配则跳过
                    continue
                band_type, valley_name = m.group(1), m.group(2)
                band_path = os.path.join(band_dir, file)
                # 对应 vec 文件
                vec_candidate_names = [
                    f"vec_{band_type}_{valley_name}_valley.npy",
                    f"vec_{band_type}_{valley_name}.npy",
                ]
                vec_path = None
                for vc in vec_candidate_names:
                    p = os.path.join(band_dir, vc)
                    if os.path.exists(p):
                        vec_path = p
                        break
                if vec_path:
                    valleys[valley_name] = {
                        "band_file": band_path,
                        "vec_file": vec_path,
                        "valley_flag": valley_name,
                    }

        # 兼容旧版：band_data 子目录
        if not valleys:
            band_data_dir = os.path.join(self.config.result_dir, "band_data")
            if os.path.isdir(band_data_dir):
                for file in os.listdir(band_data_dir):
                    if not (file.startswith("band_") and file.endswith(".txt")):
                        continue
                    band_path = os.path.join(band_data_dir, file)
                    # 优先解析新格式：band_{CBM|VBM}_{Valley}(_valley).txt
                    m = re.match(r"band_(CBM|VBM)_([A-Za-z0-9']+)(?:_valley)?\.txt", file)
                    if m:
                        bt, valley_name = m.group(1), m.group(2)
                        if bt != self.band_type:
                            continue
                        vec_candidate_names = [
                            f"vec_{bt}_{valley_name}_valley.npy",
                            f"vec_{bt}_{valley_name}.npy",
                        ]
                        vec_path = None
                        for vc in vec_candidate_names:
                            p = os.path.join(band_data_dir, vc)
                            if os.path.exists(p):
                                vec_path = p
                                break
                        if vec_path:
                            valleys[valley_name] = {
                                "band_file": band_path,
                                "vec_file": vec_path,
                                "valley_flag": valley_name,
                            }
                        continue

                    # 旧版兜底：无 band_type 前缀
                    valley_name = file.replace("band_", "").replace("_valley.txt", "").replace(".txt", "")
                    vec_file = file.replace("band_", "vec_").replace(".txt", ".npy")
                    vec_path = os.path.join(self.config.result_dir, vec_file)
                    if os.path.exists(vec_path):
                        valleys[valley_name] = {"band_file": band_path, "vec_file": vec_path, "valley_flag": valley_name}

        # 兼容旧版：直接在根目录
        if not valleys:
            for file in os.listdir(self.config.result_dir):
                if not (file.startswith("band_") and file.endswith(".txt")):
                    continue
                band_path = os.path.join(self.config.result_dir, file)

                # 优先解析新格式：band_{CBM|VBM}_{Valley}(_valley).txt
                m = re.match(r"band_(CBM|VBM)_([A-Za-z0-9']+)(?:_valley)?\.txt", file)
                if m:
                    bt, valley_name = m.group(1), m.group(2)
                    if bt != self.band_type:
                        continue
                    vec_candidate_names = [
                        f"vec_{bt}_{valley_name}_valley.npy",
                        f"vec_{bt}_{valley_name}.npy",
                    ]
                    vec_path = None
                    for vc in vec_candidate_names:
                        p = os.path.join(self.config.result_dir, vc)
                        if os.path.exists(p):
                            vec_path = p
                            break
                    if vec_path:
                        valleys[valley_name] = {"band_file": band_path, "vec_file": vec_path, "valley_flag": valley_name}
                    continue

                # 旧版兜底：无 band_type 前缀
                valley_name = file.replace("band_", "").replace("_valley.txt", "").replace(".txt", "")
                vec_file = file.replace("band_", "vec_").replace(".txt", ".npy")
                vec_path = os.path.join(self.config.result_dir, vec_file)
                if os.path.exists(vec_path):
                    valleys[valley_name] = {"band_file": band_path, "vec_file": vec_path, "valley_flag": valley_name}

        if self.config.verbose and valleys:
            print(f"已检测到 {len(valleys)} 个谷: {sorted(list(valleys.keys()))}")
        return valleys
    
    def load_structure_from_existing_files(self) -> 'StructureProcessorSpglib':
        """从现有文件中加载并处理结构信息"""
        # 加载TAPW配置文件
        if not os.path.exists(self.config.config_file):
            raise FileNotFoundError(f"配置文件不存在: {self.config.config_file}")
            
        tapw_config = Config.from_yaml(self.config.config_file)
        
        # 获取 band_type 信息（命令行优先，其次配置文件）
        if not self.band_type:
            self.band_type = getattr(tapw_config.compute, 'band_type', 'CBM').upper()
        
        if self.config.verbose:
            print(f"加载TAPW配置文件: {self.config.config_file}")
            print(f"输入文件: {tapw_config.paths.input_file}")
            print(f"扭转指数: {tapw_config.twist.twist_index_m}")
            print(f"自旋: {tapw_config.twist.spin}")
            print(f"带类型: {self.band_type}")
        
        # 检查输入文件是否存在
        input_file = tapw_config.paths.input_file
        if not os.path.exists(input_file):
            # 尝试相对于配置文件的路径
            config_dir = os.path.dirname(self.config.config_file)
            input_file = os.path.join(config_dir, input_file)
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"输入文件不存在: {tapw_config.paths.input_file}")
        
        if self.config.verbose:
            print(f"找到结构文件: {input_file}")
        
        # 初始化OpenMXFile
        openmx_structure = OpenMXFile(
            file_path=input_file,
            twist_index=tapw_config.twist.twist_index_m,
            spin=tapw_config.twist.spin
        )

        # 与主流程保持一致：使用 spglib 版本的结构处理器（type_structure 已废弃）
        processor = StructureProcessorSpglib(
            input_data=openmx_structure.sorted_species_coordinates,
            num_layers=tapw_config.twist.num_layers,
            twist_layer=tapw_config.twist.twist_layer,
            layer_eps=tapw_config.cluster.layer_eps,
            layer_min_samples=tapw_config.cluster.layer_min_samples,
            sublayer_eps=tapw_config.cluster.sublayer_eps,
            sublayer_min_samples=tapw_config.cluster.sublayer_min_samples,
            atom_eps=tapw_config.cluster.atom_eps,
            atom_min_samples=tapw_config.cluster.atom_min_samples,
            period=tapw_config.cluster.period,
            k_max=tapw_config.cluster.k_max,
            spin=openmx_structure.spin,
            twist_index=tapw_config.twist.twist_index_m,
            Tmat=openmx_structure.Tmat,
            reciprocal_Tmat=openmx_structure.reciprocal_Tmat,
        )
        
        # 执行处理过程以生成df
        processor.process()
        
        if self.config.verbose:
            if processor.df is None:
                raise RuntimeError("结构处理完成但 processor.df 为空（StructureProcessorSpglib.process 失败）")
            print(f"成功加载并处理结构: {len(processor.df)} 个原子")
            print(f"层数: {tapw_config.twist.num_layers}")
            print(f"扭转层: {tapw_config.twist.twist_layer}")
        
        return processor

    def extract_structure_info_from_processor(self, processor: 'StructureProcessorSpglib') -> Dict:
        """从StructureProcessor对象提取结构信息用于轨道分析"""
        if processor is None:
            raise ValueError("StructureProcessor对象为空，无法提取结构信息")
        
        if self.config.verbose:
            print(f"从StructureProcessor提取结构信息...")
        
        # 使用StructureProcessor的df，这已经包含了正确的层和子层信息
        df = processor.df
        if df is None:
            raise RuntimeError("processor.df 为空：结构处理未成功完成")
        df = cast(Any, df)
        
        if self.config.verbose:
            print(f"DataFrame列: {df.columns.tolist()}")
            print(f"总原子数: {len(df)}")
            if 'layer' in df.columns:
                print(f"层分布: {df['layer'].value_counts().to_dict()}")
            if 'sublayer' in df.columns:
                print(f"子层分布: {df['sublayer'].value_counts().to_dict()}")
        
        # 组织结构信息
        structure_info = {"layers": [], "sublayers": []}
        
        # 按层和子层组织
        layers = sorted(df['layer'].unique()) if 'layer' in df.columns else [1, 2]
        
        for layer_num in layers:
            layer_df = df[df['layer'] == layer_num] if 'layer' in df.columns else df.iloc[len(df)//2*(layer_num-1):len(df)//2*layer_num]
            
            # 按子层和原子种类分组
            if 'sublayer' in df.columns:
                sublayers = sorted(layer_df['sublayer'].unique())
                layer_sublayers = []
                
                for sublayer_num in sublayers:
                    sublayer_df = layer_df[layer_df['sublayer'] == sublayer_num]
                    
                    # 按原子种类分组
                    for species in sorted(sublayer_df['species'].unique()):
                        species_df = sublayer_df[sublayer_df['species'] == species]
                        
                        # 获取轨道信息
                        if 'orb_name' in species_df.columns:
                            orb_name = species_df['orb_name'].iloc[0]
                            orb_list = self.parse_orbital_name(orb_name)
                        else:
                            # 默认轨道（如果没有轨道信息）
                            orb_list = ["1s", "2s", "3s", "1px", "1py", "1pz", "2px", "2py", "2pz", 
                                       "1dz2", "1dx2-y2", "1dxy", "1dxz", "1dyz"]
                        
                        # 扁平化的子层信息
                        flat_sublayer_info = {
                            "layer": layer_num,
                            "sublayer": sublayer_num,
                            "atom": species,
                            "count": len(species_df),
                            "orbs": orb_list
                        }
                        structure_info["sublayers"].append(flat_sublayer_info)
                        
                        # 嵌套的子层信息
                        nested_sublayer_info = {
                            "sub": sublayer_num,
                            "atom": species,
                            "count": len(species_df),
                            "orbs": orb_list
                        }
                        layer_sublayers.append(nested_sublayer_info)
            else:
                # 如果没有子层信息，按原子种类直接分组
                layer_sublayers = []
                for sub_idx, species in enumerate(sorted(layer_df['species'].unique()), 1):
                    species_df = layer_df[layer_df['species'] == species]
                    
                    # 获取轨道信息
                    if 'orb_name' in species_df.columns:
                        orb_name = species_df['orb_name'].iloc[0]
                        orb_list = self.parse_orbital_name(orb_name)
                    else:
                        orb_list = ["1s", "2s", "3s", "1px", "1py", "1pz", "2px", "2py", "2pz", 
                                   "1dz2", "1dx2-y2", "1dxy", "1dxz", "1dyz"]
                    
                    # 扁平化的子层信息
                    flat_sublayer_info = {
                        "layer": layer_num,
                        "sublayer": sub_idx,
                        "atom": species,
                        "count": len(species_df),
                        "orbs": orb_list
                    }
                    structure_info["sublayers"].append(flat_sublayer_info)
                    
                    # 嵌套的子层信息
                    nested_sublayer_info = {
                        "sub": sub_idx,
                        "atom": species,
                        "count": len(species_df),
                        "orbs": orb_list
                    }
                    layer_sublayers.append(nested_sublayer_info)
            
            layer_data = {
                "layer": layer_num,
                "sublayers": layer_sublayers
            }
            structure_info["layers"].append(layer_data)
        
        if self.config.verbose:
            print(f"提取的结构信息:")
            for layer in structure_info["layers"]:
                print(f"  层 {layer['layer']}:")
                for sublayer in layer["sublayers"]:
                    print(f"    子层 {sublayer['sub']}: {sublayer['atom']} x{sublayer['count']}, 轨道: {len(sublayer['orbs'])}个")
        
        return structure_info
    
    def parse_orbital_name(self, orb_name: str) -> List[str]:
        """解析轨道名称，返回详细的轨道列表"""
        parts = orb_name.split("-")
        if len(parts) < 2:
            return []
        
        suffix = parts[1]
        orb_counts = []
        i = 0
        while i < len(suffix):
            char = suffix[i]
            if i + 1 < len(suffix) and suffix[i + 1].isdigit():
                count = int(suffix[i + 1])
                orb_counts.append((char, count))
                i += 1
            else:
                orb_counts.append((char, 1))
            i += 1
        
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
    
    def _build_orbital_mapping_cache(self, processor: 'StructureProcessorSpglib', structure_info: Dict) -> Dict:
        """构建轨道映射缓存，避免重复计算"""
        df = processor.df
        if df is None:
            raise RuntimeError("processor.df 为空：无法构建轨道映射缓存")
        df = cast(Any, df)
        mapping_cache = {}
        
        if self.config.verbose:
            print(f"构建轨道映射缓存，共{len(df)}个原子")
        
        # 按layer和sublayer分组处理
        for layer_info in structure_info["layers"]:
            layer_num = layer_info['layer']
            layer_name = f"L{layer_num}"
            
            for sublayer_info in layer_info["sublayers"]:
                sublayer_num = sublayer_info['sub']
                sublayer_name = f"S{sublayer_num}"
                species = sublayer_info['atom']
                
                # 找到对应的原子行
                mask = (df['layer'] == layer_num) & (df['sublayer'] == sublayer_num) & (df['species'] == species)
                matching_rows = df[mask]
                
                if len(matching_rows) > 0:
                    # 收集所有匹配原子的轨道索引
                    all_global_indices = []
                    orb_list = sublayer_info['orbs']
                    
                    for _, row in matching_rows.iterrows():
                        orb_global_indices = row['orb_global_index']
                        all_global_indices.extend(orb_global_indices)
                    
                    # 转换为numpy数组以提高性能
                    global_indices_array = np.array(all_global_indices, dtype=int)
                    
                    mapping_cache[(layer_name, sublayer_name)] = {
                        'global_indices': global_indices_array,
                        'orb_list': orb_list,
                        'species': species
                    }
        
        if self.config.verbose:
            print(f"轨道映射缓存构建完成，包含{len(mapping_cache)}个子层")
        
        return mapping_cache
    
    def analyze_from_files(self, band_file: str, vec_file: str, valley_flag: str):
        """从文件分析轨道成分"""
        if self.config.verbose:
            print(f"\n分析 {valley_flag} 谷的轨道成分...")
            print(f"能带文件: {band_file}")
            print(f"特征向量文件: {vec_file}")
        
        # 加载能带数据
        try:
            band_data = np.loadtxt(band_file)
            if self.config.verbose:
                print(f"加载能带数据: {band_data.shape}")
        except Exception as e:
            print(f"加载能带数据失败: {e}")
            return
        
        # 加载特征向量数据
        try:
            vec_data = np.load(vec_file)
            if self.config.verbose:
                print(f"加载特征向量数据: {vec_data.shape}")
        except Exception as e:
            print(f"加载特征向量数据失败: {e}")
            return
        
        # 加载真实的结构信息
        processor = self.load_structure_from_existing_files()
        
        # 从结构处理器提取轨道分析所需的信息
        structure_info = self.extract_structure_info_from_processor(processor)
        
        # 重置轨道映射缓存（每个新的分析会话都重新构建）
        if hasattr(self, '_orbital_mapping_cache'):
            delattr(self, '_orbital_mapping_cache')
        
        # 执行轨道分析
        result = self.perform_orbital_analysis(band_data, vec_data, structure_info, valley_flag, processor)
        
        # 保存结果
        self.save_results(result, valley_flag)
    
    def perform_orbital_analysis(self, band_data: np.ndarray, vec_data: np.ndarray, 
                                structure_info: Dict, valley_flag: str, processor: 'StructureProcessorSpglib') -> Dict:
        """执行轨道成分分析"""
        nk, nbands = band_data.shape
        
        # 确定特征向量数据的维度
        if len(vec_data.shape) == 3:
            # (nk, norb, nbands)
            vec_nk, vec_norb, vec_nbands = vec_data.shape
            actual_nbands = min(nbands, vec_nbands)
        else:
            # (nk, norb) - 只有一个能带
            vec_nk, vec_norb = vec_data.shape
            actual_nbands = 1
        
        # 处理所有k点和能带
        k_start, k_end = 0, nk
        band_start, band_end = 0, actual_nbands
        
        if self.config.verbose:
            print(f"分析 {nk} 个k点，{actual_nbands} 条能带")
        
        # 构建分析结果
        result = {
            "meta": {
                "valley": valley_flag,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "total_kpoints": nk,
                "total_bands": actual_nbands
            },
            "structure": structure_info["layers"],
            "data": []
        }
        
        # 分析每个k点
        for k_idx in tqdm(range(k_start, k_end), desc="处理k点"):
            # if k_idx % 10 == 0 and self.config.verbose:
            #     print(f"处理k点 {k_idx}/{k_end-1}")
            
            kpoint_data = {
                "k": k_idx,
                "coordinates": [float(k_idx), 0.0, 0.0],  # 临时坐标，后续可以从band_data中获取实际坐标
                "bands": []
            }
            
            # 分析每个能带
            for band_idx in range(band_start, band_end):
                # 获取特征向量
                if len(vec_data.shape) == 3:
                    eigenvector = vec_data[k_idx, :, band_idx]
                else:
                    eigenvector = vec_data[k_idx, :]
                
                # 计算轨道权重
                orbital_weights = np.abs(eigenvector) ** 2
                norb_per_spin = len(orbital_weights) // 2
                # print("sum of orbital weights up,dn:", np.sum(orbital_weights), np.sum(orbital_weights[:norb_per_spin]), np.sum(orbital_weights[norb_per_spin:]))
                
                # 分析轨道成分
                composition = self.analyze_orbital_composition(orbital_weights, structure_info, processor, valley_flag)
                
                band_data_item = {
                    "b": band_idx,
                    "e": round(float(band_data[k_idx, band_idx]), 6),
                    "comp": composition
                }
                
                kpoint_data["bands"].append(band_data_item)
            
            result["data"].append(kpoint_data)
        
        return result
    
    def _normalize_valley_for_gvec(self, valley_flag: Optional[str]) -> Optional[str]:
        """将 valley_flag 归一化为 g_vec_list 文件使用的 valley 名称。
        
        常见情况：
        - band 文件名在旧扫描逻辑下可能给出 "VBM_K2"/"CBM_Gamma"
        - g_vec_list 文件通常使用 "K2"/"Gamma"（不带 CBM/VBM 前缀）
        """
        if not valley_flag:
            return None
        v = str(valley_flag)
        for prefix in ("CBM_", "VBM_"):
            if v.startswith(prefix):
                v = v[len(prefix):]
                break
        return v

    def _infer_gvector_count_from_dimensions(self, norb_per_spin: int, structure_info: Dict) -> Optional[int]:
        """在没有 g_vec_list 文件时，用维度自洽推断 n_gvec。
        
        TAPW 的系数组织满足：
          norb_per_spin == n_gvec * sum_layer( sum_sublayer( len(orbs) ) )
        """
        try:
            total_orbs_per_gvec_all_layers = 0
            for layer_info in structure_info.get("layers", []):
                for sub in layer_info.get("sublayers", []):
                    total_orbs_per_gvec_all_layers += len(sub.get("orbs", []))
            if total_orbs_per_gvec_all_layers <= 0:
                return None
            if norb_per_spin % total_orbs_per_gvec_all_layers != 0:
                return None
            n_gvec = norb_per_spin // total_orbs_per_gvec_all_layers
            return int(n_gvec) if n_gvec > 0 else None
        except Exception:
            return None

    def _get_gvector_count(self, norb_per_spin: int, valley_flag: Optional[str] = None) -> Optional[int]:
        """从G向量文件中读取G向量数量
        
        Args:
            norb_per_spin: 每个自旋通道的轨道数
            valley_flag: 谷标识，用于匹配对应的G向量文件
            
        Returns:
            Optional[int]: G向量数量，如果无法确定则返回None
        """
        try:
            # 尝试从结果目录中查找对应valley的G向量文件
            import glob

            result_dir = os.path.abspath(self.config.result_dir)
            parent_dir = os.path.dirname(result_dir)
            grandparent_dir = os.path.dirname(parent_dir)
            search_roots = []
            for d in (result_dir, parent_dir, grandparent_dir):
                if d and d not in search_roots:
                    search_roots.append(d)

            v = self._normalize_valley_for_gvec(valley_flag)
            
            # 构建匹配特定valley的文件模式
            if v:
                patterns = []
                for root in search_roots:
                    # 新/旧格式兼容：有些版本会带 band_type，有些不带
                    patterns.extend(
                        [
                            os.path.join(root, f"g_vec_list_*_{self.band_type}_{v}_1layer.npy"),
                            os.path.join(root, f"**/g_vec_list_*_{self.band_type}_{v}_1layer.npy"),
                            os.path.join(root, f"g_vec_list_*_{v}_1layer.npy"),
                            os.path.join(root, f"**/g_vec_list_*_{v}_1layer.npy"),
                        ]
                    )
            else:
                # 回退到通用模式
                patterns = []
                for root in search_roots:
                    patterns.extend(
                        [
                            os.path.join(root, "g_vec_list_*_1layer.npy"),
                            os.path.join(root, "**/g_vec_list_*_1layer.npy"),
                        ]
                    )
            
            gvec_file = None
            for pattern in patterns:
                files = glob.glob(pattern, recursive=True)
                if files:
                    gvec_file = files[0]  # 取第一个匹配的文件
                    break
            
            if gvec_file and os.path.exists(gvec_file):
                import numpy as np
                gvec_data = np.load(gvec_file)
                n_gvec = len(gvec_data)
                
                if self.config.verbose and self._first_analysis:
                    print(f"从文件读取G向量数: {n_gvec} (文件: {os.path.basename(gvec_file)})")
                
                return n_gvec
            else:
                # 不在这里做“按原子数估算”的旧逻辑（对 TAPW 平面波组织并不可靠）；
                # 更稳健的兜底推断在 analyze_orbital_composition() 里完成。
                if self.config.verbose and self._first_analysis:
                    print(f"无法确定G向量数，未找到valley {v or valley_flag}的G向量文件")
                return None
                
        except Exception as e:
            if self.config.verbose:
                print(f"读取G向量数时出错: {e}")
            return None

    def analyze_orbital_composition(self, orbital_weights: np.ndarray, structure_info: Dict, processor: 'StructureProcessorSpglib', valley_flag: Optional[str] = None) -> Dict:
        """分析单个特征向量的轨道成分 - 基于正确的TAPW平面波基矢理解"""
        composition = {}
        total_norb = len(orbital_weights)
        
        # 缓存结构信息用于G向量计算
        self._structure_info_cache = structure_info
        
        if self.config.verbose and self._first_analysis:
            print(f"TAPW轨道分析: 总轨道数={total_norb}")
            
            # 获取G向量数量
            n_gvec = self._get_gvector_count(total_norb // 2, valley_flag)
            if n_gvec:
                print(f"检测到G向量数: {n_gvec}")
                
                # 验证维度
                total_orbs_per_layer = sum(len(sub["orbs"]) for sub in structure_info["layers"][0]["sublayers"])
                expected_norb = 2 * len(structure_info["layers"]) * n_gvec * total_orbs_per_layer  # 2自旋 × 层数 × n_gvec × 每层轨道数
                print(f"期望轨道数: {expected_norb}, 实际轨道数: {total_norb}")
                print(f"每层轨道数: {total_orbs_per_layer}, G向量数: {n_gvec}")
                
            self._first_analysis = False

        # TAPW自旋极化：9632 = 4816(up) + 4816(down)
        if self.config.spin_polarized:
            if total_norb % 2 != 0:
                raise ValueError("自旋极化计算的总轨道数必须是偶数")
            norb_per_spin = total_norb // 2
            up_weights = orbital_weights[:norb_per_spin]
            dn_weights = orbital_weights[norb_per_spin:]
        else:
            norb_per_spin = total_norb
            up_weights = orbital_weights
            dn_weights = None

        # 获取G向量数量
        n_gvec = self._get_gvector_count(norb_per_spin, valley_flag)
        if n_gvec is None:
            inferred = self._infer_gvector_count_from_dimensions(norb_per_spin, structure_info)
            if inferred is not None:
                n_gvec = inferred
                if self.config.verbose:
                    print(f"未找到 g_vec_list 文件，按维度推断 n_gvec={n_gvec}")
            else:
                v = self._normalize_valley_for_gvec(valley_flag) or valley_flag
                raise RuntimeError(
                    f"无法确定G向量数量：未找到 g_vec_list_*_{v}_1layer.npy，且维度推断失败。"
                    f"（result_dir={self.config.result_dir}）"
                )

        # 初始化composition字典结构
        for layer_info in structure_info["layers"]:
            layer_name = f"L{layer_info['layer']}"
            composition[layer_name] = {}
            for sublayer_info in layer_info["sublayers"]:
                sublayer_name = f"S{sublayer_info['sub']}"
                composition[layer_name][sublayer_name] = {
                    "atom": sublayer_info["atom"],
                    "tot": 0.0
                }
                if self.config.spin_polarized:
                    composition[layer_name][sublayer_name]["up"] = [0.0] * len(sublayer_info["orbs"])
                    composition[layer_name][sublayer_name]["dn"] = [0.0] * len(sublayer_info["orbs"])
                else:
                    composition[layer_name][sublayer_name]["weights"] = [0.0] * len(sublayer_info["orbs"])

        # 现在根据正确的TAPW波函数组织来分析轨道成分
        # 波函数组织：[Layer0的所有G向量, Layer1的所有G向量]
        # 每层每个G向量：[Sub0的orb数, Sub1的orb数, Sub2的orb数, Sub3的orb数, ...]
        
        # 计算每层每个G向量的总轨道数
        orbs_per_gvec_per_layer = {}
        layer_sublayer_orb_counts = {}
        
        for layer_info in structure_info["layers"]:
            layer_idx = layer_info['layer']  # 直接使用layer值，因为已经是0基索引
            orbs_in_this_layer_gvec = 0
            layer_sublayer_orb_counts[layer_idx] = []
            
            for sublayer_info in layer_info["sublayers"]:
                orb_count = len(sublayer_info['orbs'])
                orbs_in_this_layer_gvec += orb_count
                layer_sublayer_orb_counts[layer_idx].append(orb_count)
            
            orbs_per_gvec_per_layer[layer_idx] = orbs_in_this_layer_gvec
        
        # 计算每层的总轨道数
        orbs_per_layer = {}
        for layer_idx in orbs_per_gvec_per_layer:
            orbs_per_layer[layer_idx] = n_gvec * orbs_per_gvec_per_layer[layer_idx]
        
        if self.config.verbose and self._first_analysis:
            print(f"每层每G向量轨道数: {orbs_per_gvec_per_layer}")
            print(f"每层总轨道数: {orbs_per_layer}")
            print(f"开始轨道分析，共{len(structure_info['layers'])}层")
        
        for layer_info in structure_info["layers"]:
            layer_idx = layer_info['layer']  # 直接使用layer值，因为已经是0基索引
            layer_name = f"L{layer_info['layer']}"
            
            layer_start = 0
            try:
                # 计算该层的起始位置
                layer_start = sum(orbs_per_layer[i] for i in range(layer_idx))
                
                for sublayer_idx, sublayer_info in enumerate(layer_info["sublayers"]):
                    sublayer_name = f"S{sublayer_info['sub']}"
                    sublayer_comp = composition[layer_name][sublayer_name]
                    orb_list = sublayer_info['orbs']
                    orb_count = len(orb_list)
                    
                    # 累加该子层在所有G向量下的轨道权重
                    orb_weights_up = np.zeros(orb_count)
                    orb_weights_dn = np.zeros(orb_count) if self.config.spin_polarized else None
                    
                    for g_idx in range(n_gvec):
                        # 计算该子层在这个G向量下的起始位置
                        gvec_start = layer_start + g_idx * orbs_per_gvec_per_layer[layer_idx]
                        
                        # 计算该子层在该G向量内的偏移
                        sublayer_offset = sum(layer_sublayer_orb_counts[layer_idx][:sublayer_idx])
                        sublayer_start = gvec_start + sublayer_offset
                        sublayer_end = sublayer_start + orb_count
                        
                        # 确保索引不越界
                        if sublayer_end <= norb_per_spin:
                            # 累加这个G向量下该子层的轨道权重
                            orb_weights_up += up_weights[sublayer_start:sublayer_end]
                            if orb_weights_dn is not None:
                                assert dn_weights is not None
                                orb_weights_dn += dn_weights[sublayer_start:sublayer_end]
                        else:
                            if self.config.verbose:
                                print(f"警告: 子层 {layer_name}{sublayer_name} G向量 {g_idx} 索引越界: {sublayer_end} > {norb_per_spin}")
                    
                    # 更新结果
                    if self.config.spin_polarized:
                        sublayer_comp["up"] = orb_weights_up.round(4).tolist()
                        assert orb_weights_dn is not None
                        sublayer_comp["dn"] = orb_weights_dn.round(4).tolist()
                        sublayer_comp["tot"] = round(float(np.sum(orb_weights_up) + np.sum(orb_weights_dn)), 4)
                    else:
                        sublayer_comp["weights"] = orb_weights_up.round(4).tolist()
                        sublayer_comp["tot"] = round(float(np.sum(orb_weights_up)), 4)
                        
            except Exception as e:
                print(f"分析层{layer_idx}时出错: {e}")
                print(f"layer_start: {layer_start}, orbs_per_layer: {orbs_per_layer}")
                print(f"layer_sublayer_orb_counts: {layer_sublayer_orb_counts}")
                raise e
        
        return composition
    
    def save_results(self, result: Dict, valley_flag: str):
        """保存分析结果"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        output_file = os.path.join(self.config.output_dir, f"orbital_data_{self.band_type}_{valley_flag}.json")
        try:
            # 转换numpy类型为Python原生类型
            result_converted = convert_numpy_types(result)
            
            # 使用标准JSON编码器
            json_str = json.dumps(result_converted, indent=2)
            # 压缩数字数组到单行
            compact_json_str = compact_json_arrays(json_str)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(compact_json_str)
            if self.config.verbose:
                print(f"结果已保存: {output_file}")
        except Exception as e:
            print(f"保存结果时发生错误: {e}", file=sys.stderr)
    

    
    def run(self):
        """运行轨道分析"""
        if self.config.verbose:
            print("开始轨道成分分析...")
        
        # 检测可用的谷和文件
        valleys = self.detect_valleys_and_files()
        
        if not valleys:
            print("错误: 未找到有效的能带数据和特征向量文件")
            print("请确保计算时设置了 eig_vec_cal=True")
            return
        
        if self.config.verbose:
            print(f"检测到 {len(valleys)} 个谷: {list(valleys.keys())}")
        
        # 确定要分析的谷
        if self.config.valley == "auto":
            target_valleys = valleys.keys()
        elif self.config.valley in valleys:
            target_valleys = [self.config.valley]
        else:
            print(f"错误: 指定的谷 '{self.config.valley}' 不存在")
            print(f"可用的谷: {list(valleys.keys())}")
            return
        
        # 分析每个谷
        for valley in target_valleys:
            try:
                valley_info = valleys[valley]
                self.analyze_from_files(
                    valley_info["band_file"],
                    valley_info["vec_file"],
                    valley_info["valley_flag"]
                )
            except Exception as e:
                print(f"分析谷 {valley} 时出错: {e}")
                continue
        
        if self.config.verbose:
            print(f"\n轨道分析完成！结果保存在: {self.config.output_dir}")


def main():
    """轨道分析工具主函数"""
    parser = argparse.ArgumentParser(description='TAPW 轨道成分分析工具')
    parser.add_argument('result_dir', nargs="?", type=str, default=".", help='TAPW计算结果目录')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='TAPW配置文件路径 (config.yaml)')
    parser.add_argument('--output', '-o', type=str, default='orbital_analysis',
                       help='输出目录 (默认: orbital_analysis)')
    parser.add_argument('--valley', type=str, default='auto',
                        help='要分析的谷 (默认: auto，分析所有可用的谷)')
    parser.add_argument('--band', '--band-type', dest='band', type=str, default='CBM',
                        help='能带类型: CBM 或 VBM (默认: CBM)')
    parser.add_argument('--spin', action='store_true', default=True,
                        help='是否为自旋极化计算 (默认: True)')
    parser.add_argument('--no-spin', action='store_false', dest='spin',
                        help='非自旋极化计算')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='静默模式，减少输出信息')
    
    args = parser.parse_args()
    
    config = OrbitalAnalysisConfig(
        config_file=args.config,
        result_dir=args.result_dir,
        output_dir=args.output,
        valley=args.valley,
        band_type=(args.band or 'CBM').upper(),
        spin_polarized=args.spin,
        verbose=not args.quiet
    )
    
    try:
        analyzer = OrbitalAnalyzer(config)
        analyzer.run()
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 
