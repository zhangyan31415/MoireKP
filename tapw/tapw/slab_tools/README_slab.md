# Bi2Se3 Slab Band Structure Calculator

这个工具包提供了计算Bi2Se3薄膜（slab）能带结构的完整功能，支持不同厚度的薄膜结构分析。

## 功能特性

- **单层计算**：只考虑Rz = 0的跳跃项
- **多层薄膜**：支持任意厚度的薄膜结构
- **表面态分析**：识别和可视化表面态
- **厚度对比**：同时计算和对比不同厚度的能带
- **交互式分析**：Jupyter notebook界面
- **命令行工具**：批量计算和自动化分析

## 文件结构

```
├── slab.py                    # 主要的slab计算模块
├── slab_interactive.ipynb     # 交互式Jupyter notebook
├── run_slab_example.py        # 简单使用示例
├── README_slab.md            # 本说明文件
├── H.dat                     # 哈密顿矩阵数据（需要）
├── S.dat                     # 重叠矩阵数据（需要）
└── POSCAR                    # 结构文件（需要）
```

## 快速开始

### 1. 命令行使用

最简单的使用方法：

```bash
# 计算5层薄膜的能带结构
python slab.py --max_Rz 2

# 比较不同厚度的薄膜
python slab.py --compare --max_Rz 3

# 包含表面态分析
python slab.py --surface_analysis --max_Rz 2 --layer_orbs 15
```

### 2. 运行示例脚本

```bash
python run_slab_example.py
```

这会自动计算单层、3层和5层薄膜的能带结构，并生成对比图。

### 3. 交互式使用

打开Jupyter notebook：

```bash
jupyter notebook slab_interactive.ipynb
```

## 详细使用说明

### 命令行参数

```bash
python slab.py [选项]
```

**必需文件：**
- `--H`: 哈密顿矩阵文件路径（默认：H.dat）
- `--S`: 重叠矩阵文件路径（默认：S.dat）
- `--poscar`: POSCAR结构文件路径（默认：POSCAR）

**计算参数：**
- `--max_Rz`: 最大Rz值，决定薄膜厚度（默认：2）
- `--nseg`: 每个k路径段的点数（默认：40）
- `--kpath`: k路径选择，'KGM'或'GMKG'（默认：KGM）

**分析选项：**
- `--compare`: 比较不同厚度的薄膜
- `--surface_analysis`: 进行表面态分析
- `--layer_orbs`: 每层的轨道数（用于表面态分析，默认：15）

**输出选项：**
- `--outfile`: 输出图片文件名（默认：slab_bands.png）

### 薄膜厚度理解

`max_Rz`参数控制薄膜的厚度：

- `max_Rz = 0`: 单层（只包含Rz = 0的跳跃项）
- `max_Rz = 1`: 3层薄膜（包含Rz = -1, 0, +1）
- `max_Rz = 2`: 5层薄膜（包含Rz = -2, -1, 0, +1, +2）
- `max_Rz = n`: (2n+1)层薄膜

### 使用示例

#### 示例1：基本薄膜计算

```bash
python slab.py --max_Rz 2 --nseg 50 --outfile bi2se3_5layer.png
```

计算5层Bi2Se3薄膜，使用50个k点每段，输出到`bi2se3_5layer.png`。

#### 示例2：厚度对比研究

```bash
python slab.py --compare --max_Rz 4 --outfile thickness_comparison.png
```

对比单层、3层、5层、7层和9层薄膜的能带结构。

#### 示例3：表面态分析

```bash
python slab.py --surface_analysis --max_Rz 3 --layer_orbs 15 --outfile surface_states.png
```

分析7层薄膜的表面态，假设每层有15个轨道。

#### 示例4：自定义k路径

```bash
python slab.py --kpath GMKG --max_Rz 2 --nseg 30
```

使用Γ→M→K→Γ的k路径计算5层薄膜。

## 交互式界面使用

Jupyter notebook提供了更友好的交互界面：

1. **参数选择**：使用滑块和下拉菜单选择参数
2. **实时计算**：点击按钮即可开始计算
3. **结果可视化**：自动生成高质量图片
4. **高级分析**：包含带隙分析和态密度计算

### Notebook功能

- **厚度选择器**：多选框选择要计算的薄膜厚度
- **能量范围滑块**：调整可视化的能量窗口
- **带隙分析**：自动分析不同k点的带隙
- **态密度绘制**：计算和对比态密度

## 输出文件

### 图片输出

程序会生成高质量的能带结构图，包含：

- 不同厚度薄膜的能带对比
- 高对称点标记
- 表面态高亮显示（如果启用）
- 费米能级参考线

### 终端输出

计算过程中会显示：

- 数据加载信息
- R向量统计
- 计算进度
- 能带统计摘要

## 理论背景

### 薄膜构造方法

薄膜哈密顿量通过限制R向量的Rz分量构造：

```
H_slab(k) = Σ_{|Rz| ≤ max_Rz} H(R) * exp(ik·R)
```

其中只包含满足`|Rz| ≤ max_Rz`条件的R向量。

### 表面态分析

表面态特征通过分析本征态的空间局域性确定：

```
surface_weight = |ψ_top|² + |ψ_bottom|²
```

其中`ψ_top`和`ψ_bottom`分别是波函数在顶层和底层的分量。

## 故障排除

### 常见问题

1. **文件不存在错误**
   ```
   ERROR: Required file 'H.dat' not found
   ```
   确保H.dat、S.dat和POSCAR文件在当前目录中。

2. **内存不足**
   ```
   MemoryError: Unable to allocate array
   ```
   减少k点数量（`--nseg`参数）或薄膜厚度（`--max_Rz`参数）。

3. **矩阵维度不匹配**
   ```
   AssertionError: H.dat / S.dat orbital mismatch
   ```
   检查H.dat和S.dat文件的轨道数是否一致。

### 性能优化

- 使用较少的k点进行快速测试
- 对于大系统，先计算较薄的薄膜
- 在高性能计算机上运行大规模计算

## 进阶使用

### 自定义k路径

可以修改`slab.py`中的k路径函数来定义自己的高对称点路径：

```python
def custom_kpath(nseg=40):
    # 定义自己的高对称点
    points = {
        'Γ': np.array([0.0, 0.0]),
        'X': np.array([0.5, 0.0]),
        'Y': np.array([0.0, 0.5])
    }
    # 构造路径...
```

### 批量计算

创建脚本进行批量计算：

```python
import subprocess

thicknesses = [1, 2, 3, 4, 5]
for max_Rz in thicknesses:
    cmd = f"python slab.py --max_Rz {max_Rz} --outfile slab_{2*max_Rz+1}L.png"
    subprocess.run(cmd.split())
```

## 引用

如果您在研究中使用了这个工具，请引用相关的WannierTools论文。

## 支持

如有问题或建议，请联系开发者或查看WannierTools文档。 