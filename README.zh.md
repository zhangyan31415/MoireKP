# MoireKP

MoireKP 从局域轨道哈密顿量构造受对称性约束的莫尔连续模型。`tapw` 负责输入准备、TAPW 能带、对称表示和拓扑计算；`kp` 负责选取低能基底、下折叠、模型构造和独立导出。

[English](README.md) · [示例](examples/README.md) · [TAPW](tapw/README.md) · [KP](kp/README.md)

## 安装

使用 Python 3.11 或更高版本的独立环境：

```bash
python -m pip install .
tapw --help
kp --help
```

开发时用 `python -m pip install -e .`。读取 SIESTA HSX 需要 `python -m pip install '.[siesta]'`。大型稀疏计算可使用 `environment.yml` 中的复数 PETSc/SLEPc 环境。

## 输入与流程

`tapw prepare-hs openmx|abacus|siesta` 读取已完成的 DFT 输出，写出以 eV 为单位的 H(R)、无量纲 S(R)、结构与轨道/自旋信息。[输入准备指南](scripts/README.zh.md)说明各后端的文件和命令。

[论文算例](examples/README.md)提供配套的 NSOC 自洽／一步 SOC 输入，以及 `POSCAR_relaxed` 和 `POSCAR_rigid`，以及重新生成 H/S 和构造六个模型的命令。大矩阵由读者在本地生成。以下命令从仓库根目录运行：

```bash
TAPW_CFG=examples/zrs2_3.89/tapw_Gamma.yaml
KP_CFG=examples/zrs2_3.89/kp_Gamma.yaml
tapw run  -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```

`kp project` 按配置选择低能空间，默认使用能量线性化 Löwdin 下折叠。`kp symm` 生成模型构造所需的对称包；`kp model` 拟合并导出模型。下折叠与后续系数拟合是两个步骤。

TAPW 输出 `band/`、`symmetry/`、`symm_rep/`、`topology/`；KP 输出 `inspect/`、`projection/`、`symmetry/`、`symm_rep/`、`model/`。导出后运行 `model/evaluate.py`，无需安装 MoireKP 即可求值。

可选入口为 `tapw topo`、`tapw symm-rep`、`kp inspect` 和 `kp symm-rep`；用 `--help` 查看参数。`tapw init -o workdir` 生成起始配置。

## 目录与发布

- `tapw/tapw/`、`kp/kp/`：正式软件源码。
- `scripts/`：后端准备工具、兼容入口和打包工具。
- `examples/`：三个论文算例的输入文件和运行说明。

[示例索引](examples/README.md)列出论文使用的六个模型。软件包提供源码、第一性原理输入和模型配置。

```bash
python scripts/release/build_distribution.py --output dist
```

打包工具核对安装包与源码，并排除本地研究文件。

## 许可证

软件、仓库编写的文档和小型配置采用 `LGPL-3.0-or-later`，条款见 [COPYRIGHT](COPYRIGHT)、[COPYING.LESSER](COPYING.LESSER) 和 [COPYING](COPYING)。软件许可不会自动覆盖外部数据或 DFT 输入；ABACUS CSR 工具附有独立许可声明。
