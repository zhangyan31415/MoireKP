# MoireKP

<p align="center">
  <b>面向扭转层状材料的 TAPW 计算与连续 <i>k·p</i> 模型构建工具</b>
</p>

<p align="center">
  <a href="README.md">English</a>
  ·
  <a href="examples/README.md">示例</a>
  ·
  <a href="#kp-工作流">KP 工作流</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="Package" src="https://img.shields.io/badge/package-tapw%20%7C%20kp-4c6ef5">
</p>

MoireKP 是一个用于莫尔电子结构工作流的 Python 软件包。它以实空间哈密顿量、
可选的重叠矩阵和标准结构文件为输入，构造截断原子平面波（TAPW）源模型，并在
对称性约束和诊断下拟合低能连续模型。

| 模块 | 作用 | 主要输出 |
| --- | --- | --- |
| `tapw` | 构造 TAPW 哈密顿量，计算能带、源空间对称表示、能带表示和拓扑量 | `band/`、`symmetry/`、`symm_rep/`、`topology/` |
| `kp` | 检查 TAPW 谱，投影低能空间和对称作用，分析能带表示并导出独立连续模型 | `inspect/`、`projection/`、`symmetry/`、`symm_rep/`、`model/` |

## 安装

在仓库根目录创建环境：

```bash
conda env create -f environment.yml
conda activate moirekp
```

`environment.yml` 会以 editable 模式安装当前源码。安装后确认命令来自当前环境：

```bash
which python
which tapw
which kp
tapw --help
kp --help
```

## 快速检查

以下命令不需要示例使用的外部数据：

```bash
python -c "import tapw, kp"
tapw --help
tapw init --help
kp --help
```

数值示例需要源码仓库未包含的外部矩阵和数组。

## TAPW 工作流

生成初始工作目录：

```bash
tapw init -o workdir
```

编辑 `workdir/config.yaml`，然后使用同一个配置运行所需工作流：

```bash
tapw run      -c workdir/config.yaml
tapw symm     -c workdir/config.yaml
tapw symm-rep -c workdir/config.yaml  # 可选：能带表示
tapw topo     -c workdir/config.yaml  # 可选：拓扑计算
```

TAPW 配置示例如下：

```yaml
system:
  output: outputs
  structure: POSCAR
  hamiltonian: H.npz
  overlap: S.npz
  orbitals: {Mo: s3p2d1, Te: s3p2d2}
  twist_index: 8
  layers: [1, 1]
  spin: true

bands:
  valley: K1
  q_shell: 6
  efermi: -4.10
  save_hamiltonian: true
  kpath:
    labels: [G, M, K, G]
    points_per_segment: 20
    coordinates:
      G: [0.0, 0.0]
      M: [0.5, 0.0]
      K: [0.3333333333, 0.3333333333]

symmetry:
  valley: K1
  q_shell: 6
  efermi: -4.10
  representation:
    points:
      G: [0.0, 0.0]
      M: [0.5, 0.0]
      K: [0.3333333333, 0.3333333333]

topology:
  valley: K1
  q_shell: 6
  mesh:
    n_b1: 21
    n_b2: 21
    range_b1: [-0.5, 0.5]
    range_b2: [-0.5, 0.5]
  bands:
    vbm2: {sector: valence, indices: [-1, -2]}
  berry_curvature: [{bands: vbm2}]
  quantum_geometry: [{bands: vbm2}]
  wcc: [{bands: vbm2, loop: b2}]
```

`system.structure` 可以是 POSCAR、CIF 或其他 ASE 支持的格式。程序不再需要
OpenMX 输入文件：结构和轨道信息分别来自 `system.structure` 和
`system.orbitals`。正交基底可以省略 `system.overlap`。面内 Bravais 类型从结构
自动判断。目前支持六角和正方晶格，其他面内度量会直接报错。`layers` 始终显式
给出，普通双层也写成 `[1, 1]`。

配置中不使用工作流 `enable` 开关；命令本身决定执行哪个配置段。

## KP 工作流

同一个 KP YAML 用于低能投影、对称性精确化、能带表示分析和连续模型拟合。例如：

```bash
CFG=examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinless_q06.yaml
kp inspect  -c "$CFG"  # 可选：源谱诊断
kp project  -c "$CFG"
kp symm     -c "$CFG"
kp model    -c "$CFG"
kp symm-rep -c "$CFG"  # 可选：能带表示
```

`kp project` 选择满足要求的最小低能子空间并写出投影 Heff。`kp symm` 将 TAPW
raw-H 作用投影到该基底，并生成 `kp model` 使用的对称性数据。`kp model` 拟合并
导出独立连续模型。可选的 `kp symm-rep` 使用投影阶段实际保存的 k 点判断 little
group，并同时报告原始表示块和 polar-unitary 诊断。

当前 KP 配置具有以下结构：

```yaml
system:
  name: MoTe2
  output: ../outputs/k1_spinless
  tapw_output: ../../tapw/outputs
  layers: [1, 1]
  spin: up
  orbital_order: Te-s3p2d2,Mo-s3p2d1,Te-s3p2d2
  cell:
    - [52.4951, 0.0, 0.0]
    - [-26.2476, 45.4621, 0.0]
    - [0.0, 0.0, 27.0]

project:
  valley: K1
  q_shell: 6
  efermi: -4.10
  target: valence
  selection: auto
  e_ref: -4.60

symmetry: {}

bands:
  kpath:
    labels: [G, M, K, G]
    points_per_segment: 20
    coordinates:
      G: [0.0, 0.0]
      M: [0.5, 0.0]
      K: [0.3333333333, 0.3333333333]
  compare_to_heff: true
  band_slice: [46, 54]

model:
  target_bands: top
  max_order: {kinetic: 10, intralayer: 4, interlayer: 4}
  fit:
    method: linear
    kpoints: [0, 2]
    bands: 8
    one_sided_weight: 0.0
    two_sided_weight: 0.0
```

约定：

- 省略 `project.nlow_state_list` 即启用自动低能选择；显式列表用于复用先前选定的
  子空间并跳过搜索。
- `symmetry: {}` 表示从 TAPW 对称性包推断操作，并不表示可以省略 `kp symm`。
- `bands` 是供投影和模型比较共同使用的顶层配置段。
- 省略 `model.harmonics` 时，`kp model` 自动选择 harmonic 数；显式取值会绕过
  自动选择。
- `model.fit` 控制拟合 k 点、能带数、方法和可选低能权重；权重取值由具体案例决定，
  并记录在相应配置中。
- `kp model` 直接写出独立求值器，不再需要单独的 `kp export`。

运行 `kp model` 后，进入生成的 `model/` 目录，修改 `evaluate.py` 顶部的用户参数，
然后运行：

```bash
python evaluate.py
```

## 输出目录

TAPW 写入 `band/`、`symmetry/`、`symm_rep/` 和 `topology/`；KP 写入
`inspect/`、`projection/`、`symmetry/`、`symm_rep/` 和 `model/`。完整文件列表见
[tapw/README.md](tapw/README.md) 和 [kp/README.md](kp/README.md)。

## 示例

仓库包含五个材料目录和十个 KP 配置：

| 目录 | 已跟踪的 KP 案例 |
| --- | --- |
| `examples/mote2_3.89` | K1 spinful 和 spinless |
| `examples/mgi2_3.89` | Gamma spinful；M1 spinful 和 spinless |
| `examples/mote2_aab_5.09` | Gamma spinful；K1-A 和 K1-B 单自旋模型 |
| `examples/zrs2_3.15` | Gamma spinful |
| `examples/ptse2_7.34` | Gamma spinful 配置；完整工作流尚未测试 |

仓库跟踪配置、结构文件和小型模板。大型 OpenMX 矩阵、TAPW 数组、对称性输出和
生成的 KP 结果属于外部数据，不提交到源码仓库。

## 许可证

MoireKP 软件以及仓库作者编写的文档、配置文件和小型示例采用
`LGPL-3.0-or-later` 许可证。版权声明和完整许可条款见
[COPYRIGHT](COPYRIGHT)、[COPYING.LESSER](COPYING.LESSER) 和
[COPYING](COPYING)。

该软件许可证不会自动覆盖外部 OpenMX/TAPW/KP 输入、大型示例数据集或生成物。
