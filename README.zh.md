# MoireKP

<p align="center">
  <b>面向扭转双层材料的 TAPW 计算与连续 <i>k·p</i> 模型构建工具</b>
</p>

<p align="center">
  <a href="README.md">English</a>
  ·
  <a href="examples/README.md">示例</a>
  ·
  <a href="#kp-工作流">KP 工作流</a>
  ·
  <a href="examples/data-manifest.yaml">数据清单</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="Package" src="https://img.shields.io/badge/package-tapw%20%7C%20kp-4c6ef5">
  <img alt="Status" src="https://img.shields.io/badge/status-CPC%20release%20candidate-orange">
  <img alt="Data" src="https://img.shields.io/badge/data-external%20datasets-lightgrey">
</p>

MoireKP 是一个 Python 科研软件包，用于从 OpenMX 派生的实空间哈密顿量出发，构造扭转双层体系的 TAPW 截断基表示，并进一步得到低能连续模型。仓库提供两个命令行入口：

| 模块 | 作用 | 典型输出 |
| --- | --- | --- |
| `tapw` | 构造 TAPW 哈密顿量，计算能带、源空间对称表示和拓扑量 | `band/`、`symmetry/`、`topology/` |
| `kp` | 检查 TAPW 谱、投影低能空间、投影对称性、拟合连续模型 | `inspect/`、`projection/`、`symmetry/`、`model/` |

当前 release 接口只保留单配置文件、短命令和 canonical 输出目录。旧的 split config、旧 CLI alias 和旧输出目录不是发布接口。

## 目录

- [适合做什么](#适合做什么)
- [安装](#安装)
- [30 秒检查](#30-秒检查)
- [TAPW 工作流](#tapw-工作流)
- [KP 工作流](#kp-工作流)
- [输出目录](#输出目录)
- [示例与外部数据](#示例与外部数据)
- [发布状态](#发布状态)

## 适合做什么

MoireKP 面向需要同时处理 TAPW 源哈密顿量和低能连续模型的工作流：

- 从 OpenMX `H/S` 矩阵和结构输入生成 TAPW band 数据。
- 导出 TAPW raw-H source symmetry 表示矩阵。
- 计算 Berry curvature、quantum geometry 和 Wilson-loop/WCC。
- 从 TAPW band/Q-set 中选择低能态并构造 projected/downfolded Hamiltonian。
- 将 TAPW source symmetry 投影到 KP 连续模型基底。
- 拟合并导出 standalone `model/evaluate.py`，用于后续能带、拓扑和论文图数据生成。

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

如果你手动重建环境，推荐使用 MKL BLAS/LAPACK；不要让 `tapw` 或 `kp` 命中 `~/.local/bin` 里的旧脚本。

## 30 秒检查

这些命令不需要外部 OpenMX 矩阵、TAPW 数组或预计算 KP 输出：

```bash
python -m pytest tests/test_release_contract.py tests/kp/test_example_dependency_contract.py -q
tapw --help
tapw init --help
kp --help
```

正式发布前建议运行完整 fast baseline：

```bash
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

## TAPW 工作流

生成模板：

```bash
tapw init -o workdir
```

编辑 `workdir/config.yaml` 后，用同一个配置文件运行不同 TAPW workflow：

```bash
tapw run  -c workdir/config.yaml
tapw symm -c workdir/config.yaml
tapw topo -c workdir/config.yaml
```

一个典型 release-style TAPW 配置如下：

```yaml
case:
  output_root: outputs

bands:
  valley: K1
  q_shell: 6
  efermi: -4.10
  kpath:
    labels: [G, M, K, G]
    points_per_segment: 40
    coordinates:
      G: [0.0, 0.0]
      M: [0.5, 0.0]
      K: [0.3333333333, 0.3333333333]

symmetry:
  valley: K1
  q_shell: 6

topology:
  valley: Gamma
  q_shell: 3
  mesh:
    n_b1: 21
    n_b2: 21
    range_b1: [-0.5, 0.5]
    range_b2: [-0.5, 0.5]
  bands:
    vbm2:
      sector: valence
      indices: [-1, -2]
  berry_curvature:
    - bands: vbm2
  quantum_geometry:
    - bands: vbm2
  wcc:
    - bands: vbm2
      loop: b2
```

配置文件不写 `enable`。命令本身决定运行 `bands`、`symmetry` 还是 `topology`。

## KP 工作流

KP 使用一个 case YAML 贯穿 inspect、project、symm 和 model：

```bash
kp inspect -c kp/configs/K1_q06.yaml
kp project -c kp/configs/K1_q06.yaml
kp symm    -c kp/configs/K1_q06.yaml
kp model   -c kp/configs/K1_q06.yaml
```

第一步必须保留。`kp inspect` 用来查看 TAPW band、Q-block 谱和候选低能态，然后再决定 `project.nlow_state_list`、拟合窗口和参考 Q。

典型 KP 配置结构：

```yaml
case:
  profile: K1
  q_shell: q06
  output_root: ../outputs

material:
  name: MoTe2
  spin: up
  efermi: -4.10
  hamk_file: ../../tapw/outputs/K1/q06/band/hamiltonian_k.npy
  qset1_file: ../../tapw/outputs/K1/q06/band/g_vectors_group1.npy
  qset2_file: ../../tapw/outputs/K1/q06/band/g_vectors_group2.npy
  band_file: ../../tapw/outputs/K1/q06/band/energies_vbm.txt

inspect:
  energy_window: [-1.0, 1.0]

project:
  nlow_state_list: [[0], [0]]
  e_ref: -4.10

symm:
  tapw_symmetry_dir: ../../tapw/outputs/K1/q06/symmetry

model:
  energy_window: [-1.0, 1.0]
```

几个约定：

- `case` 描述输出身份，例如 `profile`、`q_shell` 和 `output_root`。
- `material` 描述输入数据和物理参考，例如 TAPW 文件、spin 标记和 `efermi`。
- `inspect.energy_window` 控制图中显示的 `E - material.efermi` 范围；所有能带仍会计算和绘制。
- `project.e_ref` 是 projection/downfolding 的参考能量，和 `material.efermi` 可以相同，但语义不同。
- `kp symm` 只需要 `symm.tapw_symmetry_dir`，operation 列表来自 TAPW symmetry 输出。
- `kp model` 直接导出 standalone model，不需要 `kp export`。

KP 的模型拓扑计算不新增 `kp topo` 命令。`kp model` 导出的 `model/evaluate.py` 顶部包含 Berry curvature、quantum geometry 和 WCC 的用户参数；用户进入 `model/` 目录后修改开关和网格参数，再运行：

```bash
python evaluate.py
```

## 输出目录

TAPW canonical 输出：

```text
outputs/
  K1/
    q06/
      band/
        energies_vbm.txt
        energies_cbm.txt
        wavefunctions_vbm.npy
        wavefunctions_cbm.npy
        hamiltonian_k.npy
        g_vectors_group1.npy
        g_vectors_group2.npy
        kpoints.npy
      symmetry/
        representations.npz
        residuals.csv
        summary.md
  Gamma/
    q03/
      topology/
        grid21x21_b1_m0p5_0p5_b2_m0p5_0p5/
          chern_summary.json
          berry_curvature_vbm2.txt
          berry_curvature_vbm2.pdf
          quantum_geometry_vbm2.txt
          quantum_geometry_vbm2.pdf
          quantum_geometry_vbm2_trace_condition.txt
          wcc_vbm2_loop_b2.txt
          wcc_vbm2_loop_b2.pdf
```

KP canonical 输出：

```text
outputs/
  K1/
    q06/
      inspect/
        spectrum.txt
        scatter.pdf
        wavefunctions.npz
      projection/
        heff.npy
        eigvals.txt
        wavefunctions.npz
        basis.npz
        basis.md
        scatter.pdf
      symmetry/
        representations.npz
        residuals.csv
        summary.md
      model/
        README.md
        MODEL.md
        evaluate.py
        model_data.npz
        eigvals.npy
        band_comparison.pdf
        band_comparison_all.pdf
        q_lattice_harmonics.pdf
```

目录名携带 valley 和 q-shell。非默认 topology 范围会写进 grid id，例如：

```text
grid21x41_b1_0p0_0p5_b2_m0p5_0p5
grid21x41_b1_m0p5_0p0_b2_m0p5_0p5
```

## 示例与外部数据

`examples/` 中跟踪的是配置文件和小型模板输入。大型 OpenMX 矩阵、TAPW 数组、对称性输出和预计算 KP 结果属于外部数据，当前不随源码仓库提交。

常用入口：

```text
examples/tapw/mote2_9.43/
examples/tapw/mgi2_9.43/
examples/mote2_3.89/
examples/mgi2_3.89/
```

数据依赖和发布状态记录在：

```text
examples/data-manifest.yaml
```

外部数据齐全后，按各示例目录中的配置运行 TAPW 和 KP workflow。

## 发布状态

当前 checkout 的 license、DOI 和公开数据 URL 仍需最终确认。正式归档前必须处理：

```text
RELEASE_BLOCKERS.md
RELEASE_VALIDATION.md
examples/data-manifest.yaml
```

正式打 tag 前运行：

```bash
scripts/release_gate.sh
```

该脚本会启用最终发布检查；license、DOI、公开数据 URL、checksum 和 release blocker 未完成时应当失败。
