# H/S 输入准备

统一入口是 `tapw prepare-hs BACKEND`，读取已完成的 DFT 输出，写出 H(R)、S(R)、结构、轨道/自旋信息和配置片段。H 的单位为 eV，S 无量纲。

```bash
python -m pip install .
tapw prepare-hs --help
```

| 后端 | 输入 | 准备方式 |
|---|---|---|
| [OpenMX](openmx/README.md) | `.scfout` 和对应结构 | 用 `openmx/build_openmx_symm_hs.sh` 编译 C++ 读取器 |
| [ABACUS](abacus/README.md) | 实空间 LCAO H/S CSR 输出 | 随包 Python 转换器 |
| [SIESTA](siesta/README.md) | HSX | 安装可选依赖 `.[siesta]` |

```bash
tapw prepare-hs openmx --input work/openmx.scfout --structure work/openmx.dat \
  --binary build/openmx_import_hs/analysis_symm_hs --output prepared/openmx --format both
tapw prepare-hs abacus --input work/abacus --output prepared/abacus --format both
tapw prepare-hs siesta --input work/system.HSX --output prepared/siesta --format both
```

输出目录须为新目录。补充层、扭转、谷、费米能和 k 路径后再运行 TAPW；配置与 H/S 放在同一目录时可保留相对路径。

`--symmetrize` 另外写出对称平均矩阵；已有导入结果可用 `common/symmetrize_hs.py` 处理。非磁性自旋输入使用 `--assume-nonmagnetic`；磁性输入需提供逐原子磁矩。参数和基底约定见各后端指南。

实际算法集中在 `tapw.io`。各后端 `prepare_hs.py` 和顶层 OpenMX 脚本是兼容入口；旧实现保存在 `legacy/`。`release/` 放发布工具，`benchmarks/` 放导出性能检查。论文分析与试算脚本收拢到 `devtools/paper_analysis/`，不进入软件发布包。
