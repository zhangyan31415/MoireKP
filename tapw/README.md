# TAPW (Truncated Atomic Plane Wave)

A Python package for calculating and plotting band structures of twisted materials.

## Current Status

If you are continuing the recent `tapw_mkl` non-TAPW / SLEPc / angle-suite work, start here:

- [`CURRENT_STATUS.md`](./CURRENT_STATUS.md)
- [`examples/manual_ab_notapw_mote2_3_9.43/angle_suite/HANDOFF_20260307.md`](./examples/manual_ab_notapw_mote2_3_9.43/angle_suite/HANDOFF_20260307.md)

The most relevant live result files are under:

- `examples/manual_ab_notapw_mote2_3_9.43/angle_suite/jobs/`
- `examples/manual_ab_notapw_mote2_3_9.43/angle_suite/plots/`

## Features

- Band structure calculation for twisted bilayer systems
- Support for various valleys (K1, K2, Gamma, M points)
- C3 symmetry consideration
- CPU/HPC parallel computation capabilities
- Orbital analysis and fatband plotting tools

## 安装

下面给一套当前 `tapw_mkl` 可用的、尽量简洁的安装流程。

这套流程的目标是：

- 安装一套独立的 `mambaforge`
- 用 `mamba` 创建 `tapw-mkl` 环境
- 只安装当前代码实际需要的核心依赖
- 最后对当前仓库做 editable 安装

说明：

- 下面流程默认包含 `slepc` 相关依赖
- 不安装 `primme`
- 不安装 `sparse-dot-mkl`
- 不安装 `pytest`
- 不安装 `Cython`

### 1. 克隆仓库

```bash
git clone <repository-url>
cd moirekp
```

### 2. 安装 `mambaforge`

推荐直接安装到 `~/mambaforge`：

```bash
curl -L \
  https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh \
  -o /tmp/Miniforge3-Linux-x86_64.sh

bash /tmp/Miniforge3-Linux-x86_64.sh -b -p "$HOME/mambaforge"
```

初始化 shell：

```bash
export PATH="$HOME/mambaforge/bin:$PATH"
source "$HOME/mambaforge/etc/profile.d/conda.sh"
```

建议顺手写入镜像配置，这样后续所有 `mamba create/install` 都会默认走镜像：

```bash
cat > ~/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
channel_priority: flexible

default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

conda clean -i -y
```

### 3. 创建 `tapw-mkl` 环境

先创建基础环境：

```bash
mamba create -y -n tapw-mkl \
  python=3.11 \
  numpy scipy matplotlib pandas pyyaml scikit-learn \
  joblib tqdm psutil sympy ase spglib pip
```

### 4. 切到 MKL

```bash
mamba install -y -n tapw-mkl -c defaults \
  "blas=*=mkl" mkl mkl-service
```

### 5. 安装 MPI / PETSc / SLEPc

```bash
mamba install -y -n tapw-mkl -c conda-forge \
  "petsc=*=*complex*" "slepc=*=*complex*" petsc4py slepc4py mpi4py
```

### 6. 激活环境

```bash
source "$HOME/mambaforge/etc/profile.d/conda.sh"
conda activate tapw-mkl
```

### 7. 安装当前仓库（editable）

```bash
python -m pip install -e .
```

### 8. 验证安装

```bash
which python
python -V

python - <<'PY'
import tapw
import numpy, scipy
import mpi4py, petsc4py, slepc4py
import ase, spglib
print("tapw:", tapw.__file__)
print("环境检查通过")
PY

tapw-config -h
tapw-calc -h
```

### 9. 运行前建议

如果你的 shell 会自动加载 oneAPI / Intel MPI，建议在运行前确保优先使用当前 conda 环境里的库：

```bash
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
```

## Usage

1. Prepare your configuration file (config.yaml):
```bash
tapw-config -o output_dir
```
and then you can edit the `config.yaml` and `bands.yaml` files in the output_dir, especially the `config.yaml` file, you must set the `H.dat` ,`S.dat` and `openmx.dat` files path.

2. Run the calculation:
```bash
cd output_dir
tapw-calc --config config.yaml
```

3. Plot the band structure:
```bash
cd output_dir/Q_shell_{n_g}/band
tapw-plot --config ../../bands.yaml
```
If you are using legacy outputs, the data may be under `band_data/` instead of `band/`.

Or with command line arguments:

```bash
tapw-plot --kpath-in KPATH.in --kpath-out KPATH.out --bands band1.dat band2.dat --labels "Band 1" "Band 2" --fermi 0.0
```
4. Calculate the Chern number:
```bash
cd output_dir
tapw-calc --config config.yaml --mode chern --n_g 4 --num_processes 100 --num_chern 20
#or you can directly edit the config.yaml file
#and then run the following command
tapw-calc --config config.yaml
cd output_dir/Q_shell_{n_g}
tapw-chernpost --config config.yaml -b -1,-2 -v 1 > tapw_chern.log
```

5. Orbital analysis and fatband plotting:
```bash
cd output_dir/Q_shell_{n_g}
tapw-orbital . --config ../config.yaml --valley Gamma --band CBM
tapw-plot-orbital . --valley Gamma --band CBM --orbital-dir orbital_analysis --output-dir orbital_plots
```

## Configuration

See example configuration files in the `examples` directory. 
