#!/bin/bash
# run_parallel_bench.sh
# 利用 192 逻辑核 + 2TB 内存的并行基准测试脚本

# 设置单线程，避免 MKL/OpenMP 与 MPI 抢核
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 检测可用核心数并设置 MPI 进程数
NCORES=$(nproc)
TOTAL_CORES=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
echo "nproc reports: $NCORES cores, lscpu shows: $TOTAL_CORES total CPUs"

# 如果 nproc 报告异常少的核心，使用合理的默认值
if [ $NCORES -lt 8 ]; then
    NPROCS=32  # 使用合理的默认值
    echo "Using default $NPROCS processes (nproc seems restricted)"
else
    NPROCS=$((NCORES > 64 ? 64 : NCORES))
    echo "Using $NPROCS MPI processes based on available cores"
fi

# 运行并行基准测试
mpirun -np $NPROCS \
  --bind-to core \
  python -m mpi4py bench_eigensolvers.py \
    --sizes 2000,5000,12000,20000 \
    --k 50 --sigma 0.5 \
    --kinds lap1d,lap2d,rand \
    --densities 0.001,0.005 \
    --generalized \
    --use-ilu-opinv \
    --dense-limit 8000 \
    -st_type sinvert -st_ksp_type preonly -st_pc_type lu \
    -st_pc_factor_mat_solver_type mumps \
    -eps_ncv 200 -eps_mpd 64 \
    -eps_monitor

echo "Benchmark completed. Results saved to bench_results.csv" 