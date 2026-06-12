#!/bin/bash
# run_single_bench.sh
# 单进程基准测试（用于调试和小规模测试）

# 设置多线程，让单进程充分利用资源
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

echo "Running single-process benchmark with OpenMP/MKL threads..."

# 运行单进程基准测试
python bench_eigensolvers.py \
    --sizes 1000,2000,5000 \
    --k 20 --sigma 0.5 \
    --kinds lap1d,lap2d \
    --densities 0.001 \
    --generalized \
    --use-ilu-opinv \
    --dense-limit 3000

echo "Single-process benchmark completed. Results saved to bench_results.csv" 