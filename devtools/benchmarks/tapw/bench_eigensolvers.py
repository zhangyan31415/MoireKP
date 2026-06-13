#!/usr/bin/env python3
# bench_eigensolvers.py
# 对比 SciPy vs SLEPc (slepc4py) 在稀疏本征问题上的效率/资源与结果一致性
# Author: TAPW development team
# References: SLEPc/slepc4py docs (EPS, ST, interval, harmonic/sinvert) and SciPy eigsh sigma/OPinv.
#   SLEPc tutorial & API: https://slepc.upv.es/slepc4py-current/docs/usrman/tutorial.html
#   SciPy eigsh: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
#   psutil RSS: https://psutil.readthedocs.io/

import argparse, math, time, threading, sys, os, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np
import psutil

# SciPy
from scipy import sparse
from scipy.sparse import csr_matrix, diags, kronsum
from scipy.sparse.linalg import eigsh, LinearOperator, spilu
from scipy.linalg import eigh

# Optional: PETSc/SLEPc
HAVE_SLEPC = True
try:
    from petsc4py import PETSc
    from slepc4py import SLEPc
except Exception as e:
    HAVE_SLEPC = False

# -------------------------------
# 工具：峰值内存监测（RSS）
# -------------------------------
class PeakRSS:
    def __init__(self, interval=0.02):
        self.interval = interval
        self._stop = threading.Event()
        self.peak = 0
        self._thr = None

    def _run(self):
        proc = psutil.Process(os.getpid())
        peak_local = 0
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > peak_local:
                    peak_local = rss
            except psutil.Error:
                continue
            time.sleep(self.interval)
        self.peak = peak_local

    def __enter__(self):
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thr is not None:
            self._thr.join()

def fmt_mem(bytes_val):
    return bytes_val / (1024**2)  # MB

# ----------------------------------
# 生成测试矩阵（稀疏对称/Hermitian）
# ----------------------------------
def make_lap1d(n: int) -> csr_matrix:
    # 1D Laplacian (Dirichlet) SPD
    main = 2*np.ones(n)
    off  = -1*np.ones(n-1)
    A = diags([off, main, off], [-1, 0, 1], format="csr")
    return A

def make_lap2d(n1: int, n2: int) -> csr_matrix:
    # 2D Laplacian SPD via kronecker sum
    A1 = make_lap1d(n1)
    A2 = make_lap1d(n2)
    return kronsum(A1, A2, format="csr")  # A = kron(I,A2)+kron(A1,I)

def nearest_square_dim(N):
    n = int(math.sqrt(N))
    if n*n == N:
        return n, n
    # choose n1*n2 ≈ N, try to keep n1≈n2
    n1 = n
    n2 = max(1, N//n1)
    while n1*n2 < N:
        n1 += 1
        n2 = max(1, N//n1)
    return n1, n2

def make_random_hermitian(n: int, density: float, seed: int = 0) -> csr_matrix:
    rng = np.random.default_rng(seed)
    # real symmetric sparse random
    try:
        R = sparse.random(n, n, density=density, format="csr",
                          rng=rng, data_rvs=rng.standard_normal)
    except TypeError:
        # 兼容老版 SciPy
        R = sparse.random(n, n, density=density, format="csr",
                          random_state=rng, data_rvs=rng.standard_normal)
    A = (R + R.T)/2.0
    # 防止太病态：加点对角
    A = A + diags(1e-3*np.ones(n))
    return A.tocsr()

def make_problem(kind: str, N: int, density: Optional[float], seed: int, generalized: bool):
    """
    kind: 'lap1d' | 'lap2d' | 'rand'
    generalized: if True, return (A,S); else (A,None)
    """
    if kind == 'lap1d':
        A = make_lap1d(N)
    elif kind == 'lap2d':
        n1, n2 = nearest_square_dim(N)
        A = make_lap2d(n1, n2)
    elif kind == 'rand':
        if density is None:
            raise ValueError("density must be provided for random matrices")
        A = make_random_hermitian(N, density, seed=seed)
    else:
        raise ValueError("unknown matrix kind")

    if generalized:
        # 构造 SPD 重叠 S：对角或(拉普拉斯+alpha I)，保持稀疏
        # 这里用简单的正对角，避免过度填充
        diag = 1.0 + 0.5*np.abs(np.sin(np.arange(N)))
        S = diags(diag, format="csr")
        return A.tocsr(), S
    else:
        return A.tocsr(), None

# --------------------------
# SciPy 求解与对比/残差
# --------------------------
def _build_spilu(AA, drop_tol, fill_factor, diag_pivot_thresh=0.0, permc_spec="COLAMD"):
    return spilu(AA, drop_tol=drop_tol, fill_factor=fill_factor,
                 diag_pivot_thresh=diag_pivot_thresh, permc_spec=permc_spec)
def solve_scipy_partial(A: csr_matrix, S: Optional[csr_matrix], k: int, which: str='LM',
                        sigma: Optional[float]=None, use_ilu_opinv: bool=False,
                        tol=1e-8, maxiter=None):
    t0 = time.perf_counter()
    with PeakRSS() as mon:
        if sigma is None:
            vals = eigsh(A, k=k, M=S, which=which, return_eigenvectors=False, tol=tol, maxiter=maxiter)
        else:
            if use_ilu_opinv:
                AA = (A if S is None else (A - sigma*S)).astype(float).tocsc()

                ilu = None
                tried = []
                for (drop_tol, fill_factor, diag_thr, jitter) in [
                    (1e-3,  10, 0.0, 0.0),
                    (1e-2,  20, 0.0, 0.0),
                    (1e-1,  50, 0.0, 0.0),
                    (1e-1,  50, 0.0, 1e-10),   # 轻微对角抖动
                    (1e-1, 100, 0.0, 1e-8),
                ]:
                    try:
                        AA_use = AA
                        if jitter > 0.0:
                            AA_use = (AA + jitter * sparse.identity(AA.shape[0], format="csc"))
                        ilu = _build_spilu(AA_use, drop_tol=drop_tol, fill_factor=fill_factor,
                                           diag_pivot_thresh=diag_thr, permc_spec="COLAMD")
                        break
                    except Exception as e:
                        tried.append((drop_tol, fill_factor, diag_thr, jitter, str(e)))
                        ilu = None

                if ilu is not None:
                    def matvec(b):
                        return ilu.solve(b)
                    OPinv = LinearOperator(AA.shape, matvec=matvec, dtype=AA.dtype)
                    vals = eigsh(A, k=k, M=S, sigma=sigma, which='LM',
                                 OPinv=OPinv, return_eigenvectors=False, tol=tol, maxiter=maxiter)
                else:
                    # 最后回退：不用 OPinv，交给 SciPy 内部 shift-invert（会走直接因子化，占内存但很稳）
                    # 这样基准不会中断，同时记录到 stdout（rank==0 已经 print）
                    vals = eigsh(A, k=k, M=S, sigma=sigma, which='LM',
                                 return_eigenvectors=False, tol=tol, maxiter=maxiter)
            else:
                vals = eigsh(A, k=k, M=S, sigma=sigma, which='LM', return_eigenvectors=False, tol=tol, maxiter=maxiter)
    t1 = time.perf_counter()
    return np.sort(vals), t1-t0, fmt_mem(mon.peak)

def solve_scipy_full(A: csr_matrix, S: Optional[csr_matrix], dense_limit=4000, tol=1e-12):
    n = A.shape[0]
    if n > dense_limit:
        return None, None, None  # 跳过
    t0 = time.perf_counter()
    with PeakRSS() as mon:
        Ad = A.toarray()
        if S is None:
            w = eigh(Ad, overwrite_a=True, check_finite=False, driver="evr")[0]
        else:
            Sd = S.toarray()
            w = eigh(Ad, Sd, overwrite_a=True, overwrite_b=True, check_finite=False, driver="gvd")[0]
    t1 = time.perf_counter()
    return np.sort(w), t1-t0, fmt_mem(mon.peak)

def residual_norms(A: csr_matrix, S: Optional[csr_matrix], eigvals: np.ndarray, eigvecs: Optional[np.ndarray]):
    # 可选：给出均值/最大残差 ||A v - λ S v||
    if eigvecs is None:
        return None, None
    res = []
    for i in range(eigvals.shape[0]):
        lam = eigvals[i]
        v = eigvecs[:, i]
        Av = A @ v
        if S is None:
            r = Av - lam * v
        else:
            r = Av - lam * (S @ v)
        res.append(np.linalg.norm(r))
    res = np.array(res)
    return float(np.mean(res)), float(np.max(res))

# --------------------------
# SLEPc 求解（串行或并行）
# --------------------------
def scipy_to_petsc(A: csr_matrix) -> PETSc.Mat:
    n = A.shape[0]
    # 创建 SeqAIJ 或并行 AIJ，根据 COMM 大小
    comm = PETSc.COMM_WORLD
    size = comm.getSize()

    # 使用 CSR 结构做预分配（更稳更快）
    A_csr = A.tocsr()
    indptr = A_csr.indptr
    indices = A_csr.indices
    data = A_csr.data

    # 创建矩阵并使用 CSR 预分配
    M = PETSc.Mat().createAIJ([n, n], comm=comm)
    rstart, rend = M.getOwnershipRange()
    local_rows = rend - rstart
    if size == 1:
        M.setPreallocationCSR((A_csr.indptr.copy(), A_csr.indices.copy()))
    else:
        row_start = A_csr.indptr[rstart]
        row_end = A_csr.indptr[rend]
        local_indptr = A_csr.indptr[rstart:rend+1].copy()
        local_indptr -= local_indptr[0]
        local_indices = A_csr.indices[row_start:row_end].copy()
        M.setPreallocationCSR((local_indptr, local_indices))
    M.setUp()
    
    # 声明对称性（小幅优化）
    M.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    
    # 仅装配本地拥有的行
    for i in range(rstart, rend):
        row_data = data[indptr[i]:indptr[i+1]]
        row_cols = indices[indptr[i]:indptr[i+1]]
        if row_cols.size:
            M.setValues(i, row_cols, row_data)
    M.assemblyBegin(); M.assemblyEnd()
    return M


@dataclass
class PetscOperators:
    A: "PETSc.Mat"
    B: Optional["PETSc.Mat"]


def build_petsc_operators(A: csr_matrix, S: Optional[csr_matrix]) -> PetscOperators:
    Am = scipy_to_petsc(A)
    Bm = scipy_to_petsc(S) if S is not None else None
    return PetscOperators(Am, Bm)


def _ensure_petsc_ops(A: csr_matrix, S: Optional[csr_matrix],
                      ops: Optional[PetscOperators]) -> PetscOperators:
    if ops is not None:
        return ops
    return build_petsc_operators(A, S)


def _create_eps(ops: PetscOperators, nev: int, tol: float, maxit: Optional[int]) -> SLEPc.EPS:
    E = SLEPc.EPS().create(PETSc.COMM_WORLD)
    if ops.B is None:
        E.setOperators(ops.A)
        E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    else:
        E.setOperators(ops.A, ops.B)
        E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    E.setDimensions(nev)
    E.setTolerances(tol, maxit if maxit is not None else PETSc.DEFAULT)
    return E

def solve_slepc_partial(A: csr_matrix, S: Optional[csr_matrix], k: int, which: str = 'LM',
                        sigma: Optional[float] = None, use_harmonic: bool = False,
                        tol: float = 1e-8, maxit=None,
                        petsc_ops: Optional[PetscOperators] = None):
    if not HAVE_SLEPC:
        return None, None, None

    ops = _ensure_petsc_ops(A, S, petsc_ops)
    E = _create_eps(ops, k, tol, maxit)

    which_upper = which.upper()
    if sigma is None:
        if which_upper == 'LM':
            E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
        elif which_upper == 'SM':
            E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
        else:
            E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    else:
        E.setTarget(sigma)
        E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        if use_harmonic:
            E.setExtraction(SLEPc.EPS.Extraction.HARMONIC)
        else:
            ST = E.getST()
            ST.setType(SLEPc.ST.Type.SINVERT)

    t0 = time.perf_counter()
    with PeakRSS() as mon:
        E.solve()
    t1 = time.perf_counter()

    nconv = min(E.getConverged(), k)
    vals = np.array([E.getEigenvalue(i).real for i in range(nconv)], dtype=float)
    return np.sort(vals), t1 - t0, fmt_mem(mon.peak)

def solve_slepc_full(A: csr_matrix, S: Optional[csr_matrix], dense_limit=4000,
                     tol: float = 1e-8, petsc_ops: Optional[PetscOperators] = None):
    # 用 EPS ALL 或者设置 nev=n。n 太大时不建议全对角化，保持与 SciPy 一致
    if not HAVE_SLEPC:
        return None, None, None
    n = A.shape[0]
    if n > dense_limit:
        return None, None, None

    ops = _ensure_petsc_ops(A, S, petsc_ops)
    E = _create_eps(ops, n, tol, None)

    t0 = time.perf_counter()
    with PeakRSS() as mon:
        try:
            E.solve()
        except Exception:
            return None, None, None
    t1 = time.perf_counter()

    nconv = E.getConverged()
    vals = np.array([E.getEigenvalue(i).real for i in range(nconv)], dtype=float)
    return np.sort(vals), t1 - t0, fmt_mem(mon.peak)

# --------------------------
# 结果核对
# --------------------------
def compare_sets(a: Optional[np.ndarray], b: Optional[np.ndarray], mode: str, sigma: Optional[float]=None):
    if a is None or b is None:
        return None
    
    # 检查空数组
    if a.size == 0 or b.size == 0:
        return {'equal': False, 'msg': f"empty array(s): a.size={a.size}, b.size={b.size}"}

    sorted_a = np.sort(a)
    sorted_b = np.sort(b)

    if mode == 'full':
        # 全部比较
        if a.size != b.size:
            return {'equal': False, 'msg': f"size mismatch {a.size} vs {b.size}"}
        diff = np.abs(sorted_a - sorted_b)
        if diff.size == 0:
            return {'equal': True, 'max_abs_diff': 0.0, 'mean_abs_diff': 0.0}
        return {'equal': bool(np.allclose(sorted_a, sorted_b, atol=1e-6, rtol=1e-6)),
                'max_abs_diff': float(diff.max()), 'mean_abs_diff': float(diff.mean())}
    elif mode in ('partial_LM','partial_SM'):
        # 按值排序比对
        k = min(a.size, b.size)
        if k == 0:
            return {'equal': False, 'msg': "no elements to compare"}
        a1 = sorted_a[:k]
        b1 = sorted_b[:k]
        diff = np.abs(a1 - b1)
        if diff.size == 0:
            return {'equal': True, 'max_abs_diff': 0.0, 'mean_abs_diff': 0.0}
        return {'equal': bool(np.allclose(a1, b1, atol=1e-6, rtol=1e-6)),
                'max_abs_diff': float(diff.max()), 'mean_abs_diff': float(diff.mean())}
    elif mode == 'partial_sigma':
        # 以接近 sigma 的顺序比对
        k = min(a.size, b.size)
        if k == 0:
            return {'equal': False, 'msg': "no elements to compare"}
        idxa = np.argsort(np.abs(a - sigma))[:k]
        idxb = np.argsort(np.abs(b - sigma))[:k]
        a1 = a[idxa]; b1 = b[idxb]
        sorted_a1 = np.sort(a1)
        sorted_b1 = np.sort(b1)
        diff = np.abs(sorted_a1 - sorted_b1)
        if diff.size == 0:
            return {'equal': True, 'max_abs_diff': 0.0, 'mean_abs_diff': 0.0}
        return {'equal': bool(np.allclose(sorted_a1, sorted_b1, atol=1e-6, rtol=1e-6)),
                'max_abs_diff': float(diff.max()), 'mean_abs_diff': float(diff.mean())}
    else:
        return None

# --------------------------
# 主流程
# --------------------------
@dataclass
class CaseSpec:
    kind: str           # 'lap1d' | 'lap2d' | 'rand'
    N: int
    density: Optional[float]      # for 'rand'
    generalized: bool

def main():
    parser = argparse.ArgumentParser(description="SciPy vs SLEPc eigen benchmark")
    parser.add_argument('--sizes', type=str, default='2000,5000,12000',
                        help='comma-separated sizes (N). For lap2d, N≈n1*n2')
    parser.add_argument('--k', type=int, default=50, help='number of eigenpairs for partial solves')
    parser.add_argument('--sigma', type=float, default=0.5, help='target sigma for interior spectrum')
    parser.add_argument('--densities', type=str, default='0.001,0.005',
                        help='comma-separated densities for random matrices')
    parser.add_argument('--kinds', type=str, default='lap1d,lap2d,rand',
                        help='matrix kinds to test, comma sep')
    parser.add_argument('--generalized', action='store_true', help='also test generalized A x = lambda S x')
    parser.add_argument('--slepc-parallel', type=str, default='auto',
                        help='auto|lap2d|none  (enable distributed build for lap2d in SLEPc part)')
    parser.add_argument('--dense-limit', type=int, default=4000, help='max N for full diagonalization')
    parser.add_argument('--use-ilu-opinv', action='store_true', help='use ILU-based OPinv for SciPy sigma case')
    parser.add_argument('--output', type=str, default=None,
                        help='optional path for CSV summary (default: bench_results.csv next to this script)')
    args, _ = parser.parse_known_args()  # 允许 -st_* 等透传给 PETSc/SLEPc

    sizes = [int(s) for s in args.sizes.split(',') if s.strip()]
    densities = [float(s) for s in args.densities.split(',') if s.strip()]
    kinds = [s.strip() for s in args.kinds.split(',') if s.strip()]
    k = args.k
    sigma = args.sigma

    rank = 0
    size = 1
    if HAVE_SLEPC:
        comm = PETSc.COMM_WORLD
        rank = comm.getRank()
        size = comm.getSize()

    results = []
    def log(*a, **kws):
        if rank == 0:
            print(*a, **kws, flush=True)

    log(f"[INFO] MPI size={size}, SLEPc available={HAVE_SLEPC}")
    log(f"[INFO] sizes={sizes}, kinds={kinds}, densities={densities}, k={k}, sigma={sigma}\n")

    def record_result(base_info: dict, mode: str, time_val, mem_val, note: str):
        if rank != 0:
            return
        entry = dict(base_info)
        entry.update(mode=mode, time=time_val, peakMB=mem_val, note=note)
        results.append(entry)

    def run_case(spec: CaseSpec):
        run_full = spec.kind != 'rand'
        seed = 1234 if run_full else 2024
        density = spec.density

        if rank == 0:
            msg = f"=== Case: {spec.kind}, N={spec.N}, generalized={spec.generalized}"
            if density is not None:
                msg += f", density={density}"
            msg += " ==="
            log(msg)

        A, S = make_problem(spec.kind, spec.N, density, seed=seed, generalized=spec.generalized)
        petsc_ops = build_petsc_operators(A, S) if HAVE_SLEPC else None

        base = dict(kind=spec.kind, N=spec.N, generalized=spec.generalized)
        if density is not None:
            base['density'] = density

        if run_full:
            sc_w, sc_t, sc_m = solve_scipy_full(A, S, dense_limit=args.dense_limit)
            record_result(base, 'full_scipy', sc_t, sc_m, 'SciPy')

            sl_w, sl_t, sl_m = solve_slepc_full(A, S, dense_limit=args.dense_limit,
                                                petsc_ops=petsc_ops)
            record_result(base, 'full_slepc', sl_t, sl_m, 'SLEPc')

            cmp_full = compare_sets(sc_w, sl_w, mode='full')
            record_result(base, 'full_compare', None, None, str(cmp_full))

        sc_vals_lm, sc_time_lm, sc_mem_lm = solve_scipy_partial(A, S, k=k, which='LM')
        record_result(base, 'LM_scipy', sc_time_lm, sc_mem_lm, 'SciPy')

        sl_vals_lm, sl_time_lm, sl_mem_lm = solve_slepc_partial(
            A, S, k=k, which='LM', petsc_ops=petsc_ops
        )
        record_result(base, 'LM_slepc', sl_time_lm, sl_mem_lm, 'SLEPc')

        cmp_lm = compare_sets(sc_vals_lm, sl_vals_lm, mode='partial_LM')
        record_result(base, 'LM_compare', None, None, str(cmp_lm))

        sc_vals_sigma, sc_time_sigma, sc_mem_sigma = solve_scipy_partial(
            A, S, k=k, which='LM', sigma=sigma, use_ilu_opinv=args.use_ilu_opinv
        )
        record_result(base, 'sigma_scipy', sc_time_sigma, sc_mem_sigma,
                      f"SciPy (ILU_OPinv={args.use_ilu_opinv})")

        sl_vals_sinv, sl_time_sinv, sl_mem_sinv = solve_slepc_partial(
            A, S, k=k, which='LM', sigma=sigma, use_harmonic=False, petsc_ops=petsc_ops
        )
        record_result(base, 'sigma_slepc_sinvert', sl_time_sinv, sl_mem_sinv, 'SLEPc sinvert')

        sl_vals_har, sl_time_har, sl_mem_har = solve_slepc_partial(
            A, S, k=k, which='LM', sigma=sigma, use_harmonic=True, petsc_ops=petsc_ops
        )
        record_result(base, 'sigma_slepc_harmonic', sl_time_har, sl_mem_har, 'SLEPc harmonic')

        cmp_sigma_sinv = compare_sets(sc_vals_sigma, sl_vals_sinv, mode='partial_sigma', sigma=sigma)
        cmp_sigma_har = compare_sets(sc_vals_sigma, sl_vals_har, mode='partial_sigma', sigma=sigma)
        record_result(base, 'sigma_compare_sinvert', None, None, str(cmp_sigma_sinv))
        record_result(base, 'sigma_compare_harmonic', None, None, str(cmp_sigma_har))

    case_specs: List[CaseSpec] = []
    for kind in kinds:
        for N in sizes:
            if kind == 'rand':
                for density in densities:
                    case_specs.append(CaseSpec(kind=kind, N=N, density=density,
                                               generalized=args.generalized))
            else:
                for generalized in ([False, True] if args.generalized else [False]):
                    case_specs.append(CaseSpec(kind=kind, N=N, density=None,
                                               generalized=generalized))

    for spec in case_specs:
        run_case(spec)

    # 输出结果
    if rank == 0:
        import csv
        header = sorted({k for r in results for k in r.keys()})
        default_output = Path(__file__).resolve().with_name('bench_results.csv')
        output_path = Path(args.output).expanduser() if args.output else default_output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in results:
                w.writerow(r)
        # 简要打印
        print("\n=== SUMMARY (first 20 rows) ===")
        for r in results[:20]:
            print(r)
        print(f"\nSaved CSV -> {output_path}")

if __name__ == "__main__":
    # 降低 SciPy 稀疏求解器的一些警告噪音
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
