# quick_compare_hermitian_diag.py
import numpy as np
import scipy.linalg as sla

import numpy as np, scipy, sys
print("Python", sys.version)
print("NumPy", np.__version__, "SciPy", scipy.__version__)





try:
    from threadpoolctl import threadpool_info
    for info in threadpool_info():
        print("BLAS", info.get("internal_api"), info.get("library"), "threads=", info.get("num_threads"))
except Exception:
    print("Install threadpoolctl for BLAS info: pip install threadpoolctl")


np.random.seed(0)
n = 200

# 1) 必须这样构造“复厄米”矩阵（注意 conj().T）
X = np.random.randn(n, n) + 1j*np.random.randn(n, n)
A = (X + X.conj().T) / 2
print("hermitian_check =", np.linalg.norm(A - A.conj().T) / np.linalg.norm(A))

# 2) NumPy eigh（调用 _heevd/_syevd）
w_np, _ = np.linalg.eigh(A)

# 3) SciPy eigh：默认 driver=evr，再和 evd（与 NumPy 更一致）都测一下
w_sp_def, _ = sla.eigh(A)                 # default = evr
w_sp_evd, _ = sla.eigh(A, driver="evd")   # divide & conquer（与 NumPy常用driver一致）

# 4) 统一排序后比较
def cmp(a, b, rtol=1e-12, atol=1e-10):
    a = np.sort(a); b = np.sort(b)
    same = np.allclose(a, b, rtol=rtol, atol=atol)
    mad  = float(np.max(np.abs(a - b)))
    mrd  = float(np.max(np.abs((a - b)/(np.abs(b)+1e-300))))
    return same, mad, mrd

print("np vs scipy(default evr):", cmp(w_np, w_sp_def))
print("np vs scipy(evd)        :", cmp(w_np, w_sp_evd))
