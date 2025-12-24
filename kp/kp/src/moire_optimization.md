# `moire.py` 运行瓶颈分析与优化推导记录

本文记录对 `kp/configs/mgi2_G/src/moire.py` 的一次“正确性优先”的加速过程：先用轻量 profiling 定位瓶颈，再在**不改变物理/线性代数结果**的前提下做等价变换与实现层面的优化。

> 目标：显著缩短最后的 `k`-loop（`compute_eigenvalues`）运行时间；不引入并行；对所有加速路径都提供**可验证的回退**以保证正确性。

---

## 1. 轻量 profiling：瓶颈在哪里？

用户实测（优化前）输出大致为（按每个 k 点平均）：

- `total ≈ 1.104 s/k`
- `symm_real ≈ 0.539 s/k`
- `symm_imag ≈ 0.559 s/k`
- `schur, eig` 只占极小比例

结论：**对称化（symmetrization）占据几乎全部时间**，且 real/imag 两次对称化基本等量，意味着“同一件事做了两遍”。

---

## 2. 关键优化 A：一次对称化同时得到 `S[Y]` 与 `S[iY]`

### 2.1 数学推导（严格等价）

记 `Y(k)` 为某个 term 的基矩阵（随 `k` 变化），群元素 `g` 在矩阵空间的作用写作

- **unitary**：`g(Y)(k) = D(g) · Y(g^{-1}k) · D(g)^{-1}`
- **antiunitary**：`g(Y)(k) = D(g) · Y(g^{-1}k)^* · D(g)^{-1}`

其中 antiunitary 的 `*` 来自复共轭（等价于 `UK` 作用里的 `K`）。

对 `iY`：

- unitary 线性：`g(iY) = i·g(Y)`
- antiunitary 含复共轭：`g(iY) = D(g)·(iY)^*·D(g)^{-1} = D(g)·(-i)Y^*·D(g)^{-1} = -i·g(Y)`

因此令

- `S[Y](k) = Σ_g g(Y)(k)`
- `Y_anti(k) = Σ_{g antiunitary} g(Y)(k)`

则

```
S[iY](k) = Σ_{g unitary} ( i·g(Y)(k) ) + Σ_{g antiunitary} ( -i·g(Y)(k) )
         = i·( Σ_g g(Y)(k) - 2·Σ_{g antiunitary} g(Y)(k) )
         = i·( S[Y](k) - 2·Y_anti(k) )
```

这说明：**只要在做 `S[Y]` 的同一趟循环里额外累加 antiunitary 部分，就能得到 `S[iY]`，完全不需要再跑一遍对称化。**

### 2.2 代码落点

- 新增：`ContinuumModelBuilder.symmetrize_Y_and_iY_basis_static`（`kp/configs/mgi2_G/src/moire.py:1765`）
  - 一趟循环同时累计 `Y_symm = Σ_g g(Y)` 与 `Y_anti = Σ_{anti} g(Y)`
  - 最后用 `Y_symm_i = 1j * (Y_symm - 2*Y_anti)` 得到 `S[iY]`
  - 保持原逻辑：若非 Hermitian（数值误差/基构造导致），则做 `Y <- Y + Y†` 的 Hermitian 化（注意这里**不除以 2**，保持与原实现一致）

- 更新调用点（避免重复对称化）：
  - `ContinuumModel.assemble_hamiltonian`（`kp/configs/mgi2_G/src/moire.py:917`）
  - `ContinuumModelBuilder.stack_Y_for_term`（`kp/configs/mgi2_G/src/moire.py:1958`）
  - `compute_eigenvalues` 最后 k-loop（`kp/configs/mgi2_G/src/moire.py:3208` 附近）

---

## 3. 关键优化 B：利用对称算符的 monomial 结构加速 `D·Y·D^{-1}`

### 3.1 结构观察

对称生成器返回的表示矩阵 `D(g)` 在本模型基底上通常是 **monomial matrix**：

- 每行/每列恰好一个非零元（= 置换矩阵 × 对角相位/符号）

令该矩阵满足：

- `D_{i, perm[i]} = v_i`

则对 unitary：

```
(D Y D†)_{i,j} = v_i · Y_{perm[i], perm[j]} · v_j^*
```

对 antiunitary：

```
(D Y* D†)_{i,j} = v_i · (Y_{perm[i], perm[j]})^* · v_j^*
```

因此不需要稠密矩阵乘法：只需要

1) `Y` 在两个轴上按 `perm` 置换；  
2) 行/列按 `v` 与 `v^*`（或等价的 `v^{-1}`）缩放；  
3) antiunitary 情况再做一次复共轭。

### 3.2 代码落点与正确性保障

- monomial 提取：`_extract_monomial_matrix`（`kp/configs/mgi2_G/src/moire.py:1469`）
- 快速应用：`_apply_symmetry_op_to_matrix`（`kp/configs/mgi2_G/src/moire.py:1536`）
- **一次性验证**（确保结果严格一致）：
  - `_validate_monomial_op_once` 在每个 `(symmetry_gen, op_name, param)` 上只做一次：
    - 生成随机复矩阵 `Y0`
    - 比较 “稠密乘法结果” vs “monomial 快速路径结果”
    - 若不一致则自动失效该 op 的 fast-path 并回退到稠密算法

该设计保证：即使未来某些对称算符不再是 monomial，也不会产生错误结果，只会变慢。

---

## 4. 关键优化 C：将多步对称操作序列合成为一步（减少分配与调用）

对称化时每个 orbit 元素对应一个操作序列 `op_seq`。在 fast-path 可用时，可以把多步 monomial 操作合成为单个 monomial 操作：

- `_get_composed_symmetry_action`（`kp/configs/mgi2_G/src/moire.py:1579`）返回 `(perm, vals, inv_vals, is_anti_total)`
- `_validate_composed_symmetry_action_once`（`kp/configs/mgi2_G/src/moire.py:1626`）用“逐步应用”作为真值做一次性比对

在 `symmetrize_Y_basis_static` 和 `symmetrize_Y_and_iY_basis_static` 中优先走 composed fast-path；失败则逐步回退到 `_apply_symmetry_op_to_matrix`。

---

## 5. 其它实现级优化（不改物理结果）

- **避免无效 cache 开销**：当 `use_cache=False` 时不再构造 cache key，也不写入 cache（之前会浪费时间与内存）。
- **减少中间数组**：对称化累加使用就地加法；`S[iY]` 通过 `Y_symm`/`Y_anti` 的线性组合得到，避免第二次对称化循环。
- **轻量 profiling**：在 `compute_eigenvalues` 中加入按 k 点平均的计时统计（loop/symm/schur/eig）。
- **`Y_basis` 稀疏评估**（等价）：在 `make_Y_basis_function` 中把 `(row,col)` 的匹配关系（`Q - p = Q'`）提前算好，并暴露 `Y_basis.eval_sparse(k)` 返回 `(rows, cols, vals)`；对称化时在 monomial/composed action 可用的情况下只对这些 `nnz` 元素做置换与缩放，再累加回稠密输出矩阵。
- **`(k-Q)_z` 幂缓存**（等价）：对每个 orbit 点 `k` 与每层 `Q_rows`，缓存 `kz^m (m=0..max(Mz,Mz_star))`，term 只做查表与 `kz^Mz · (kz^Mz_star)^*`，避免大量 `complex ** int`。
- **orbit + action 缓存**：新增 `_get_symmetry_orbit_actions_cached`，对同一个 `(k, sym_ops, symmetry_gen)` 只生成一次 orbit，并预存 composed monomial action（`perm/inv_perm/vals/inv_vals/is_anti`），多 term 直接复用。
- **k-loop 侧的工程优化**：
  - 预先生成 `ACTIVE_TERMS`（过滤 `|r|<1e-33` 的项），避免每个 k 扫描全表；
  - `n_jobs=1` 时用普通 for-loop 替代 `joblib.Parallel` 减少调度开销；
  - `remove=0` 时跳过 Schur complement 的空块运算；
  - 用 `scipy.linalg.eigh(check_finite=False)` 替代 `np.linalg.eigh`（在本环境下能显著加速）。
- **（已被进一步替代）对称化输出缓冲复用**：`symmetrize_Y_and_iY_basis_static` 支持 `out_real/out_imag/out_anti`（`use_cache=False` 时），用于减少中间矩阵分配；在后续“直接组装 H(k)”方案中已不再需要构造 `Y_symm` 稠密矩阵。

---

## 6. 关键优化 D：直接组装 `H(k)`（避免构造稠密 `Y_symm`）

### 6.1 关键观察

虽然 `Y_symm(k)`/`Y_symm_i(k)` 是 `76×76` 的稠密数组，但它们的非零元数量实际只有
`O(|orbit| · nnz(Y_basis))`（在当前配置中通常是几百个量级）。因此在 k-loop 中若每个 term 都走

```
Y_symm, Y_symm_i = symmetrize(...)
H += r_real * Y_symm + r_imag * Y_symm_i
```

则 `r*Y` 与 `H+=...` 的稠密逐元素操作会产生不必要的 `O(dim^2)` 内存带宽开销。

### 6.2 数学等价推导（严格）

对每个 orbit 元素 `g`，记其对矩阵的作用为

- unitary：`g(Y) = D(g) · Y(kk) · D(g)^{-1}`
- antiunitary：`g(Y) = D(g) · Y(kk)^* · D(g)^{-1}`

其中 `kk = g^{-1}k`，这正是 `symmetrize_Y_and_iY_basis_static` 的定义。

令 `s(g)=+1`（unitary），`s(g)=-1`（antiunitary）。则同一趟 orbit 循环中，单个 `g` 对哈密顿量的贡献权重可以写成

```
w_g = r_real + i · s(g) · r_imag
```

因此可以在 **不构造 `Y_symm`/`Y_symm_i` 稠密矩阵** 的情况下，直接对每个 `g` 的稀疏元做置换+相位缩放后累加到 `H_out`。

### 6.3 关键难点：必须完全复刻“条件 Hermitian 化”的语义

原实现（`symmetrize_Y_and_iY_basis_static`）对 `Y_symm` 与 `Y_symm_i` 分别做：

```
if not allclose(Y, Y†):  Y <- Y + Y†   # 注意不除以 2（保持旧语义）
```

这是一个 **非线性**（带条件分支）的操作，不能简单把 “Hermitian 化” 推迟到最终 `H(k)` 上，否则会改变结果。

直接组装需要做的是：在每个 `g` 的稀疏累加时，按是否触发 Hermitian 化决定是否再额外累加一次“共轭转置贡献”：

- 若 `Y_symm` 需要 Hermitian 化：对每个 `g(Y)` 额外加上 `r_real · g(Y)†`
- 若 `Y_symm_i` 需要 Hermitian 化：对每个 `g(Y)` 额外加上 `(-i·s(g)·r_imag) · g(Y)†`

合并后等价于在稀疏层面对 `(i,j)` 与 `(j,i)` 成对累加。

### 6.4 代码落点 + 对拍

- 新增/更新：`ContinuumModelBuilder.add_symmetrized_term_to_matrix_static`（`kp/configs/mgi2_G/src/moire.py:2048`）
  - 直接把某个 term 的贡献累加进 `H_out`（稀疏 nnz 级别操作），避免构造 `Y_symm` 稠密矩阵与 `H += r*Y` 的 `O(dim^2)` 写带宽。
  - 为保证严格等价：
    - 在 `symmetrize_Y_and_iY_basis_static` 内部记录每个 term 是否触发过 `Y<-Y+Y†`（属性：`_moire_needs_hermitize_real/_imag`），并检测不同 k 下是否不一致（不一致则对该 term 回退到稠密路径）。
- 对拍开关：`MOIRE_COMPARE_ASSEMBLY=1`
  - 比较 “基准稠密组装” vs “直接组装” 的 `H(k)`（默认测 `kpath` 的 `[0, mid, last]`）
  - 示例输出：`max|ΔH| ~ 1e-16`（远小于 `1e-10` 目标）

---

## 7. 效果（示例）

在用户给出的同一组参数下，优化后观测到（示例）：

- 阶段 1（优化 A/B/C 完成后）：
  - `avg per k ≈ 0.174 s/k`
  - `symm ≈ 0.143 s/k`
  - `eig ≈ 0.025 s/k`

- 阶段 2（稀疏 `Y_basis` + orbit/action 缓存 + SciPy `eigh` 等进一步优化后，示例一次运行）：
  - `avg per k ≈ 0.042 s/k`
  - `symm ≈ 0.040 s/k`
  - `eig ≈ 0.001 s/k`（已不再是瓶颈）

- 阶段 3（加入 `kz^m` 幂缓存后，示例一次运行）：
  - `avg per k ≈ 0.033 s/k`
  - `symm ≈ 0.030 s/k`
  - `eig ≈ 0.001 s/k`

- 阶段 4（直接组装 `H(k)`，避免构造 `Y_symm` 稠密矩阵；示例一次运行）：
  - `avg per k ≈ 0.013 s/k`
  - `symm/assemble ≈ 0.011 s/k`
  - `eig ≈ 0.002 s/k`

总耗时约从 `~1.10 s/k` 降至 `~0.013 s/k`，即 **~80×+ 级别加速**，且未引入并行。

---

## 8. 可控开关

- `ContinuumModelBuilder._SYMM_VALIDATE_MONOMIAL = True/False`
  - `True`：每个对称算符/序列只验证一次（推荐在开发/修改对称实现时打开）
  - `False`：完全跳过验证（极致性能）
- `PROFILE_LIGHT = True/False`
  - 控制最后 `k`-loop 计时输出

---

## 9. 仍可继续的优化方向（仍然保持严格等价）

当前对称化仍占主导，进一步压榨的方向主要是“减少重复算幂/减少 Python 调度开销”：

1) **按 k 点预计算 `(k-Q)` 的幂**：对每个 `k` 与每层 `Q_set` 先算 `kz=(k-Q)_z`，并缓存 `kz^m` 与 `kz*^m`（`m=0..max_order`），这样每个 term 的 `Mz/Mz_star` 只做查表与点乘，避免大量 `complex ** int`。
2) **按 k 点预分组 orbit_actions**：当前 `_get_symmetry_orbit_actions_cached` 已经缓存，但每次仍要做 dict lookup；可以在 `kpath` 上一次性预生成 `orbit_actions[k_index][sym_ops_key]`，在 k-loop 内直接索引。
3) **只要谱不需要本征矢**：把 `eigh` 改成 `eigvals_only=True`（或 `np.linalg.eigvalsh`）能进一步减少后处理成本。

以上方向都属于“实现层面等价变换”，不改变物理公式与最终矩阵。

---

## 10. 推荐的下一步修改计划（分阶段，先写计划不动代码）

> 原则：每一步都必须配套“等价性验证”，默认容许浮点误差 `~1e-10`；不引入并行（`n_jobs=1`）。

### 阶段 0：基线与回归（必须先做）

目标：确保后续任何改动都能一键发现“物理/线代语义变化”。

- 固化 3 类验证入口（当前已有，建议保持随时可跑）：
  - 对称算符审计：`MOIRE_AUDIT_SYMM=1 python kp/configs/mgi2_G/src/moire.py`
  - 局部对拍自测：`MOIRE_SELFTEST=1 python kp/configs/mgi2_G/src/moire.py`
  - 组装对拍：`MOIRE_COMPARE_ASSEMBLY=1 python kp/configs/mgi2_G/src/moire.py`
- 建议新增（未来要做时再加）：`MOIRE_COMPARE_EIG=1` 随机抽若干 k 点，比较 `eigvals` 最大差异（以及 `||H1-H2||_F`），把阈值写死为 `1e-10`。

预期收益：0×（只为防回归）；风险：极低。  
验证方法：上述 3 个入口均 PASS。

### 阶段 1（低风险，保持 dense 输出/算法不变）：减少 Python 调度与内存写带宽

目标：在不改变对称化数学定义的前提下，把 “term×orbit×nnz” 的 Python 循环与 scatter-add 做到更接近带宽上限。

- **按 `sym_ops_key` 分组预取 orbit/action**：把 `ContinuumModelBuilder._get_symmetry_orbit_actions_cached(...)` 的 dict lookup 从“每 term”降到“每组 sym_ops 一次”；并把每个 orbit 元素的 `(perm, inv_perm, vals, inv_vals, is_anti)` 直接存成紧凑数组结构。
- **批量化 index 变换与相位缩放**：对同一个 `(term, orbit_action)`，把
  - `rr = inv_perm[rows]`, `cc = inv_perm[cols]`
  - `vv = vals0 * vals[rr] * inv_vals[cc]`（含 antiunitary 的共轭）
  用更少的中间数组/更少的 Python 分支实现（例如预分配工作缓冲）。
- **把多次 `np.add.at` 合并为少数大批次 scatter**：把多个 orbit 元素拼接成一次大的 `(rr_all, cc_all, vv_all)` 再做 `np.add.at`（或 `np.bincount` 到扁平索引 `rr*dim+cc`），减少 `add.at` 调用次数。
- **更激进的写路径选择**：对 `Y_basis._moire_sparse_unique=True` 的情形，继续走 `H_out[rr,cc]+=...`；对重复索引才退回 `np.add.at`（当前已做，但未来批量化时要保持该语义）。

预期收益：通常 `~1.5×–3×`（取决于 term 数量与 orbit 大小）；风险：低（主要是索引/共轭/符号细节）。  
验证方法：必须跑 `MOIRE_COMPARE_ASSEMBLY=1`（H 对拍）+ 新增的 `MOIRE_COMPARE_EIG=1`（谱对拍）。

### 阶段 2（低风险，但会改 API 形状一点点）：把“组装计划”从 term 侧提升到全局

目标：把每个 term 的 `(rows, cols)` 结构与对称作用后的映射进一步“静态化”，减少每 k 的重复工作。

- **为每个 term 预生成 `assembly_plan`**：结构中包含
  - `base_rows/base_cols/base_row_q_idx`
  - 每个 `orbit_action` 下的 `rr/cc`（以及是否需要 antiunitary 共轭、Hermitian 补项标志）
  - 相位缩放中“只依赖 rr/cc 的部分”（例如 `vals[rr]`, `inv_vals[cc]`）可预存索引，k 时只做 gather+乘法
- **k 侧只生成数值向量 `vals0(k)`**：利用已有的 `_KZ_POW_CACHE`，让每个 orbit 点的 `vals0(k)` 生成尽可能接近纯 numpy 向量运算。

预期收益：`~2×–5×`（取决于计划复用程度）；风险：中（更复杂的数据结构，容易出现错位）。  
验证方法：`MOIRE_COMPARE_ASSEMBLY=1` + `MOIRE_COMPARE_EIG=1` 必须覆盖更多随机 k（例如 20 个）。

### 阶段 3（结构性方案，目标 10×+）：全局 COO/CSR 或 JIT 编译内层循环

目标：彻底摆脱 Python 层循环开销，让 “term×orbit×nnz” 的主循环在编译态/低层完成。

方案 A（全局 COO/CSR）：
- **构造全局扁平索引与权重模板**：把所有 term×orbit 的 `(rr,cc)` 以及相位缩放所需的索引/共轭标志组织成一个或几个大表。
- **每个 k 只生成所有 term 需要的 `vals0(k)` 并批量乘到权重上**，最后一次性 scatter 到 `H`（dense 或 sparse）。
- 若后续愿意尝试：直接在 sparse `H` 上做 `eigsh`（这会改变求解器与数值路径，需要更严格的误差评估）。

方案 B（Numba/JIT，不并行但编译）：
- 把 `add_symmetrized_term_to_matrix_static` 的最内层（对 `rr/cc/vv` 的循环与 Hermitian 补项）迁移到 JIT 函数，输入尽量是 numpy 原生数组（int/complex），输出写入 `H_out`。

预期收益：`~5×–20×`（与 term/orbit/nnz 规模强相关）；风险：中-高（实现复杂度提升、调试成本高）。  
验证方法：扩大对拍覆盖：随机 k（≥50）比较 `H`、`eigvals`，并保留一份“旧实现回退开关”方便 bisect。

### 阶段 4（如果你只关心能带，不要本征矢）：让 `eig` 不再拖后腿

目标：当 `symm/assemble` 进一步变快后，`eig` 会成为主要瓶颈。

- 仅输出本征值：把 `scipy.linalg.eigh` 改成 `eigvals_only=True`（或直接 `eigvalsh`），避免构造本征矢矩阵。
- 若只需要少数能带：考虑 `eigsh`/shift-invert（这会引入稀疏/迭代求解的路径变化，需要额外严格验证）。

预期收益：对 `eig` 部分 `~1.5×–10×`（取决于需求与求解器）；风险：低-中（主要是数值路径变化）。  
验证方法：对拍 `eigvals`，并在关键 k 点（Γ/K/M）检查误差与简并结构。
