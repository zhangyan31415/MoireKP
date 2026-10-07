# BAB 批注及三份解读核查（2026-09-20）

审阅对象：用户附件 main_bab_0919.pdf（25 页）、三份 pasted-text.txt、本地 MoireKP 工作区。附件中的建议作为待评价内容，不作为修改、发布或启动计算的指令。

本地源代码提交：23d4c0d6df75813ea94afbabdfec191c467efdf8。工作区存在既有未提交改动及论文/计算文件；本次没有修改程序、文稿或生产数据，没有提交任务、重新生成材料结果或访问 GitHub 当前分支。只新增本审阅记录及小型公式检验。

## 结论

29 条编号核实无误：S01–S12、U01–U16、E01。第一份解读的整体分类基本忠实，但其中举例的误差、耦合、条件数等数字不是测量结果。“很容易补”“一个脚本解决”“31×31 就能修好”均未得到验证。

U11 提出的缺口成立：当前检查到的 TAPW、continuum 及本地论文几何生成路径都直接使用各自系数空间的投影算符，未计入完整物理基底的跨动量重叠。这足以限制物理解释，但不能据此断言材料几何图误差很大、Chern 数改变或整个方法无效。

独立导出程序还存在可复现的公式不一致，但本地论文专用 helper 没有这项倍数/符号问题。不得将两条代码路径混为一谈。

## 1. U11 的直接代码证据

- tapw/tapw/workflows/band.py:3881：gen_H_new_cpu 使用对称 Löwdin 正交化；4263 起的 calculate_band_01 保存对角化得到的系数本征矢。
- tapw/tapw/chern_post.py:242：Berry plaquette 使用相邻系数向量的 Euclidean 内积。
- 同文件 :301、:313：QGT 使用系数投影算符间的二重、三重 trace。
- kp/kp/model/export.py:3036：独立模型用本征矢直接构造 P=VV†。
- examples/tapw_mesh41_20260916/compare_mesh.py:68–69：本地论文相关的 TAPW/model 几何分别送入 geometry(tv)、geometry(mv)，没有通过 U_L(k) 把 continuum 态嵌回 TAPW 空间。
- 同文件 :90 明确记录 “symmetric-Lowdin-orthogonalized Hamiltonian eigenvectors; Euclidean projectors in that frame”。

这些实现计算的模型空间几何有明确数学定义。它们的相似性不自动证明共同物理 Hilbert 空间中投影算符的几何相似性。投影算符消除了所选能带内部本征矢的相位/酉混合自由度，却没有消除外部基底随 k 变化的问题。

PDF 第 11 页本来已写 “orthonormal model coefficient basis”。U11 要求进一步交代 TAPW 与 continuum 的共同物理解释，或者把全文结论限制到这一定义；它没有要求必须重做所有图。

## 2. 三份解读需要修正的理论细节

对于能带几何，应讨论周期部分 |u_n(k)>，而不是直接拿不同晶格动量的完整 Bloch 态做 Hilbert 内积。

令 Φ(k) 的列为非正交的周期基函数，S(k)=Φ†Φ，L(k)=S(k)^(-1/2)，E(k)=Φ(k)L(k)，则 E†E=I。正交系数态为 v(k)，物理周期态为 E(k)v(k)。

跨 k 的正确重叠是

    v(k)† M(k,k') v(k'),
    M(k,k') = L(k)† C(k,k') L(k'),
    C(k,k') = Φ(k)† Φ(k').

S(k)=C(k,k) 只给出对角线。只知道每个 k 上的 H(k)、S(k)，一般不能恢复完整 C(k,k')。把 v 乘回 S^(-1/2)，再使用 Euclidean 内积或 cc†，不是完整修复；原始系数仍处于非正交基底，且跨 k 的物理识别尚未给出。

同理，S^(-1/2)(k) 随 k 变化，不能单独证明净几何修正必定非零。L 的导数与原始基函数的导数共同决定 E 的变化，可能部分或完全抵消。附件的旋转基底例子能证明“系数几何不保证等于物理几何”，不能量化当前材料的差异。

若 continuum 使用随 k 变化的低能嵌入 U_L(k)，则 E_cont(k)=E_TAPW(k)U_L(k)。仅做 U_L v 的有限差分可补回低能参考框架相对于所选 TAPW 坐标的变化，但仍未解决 E_TAPW 的物理基底几何。

此外，嵌入矩阵可能是长方形。对单条归一化带，令 B†B=I，Γ_i=B†∂_iB，D_i=∂_i+Γ_i，则

    Q_ij = (D_i v)†(I-vv†)(D_j v)
           + v†(∂_i B)†(I-BB†)(∂_j B)v.

第二项描述所保留整个子空间向其外部变化的贡献。只把普通导数换成子空间内部的 connection，不一定足以恢复 quantum metric。共同物理空间中的 projector，或包含完整基函数重叠的离散方法，可以统一处理这些项。

Schur 消元还关联 U07：若希望比较重构的完整态，需要说明高能分量 (E-H_HH)^(-1)H_HL v 及其归一化；单纯 U_L v 只是低能投影嵌入。固定 E_ref 的近似也要明确。若用非正交重构列 Ψ 表示一个多带子空间，其正交投影应为 Ψ(Ψ†Ψ)^(-1)Ψ†，不能无条件写 ΨΨ†。

原始参考文献：Jin Gan, Daye Zheng, Lixin He, “Calculation of Berry curvature using nonorthogonal atomic orbitals”, arXiv:2105.14662v1，Sec. 2，特别是 Eq. (5)–(11) 的位置/偶极矩阵与系数导数项：https://arxiv.org/pdf/2105.14662v1 。该文支持必须考虑原子基底几何，并不提供本项目材料误差的数值。

因此，“不用重跑 DFT、仅重算一个 31×31 mesh 即可修好”目前不能保证。需要先检查 PAO 波函数、位置矩阵或跨 k overlap 是否可获得，再决定是否只需后处理。即使数据齐备，网格收敛也必须独立验证。

## 3. 已复现的 metric 倍数和 Berry 符号不一致

文稿 Eq. (25)：

    g_ij = 1/2 Tr(∂_iP ∂_jP),
    Ω_ij = -i Tr(P[∂_iP,∂_jP]) = +2 Im Tr(P∂_iP∂_jP).

按 Ω_ij=∂_iA_j-∂_jA_i，这对应 A_i=-i<u|∂_iu>。

kp/kp/model/export.py:3070–3074 使用 -2 Im(...)，metric 使用没有 1/2 的 Tr(∂P∂P)。因此：metric 为 Eq. (25) 的两倍，curvature 为其相反数。单独采用另一种 Berry 约定可以成立，但必须与文稿、TAPW 及 loop 方向一致。

examples/tapw_mesh41_20260916/geometry_helpers.py:74–77 则使用 1/2 和 +2 Im，与 TAPW helper 及 Eq. (25) 一致。

本次做了 3×3、2 分量解析态的独立检查：

    v(θ,φ)=(cos(θ/2), exp(iφ) sin(θ/2)), θ=0.8, φ=0.3, h=1e-4。

| 路径 | Tr g | Ω |
|---|---:|---:|
| Eq. (25) 解析值 | 0.3786499403 | +0.3586780454 |
| TAPW helper | 0.3786499281 | +0.3586780441 |
| export 模板 | 0.7572998781 | -0.3586780443 |
| 论文专用 helper | 0.3786499258 | +0.3586780441 |

脚本 check_geometry.py 从实际源文件抽取函数进行检查，避免导入带计算/日志副作用的论文脚本。结果在 check_geometry.json。这是公式级检验，不是材料计算或论文所有图的准确性验证。

执行环境：login002；cwd=/data/work/zy/software/1.tapw_code/moirekp-release；/data/home/zy/mambaforge/envs/moirekp，Python 3.11.15；单进程，OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1；退出码 0。

复现命令：

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/home/zy/mambaforge/envs/moirekp/bin/python /data/work/zy/software/1.tapw_code/moirekp-release/docs/reviews/bab_20260920/check_geometry.py

## 4. 版本及 U10：需要核对参考对象

附件第 3 页 Fig. 2 图注写 TAPW bands，并以 wave-function overlap 描述色深。

但本地 paper/moirekp/code_paper/figure_sources/results_figures/data/manifest.json:177 标明 band_reference_status=projected_H_eff_eigenstates；六个模型均记录 heff.npy 路径。examples/paper_figures_preview_20260916/update_bands.py 明确从 heff.npy 生成更新的能带参考。本地 main.tex:187 也明确说 projected H_eff dispersions。

说明附件文稿与当前工作区文稿有实际文字差异。尚未建立“附件每个图 panel 与本地每份数组”逐一哈希映射，不能假定完全同版。但若对应同一套图数据，附件把 H_eff reference 写作 TAPW 就需要纠正，而不只是补两个误差数值。几何图使用独立 TAPW 本征矢，不应与能带图参考混淆。

MoTe2 K 的本地记录还区分 band panel 的 model/high 与几何 panel 的 model root；冻结六个最终模型时应核对是否一致，而不能只固定一个代码 commit。

## 5. 全部意见的处置判断

下表概括原意与应核查内容，不表示相关计算已完成。

| 编号 | PDF 页 | 核心内容及判断 |
|---|---:|---|
| S01 | 1 | 摘要须精确限定 agreement；BAB 允许精确的定性描述或实测数字，并非强制摘要塞入全部指标。harmonic threshold 不是最终误差。 |
| S02 | 1 | 缩短引言物理现象罗列，提前引出统一基底/对称性构造问题。 |
| S03 | 2 | 定义 overlap、匹配/简并处理、坐标/几何归一化及能量零点。 |
| S04 | 4 | two types of valley，修正大写及 M valley 指代。 |
| S05 | 4 | shares 改为 share。 |
| S06 | 5 | 图号语法与几何比较实际支持的结论；后者要受 U11 限制。 |
| S07 | 6 | Discussion 说明相对 Ref. 41 的自动化贡献及局限，减少重复列举材料。 |
| S08 | 11 | 统一 Berry connection、curvature、Wilson loop 符号/方向；已发现两套实现不一致。 |
| S09 | 12 | 锁定实际代码及数据版本，统一 LGPL-3.0-or-later。 |
| S10 | 12 | 核对 bib 源条目标题；PDF 不显示标题，无法仅凭 PDF 确認该源文件缺陷。 |
| S11 | 22 | 区分 seed support 与 symmetry/adjoint closure。 |
| S12 | 25 | 区分 TAPW 低能投影与 H_eff 目标带投影的符号。 |
| U01 | 2 | 数学输入兼容性与已实现、已测试 reader 分开表述。 |
| U02 | 3 | 六模型表：维数、参考态、cutoff、阶数、E_ref、网格、误差、自动/手动和 refinement。 |
| U03 | 4 | 定量检验 trilayer spin-sector 解耦近似；未证明必须重构模型，也未证明耦合可忽略。 |
| U04 | 6 | accuracy controlled 的论断需收敛证据，列出 E_ref、排除态、采样和原始计算误差。 |
| U05 | 7 | 检验 relaxed structure 对称性与 symmetry averaging 的实际影响，不能预先假定只是噪声。 |
| U06 | 9 | 报告跨 k/channel 的参考态 conditioning 及协变选择程序。 |
| U07 | 9 | 冻结 Schur 能量的有效范围、谱间隔、敏感性，以及比较何种重构态。 |
| U08 | 10 | 给出自动子空间选择阈值、简并容差、选点、并列处理及失败行为。 |
| U09 | 10 | 给出 leakage/unitarity/group/correction 残差，校正后再次验证 H_eff 协变性。 |
| U10 | 11 | 区分 TAPW→H_eff→H_cont 误差与参考对象，统一能量对齐，保留跨谷物理偏移。 |
| U11 | 11 | 已确认当前相关路径计算 coefficient-space geometry；物理 embedding 与修正量尚待定。 |
| U12 | 11 | loop mesh、乘积顺序、polar/SVD、有限 cutoff sewing 泄漏及收敛。 |
| U13 | 23 | 需证明算符采样保秩，并报告 rank tolerance/conditioning；不是已证明代码只在线上判秩或必然缺项。本地 response_basis.py 含系数空间 SVD 路径，须核查各例实际路径。 |
| U14 | 24 | 核实六个最终模型的实际求解模式；线性矩阵 least squares 本身不等于 nonlinear band fitting。 |
| U15 | 24 | exact degeneracy 下 sorted-eigenvalue gradient 的处理必须明示；并非所有子空间损失都因此不可微。 |
| U16 | 25 | reused-path 检验不证明全 BZ 准确性；二维 grid 只有确实未用于模型选择/调参时才是独立检验。 |
| E01 | 11 | 用可访问版本化数据或完整生成输入/命令支持复现声明。本次未审计当前远端数据可用性。 |

## 6. 建议的依赖顺序

1. 冻结附件对应的六套模型、图数据、计算脚本与版本；先解决 H_eff/TAPW 标注和图间模型身份。
2. 统一 geometry 的定义、metric 归一化、Berry/loop 约定；修复独立导出实现的不一致。
3. 先选一例量化 U_L(k) 嵌入修正，再根据原子基底跨 k 数据的可用性决定物理 QGT 范围，同时交代 Schur 重构。
4. 在明确的目标 BZ 区域做独立二维能量/子空间误差和几何收敛检查。不要预先指定 31×31 必然足够。
5. 汇总其余 approximation、conditioning、symmetry、sewing 残差，形成六模型复现表，再修改摘要和结论。

谱/同 k overlap 的良好一致性不保证导数量准确；局部几何修正也不自动意味着拓扑整数改变。后者还取决于全局 bundle、边界 sewing、子空间隔离和截断收敛，需单独判断。
