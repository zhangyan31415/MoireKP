# 中文稿对称适配低能规范补充设计

## 目标

根据当前 KP 投影与对称化实现，在 `paper/moirekp/zh/main.tex` 中补充对称适配低能规范的数学定义和程序数据一致性约束。修改只解释方法与实现，不加入材料算例数值，也不改写第 6 章已有的解析物理基。

## 落稿位置

1. 在第 4 章“从 TAPW-DFT 到低能连续基”中新增“对称适配的低能规范”小节。该小节说明低能 frame 的幺正规范自由度、由实际 exactified actions 构造规范的规则、Kramers 配对、简并子空间相位锚定和无可用分辨操作时的单位规范回退。
2. 在第 8 章“输出文件与后处理流程”中增加实现段落，说明同一个 frame artifact 同时约束投影哈密顿量、波函数、spin operator 和 exactified continuum matrices，并通过 frame hash 防止不同规范的产物被混用。

## 数学约定

若低能 frame 为 (U_L)，内部规范变换为 (U_L'=U_LW)。所有低能对象必须使用同一个 (W)：

\[
h'=W^\dagger hW,\qquad
O'=W^\dagger OW,\qquad
u'=W^\dagger u.
\]

幺正操作的表示变换为 (D_g'=W^\dagger D_gW)，反幺正 sewing part (D_gK) 的变换为 (D_g'=W^\dagger D_gW^*)。全文明确说明算法使用显式 action metadata 和实际矩阵谱，不从 operation family 字符串推断几何作用。

## 表述边界

- 不声称每个 valley 都存在可对角化内部基的有限阶幺正操作。
- 不按本征值绝对值排序，因为幺正表示本征值的模均为 1；排序依据离散本征相位和确定性规则。
- 无 sector-preserving resolving unitary 时保留单位 frame，并报告 `not_applicable`，不把该状态写成失败或缺少物理对称性。
- 不把 sampled low-energy sewing diagnostics 与 production exactified continuum action 混为一谈。
- 不加入未在论文中已有依据的误差、时间或材料结论。

## 验证

在 `validation_runs/` 下使用隔离输出目录执行中文稿原有 LaTeX/BibTeX 编译流程，检查未定义引用、公式语法和 PDF 生成。构建产物不写回或提交到 `paper/moirekp/zh/`。
