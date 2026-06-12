# MoireKP Examples

This directory contains release-facing examples. Examples are grouped by
material and twist angle so the TAPW inputs/outputs and downstream KP configs
stay together.

## Canonical Layout

```text
examples/<material>_<angle>/
  openmx/
  tapw/
  kp/
    configs/
      source/
      model/
        reference/
        diagnostics/
    notebooks/
    outputs/
      plot/<case_id>/
      project/<case_id>/
      symm/<case_id>/
      model/
        <case_id>/
        reference/<case_id>/
        diagnostics/<case_id>/
```

其中：

- `case_id = <material>_<angle>_<valley>`
- 例子：
  - `mote2_3.89_K1`
  - `mgi2_3.89_Gamma`
  - `mgi2_3.89_M1`

## Config Roles

- `configs/source/<case_id>.yaml`
  - `kp plot / kp project / kp symm` 的统一输入
- `configs/model/<case_id>.yaml`
  - canonical continuum model 配置；目录名不携带运行模式语义
  - 只能使用 validated representation 或 `action -> exactify -> continuum_internal_rep_exact`
- `configs/model/reference/<case_id>.yaml`
  - notebook / toy / legacy 参考路径
- `configs/model/diagnostics/<case_id>_<tag>.yaml`
  - raw action / polar / mixed / smoke 等诊断实验

## Active GMK Cases

- K:
  - `examples/mote2_3.89/kp/configs/source/mote2_3.89_K1.yaml`
  - `examples/mote2_3.89/kp/configs/model/mote2_3.89_K1.yaml`
  - `examples/mote2_3.89/kp/configs/source/mote2_3.89_K1_spinful.yaml`
  - `examples/mote2_3.89/kp/configs/model/mote2_3.89_K1_spinful.yaml`
- Gamma:
  - `examples/mgi2_3.89/kp/configs/source/mgi2_3.89_Gamma.yaml`
  - `examples/mgi2_3.89/kp/configs/model/mgi2_3.89_Gamma.yaml`
- M:
  - `examples/mgi2_3.89/kp/configs/source/mgi2_3.89_M1.yaml`
  - `examples/mgi2_3.89/kp/configs/model/mgi2_3.89_M1.yaml`
  - `examples/mgi2_3.89/kp/configs/source/mgi2_3.89_M1_spinful.yaml`
  - `examples/mgi2_3.89/kp/configs/model/mgi2_3.89_M1_spinful.yaml`

## TAPW Templates

- `examples/tapw/basic/`: additional TAPW example configs.
- `examples/tapw/kpaths/`: alternative K-path inputs.
- `tapw-config` uses package templates from `tapw/tapw/templates/`; those are not user examples.

## Cleanup Policy

- active 配置只指向 `kp/outputs/...` 下的 canonical source / model / reference / diagnostics 产物。
- 失败实验、旧命名、过时输出、重复配置，统一移到：

```text
examples/garbage/<material>/<stamp>/...
```

- 不在主树里保留：
  - `_smoke`
  - `_raw_action`
  - `_polar`
  - `_mix_*`
  - 旧的平铺 `*_model*.yaml`
  - 不再引用的输出目录

## Notes

- `tests/kp/test_examples_gmk_pipeline.py` 负责检查这套目录和配置约定。
- K、Gamma、M 和 spinful 示例当前都有 canonical saved model output，可通过 `run_summary.json` 做数值回归。
- 用户面配置只写标准 family name，例如 `C3z`、`TR`、`C2`、`C2T`。
- 几何 action、sector map 和精确化后的 continuum representation 来自 `kp symm` manifest，不在 model YAML 中手写。
- M valley 的 spinless effective 信息只作为 metadata/provenance 保留；active operation label 仍归一化为标准 family name。

## Historical Artifacts

本 checkout 可能仍包含旧的 `kp/runs/...` 和 `kp/outputs/model/production/...` 目录。它们只作为历史比较产物保留；active README 命令和 active YAML 配置不得把这些目录作为输入或输出目标。
