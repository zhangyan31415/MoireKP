# Autoproject Examples Design

## Goal

Provide a release-facing `examples/autoproject/` collection whose ten configurations reproduce the automatic low-energy state choices already validated for MoTe2, MgI2, ZrS2, multilayer A-AB MoTe2, and PtSe2. Running `kp project <config>` must print and save the resolved band indices together with layer, spin, and dominant orbital weights.

## Approaches considered

1. Duplicate every TAPW Hamiltonian below `examples/autoproject/`. This is physically self-contained but duplicates large external data and violates the release workspace's generated-data policy.
2. Keep only wrapper configs that use machine-specific absolute paths. This is easy locally but is not portable or release-safe.
3. Keep portable configs, documentation, expected selections, and a batch runner below `examples/autoproject/`, while resolving the existing curated TAPW inputs through repository-relative paths. This is the selected approach. Large inputs remain governed by `examples/data-manifest.yaml`.

## Directory layout

```text
examples/autoproject/
  README.md
  expected_selections.yaml
  run_all.py
  configs/
    mote2_k_spinless.yaml
    mote2_k_spinful.yaml
    mgi2_m_spinless.yaml
    mgi2_m_spinful.yaml
    mgi2_gamma.yaml
    zrs2_gamma.yaml
    mote2_aab_k_a.yaml
    mote2_aab_k_b.yaml
    mote2_aab_gamma.yaml
    ptse2_gamma.yaml
```

Each config uses `project.selection.mode: auto`, contains no hand-written `project.nlow_state_list`, writes into `examples/autoproject/runs/<case>/`, and references curated material data by relative path. Generated outputs stay ignored.

## Project output

The existing `low_energy_selection.json` remains the machine-readable source of truth. A small formatter renders a terminal summary after the automatic decision is available:

```text
[kp] automatic low-energy selection: PASS
[kp] resolved nlow_state_list = [[40, 41], [42, 43]]
[kp] selected Gamma_joint:40 -> layer 1; spin up=... down=...; orbitals p_x=..., p_y=...
```

Orbital entries are sorted by descending weight and truncated only for terminal readability; the JSON and Markdown reports retain the complete grouped weights. Empty inactive layers remain visible in `nlow_state_list`.

## Validation and warnings

The ten expected selections are checked against `expected_selections.yaml`. PASS cases must exit zero and match exactly. The two A-AB K sector cases may exit zero with the already understood best-available warning, but their selected states and subsequent production symmetry validation must match the expected result. MgI2 M spinless records the known cross-spin sewing limitation explicitly rather than hiding it.

Runs are distributed across bigmem001--003 without modifying the example configs. Each node writes only below `examples/autoproject/runs/` or `validation_runs/`; no node paths or logs enter release-facing files.

## Tests

- Unit-test deterministic terminal formatting, including orbital ordering and empty layer rows.
- Contract-test that all ten configs use automatic selection and contain no explicit `nlow_state_list`.
- Contract-test portable paths and expected selections.
- Run all ten projects on bigmem and compare JSON reports with the expected manifest.

