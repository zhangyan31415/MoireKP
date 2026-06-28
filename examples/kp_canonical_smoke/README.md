# KP Canonical Smoke Examples

These tiny synthetic examples exercise the release-facing KP canonical layout
without external TAPW/OpenMX data. They are for CLI and I/O checks, not for
physical validation.

## Prepare Inputs

```bash
python examples/kp_canonical_smoke/generate_inputs.py
```

This writes small ignored `.npy` files under `examples/kp_canonical_smoke/data/`.

## K1 Example

```bash
kp inspect -c examples/kp_canonical_smoke/kp/configs/K1_q01.yaml
kp project -c examples/kp_canonical_smoke/kp/configs/K1_q01.yaml
```

Expected outputs:

```text
examples/kp_canonical_smoke/kp/outputs/K1/q01/
  summary.md
  inspect/
    spectrum.txt
    candidates.md
    scatter.png
    blocks.csv
  projection/
    heff.npy
    eigvals.npy
    eigvals.txt
    basis.md
    basis.npz
    scatter.png
    vectors.npy
    k_indices.npy
    basis_selection.json
    basis_selection.md
    heff_list.npy
    heff_eig.npy
    heff_vec.npy
```

## Gamma Example

```bash
kp inspect -c examples/kp_canonical_smoke/kp/configs/Gamma_q01.yaml
kp project -c examples/kp_canonical_smoke/kp/configs/Gamma_q01.yaml
```

Expected outputs:

```text
examples/kp_canonical_smoke/kp/outputs/Gamma/q01/
  summary.md
  inspect/
    spectrum.txt
    candidates.md
    scatter.png
    blocks.csv
  projection/
    heff.npy
    eigvals.npy
    eigvals.txt
    basis.md
    basis.npz
    scatter.png
    vectors.npy
    k_indices.npy
    basis_selection.json
    basis_selection.md
    heff_list.npy
    heff_eig.npy
    heff_vec.npy
```
