# Bilayer MgI2 3.89 degree Gamma- and M-valley examples

The curated cases are:

```text
tapw/configs/mgi2_3.89_Gamma_spinful_q04.yaml
tapw/configs/mgi2_3.89_M1_spinful_q07.yaml
kp/configs/mgi2_3.89_Gamma_spinful_q04.yaml
kp/configs/mgi2_3.89_M1_spinful_q07.yaml
kp/configs/mgi2_3.89_M1_spinless_q07.yaml
```

Gamma remains spinful only. The M1 spinless KP model selects the spin-up
sector of the spinful M1 TAPW Hamiltonian; it is not a separate non-SOC TAPW
calculation.

The TAPW configs use the compact `system` block. `structure/POSCAR` supplies
the lattice and atomic positions, while the OpenMX basis is declared directly
as `{Mg: s2p2, I: s3p2d2}` in YAML. TAPW infers the hexagonal Bravais family
from the POSCAR; automatic inference accepts only hexagonal and square cells
and reports every other in-plane metric as unsupported. The POSCAR cell and
exact site order match the atom/orbital block order of the external symmetrized
H/S matrices; an equivalent structure with reordered sites is not interchangeable.

## Run TAPW

```bash
tapw run  -c examples/mgi2_3.89/tapw/configs/mgi2_3.89_Gamma_spinful_q04.yaml  # external-data
tapw symm -c examples/mgi2_3.89/tapw/configs/mgi2_3.89_Gamma_spinful_q04.yaml  # external-data
tapw run  -c examples/mgi2_3.89/tapw/configs/mgi2_3.89_M1_spinful_q07.yaml     # external-data
tapw symm -c examples/mgi2_3.89/tapw/configs/mgi2_3.89_M1_spinful_q07.yaml     # external-data
```

## Run KP

```bash
kp project -c examples/mgi2_3.89/kp/configs/mgi2_3.89_Gamma_spinful_q04.yaml  # external-data
kp model   -c examples/mgi2_3.89/kp/configs/mgi2_3.89_Gamma_spinful_q04.yaml  # external-data

kp project -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinful_q07.yaml     # external-data
kp model   -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinful_q07.yaml     # external-data

kp project -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinless_q07.yaml    # external-data
kp model   -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinless_q07.yaml    # external-data
```

`kp project` also writes the Q-block eigenspectrum and exactified symmetry
package, so standalone `kp inspect` and `kp symm` are optional diagnostics.
These configs retain the reviewed projection and explicit model choices.
Generated arrays and symmetry exports are external data declared in
`examples/data-manifest.yaml`; the complete local validation checkout exposes
them through relative links.
