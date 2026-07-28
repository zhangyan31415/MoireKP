# PtSe2 OpenMX inputs

The `soc/` and `nsoc/` directories retain the top-level OpenMX inputs and
scientific outputs used by this example. Large files are relative links to
`examples/toZY/{soc,nsoc}`; OpenMX restart scratch directories are intentionally
not duplicated.

The rigid structures are the two small material-local files:

```text
soc/openmx.dat_rigid
soc/POSCAR_rigid
nsoc/openmx.dat_rigid
nsoc/POSCAR_rigid
```

They come from the same AA, twist-index-4, 7.34-degree cell-fixed OpenMX source.
The `DATA.PATH` line in each retained OpenMX input is portable and must be
adjusted to the user's OpenMX `DFT_DATA19` installation before running OpenMX.
The two rigid inputs have identical geometry: 366 atoms in file order
(`Pt` 122, then `Se` 244), with the same 3-by-3 cell and Cartesian coordinates.

`tapw/configs/ptse2_7.34_Gamma_spinful_q04.yaml` consumes:

```text
soc/POSCAR_rigid
soc/H_symm.npz
soc/S_symm.npz
```

`POSCAR_rigid` was generated without atom reordering from
`openmx.dat_rigid`. The calculated `openmx.dat`/`openmx.cif` links document the
relaxed OpenMX geometry used to produce the retained electronic-structure
outputs; they are not substituted for the TAPW rigid structure input.
