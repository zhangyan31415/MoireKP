# KP Automatic Low-Energy Subspace Selection Design

**Date:** 2026-07-18

## Goal

Replace ordinary-user `hamk_index`, `ref_q_index`, and
`project.nlow_state_list` choices with an automatic search for the smallest
low-energy subspace that is close to the requested Fermi-level edge, closed
under the TAPW source symmetries, and accurate after projection.

The selected local orbitals are defined once at the valley expansion origin
and at the minimum-|Q| reference block.  Their block-band indices are then
held fixed for every Q and every k.  The selector must not adaptively switch
band indices at high Q or away from the expansion origin.

## User-Facing Contract

For an ordinary case, the user states the physical edge through the existing
case/profile/model contract.  The projection may omit all three internal
indices:

```yaml
project:
  selection: auto
```

The existing explicit `nlow_state_list` remains an expert and compatibility
override.  `hamk_index` and flattened `ref_q_index` are not part of the
ordinary projection contract.

## Reference Point Resolution

The selector resolves the reference Hamiltonian row from the canonical TAPW
`kpoints.npy` artifact and valley metadata.  It selects the row at the local
valley expansion origin, k=0.  The resolved array index and coordinate are
reported.  A missing expansion-origin row is a structural error rather than a
reason to silently use array row zero.

For each source group, the selector chooses the Q vector with minimum norm.
For non-Gamma valleys, each physical layer in the group uses its corresponding
minimum-|Q| block.  For Gamma, the selector uses the single same-Q block that
jointly contains all physical layers.  Exact norm ties use canonical Q-array
order and are reported.

## Candidate Generation

At K and M, the selector diagonalizes the minimum-|Q| block of every physical
layer and pools the resulting states with explicit physical-layer labels.  At
Gamma, it diagonalizes the joint multilayer minimum-|Q| block and does not
assign layer ownership before symmetry analysis.

States are ordered by distance from the requested side of the Fermi level.
The first seed is the nearest energy cluster; a nondegenerate seed may contain
one state.  Subsequent seeds add the next energy cluster.  Candidate sizes are
not restricted to even numbers or powers of two.

For every seed, the TAPW raw-H source actions are applied to the seed
projector.  Eigenstates receiving the symmetry image are added until the
off-subspace symmetry leakage passes the configured threshold or no closure
can be formed.  This closure may add partners in another spin, physical layer,
or continuum sector.  A spinless K-valley state may remain one-dimensional.
At a spinful time-reversal-invariant point, internal time reversal may require
a Kramers pair.  In the PtSe2 Gamma case, a seed at block band 54 closes with
band 55; only after closure are the two states assigned to the exchanged
continuum sectors.

The resolved band list for each physical layer or Gamma joint block is frozen
after this reference-point selection.  Projection uses the same band indices
at every Q and every k.  High-Q band-index crossings do not redefine the
continuum orbitals.

## Candidate Evaluation And Selection

Each structurally valid candidate is projected with the production projection
path and evaluated using:

- RMS and maximum projected-band error against the target TAPW bands;
- principal-angle target-subspace overlap;
- projected source-symmetry residual and off-subspace leakage;
- existing projection rank and downfolding stability checks.

The state selection is performed only at k=0 and minimum |Q|.  Band error,
subspace overlap, and symmetry residual may be evaluated over the configured
validation k points without changing the frozen selection.

Candidates that pass all quality thresholds are ordered lexicographically by
local-orbital dimension, band error, overlap deficit, and symmetry residual.
The smallest passing candidate is selected.

If no structurally valid candidate passes every quality threshold, selection
continues with a warning.  For each metric, its threshold violation is
normalized by the threshold.  The fallback minimizes the maximum normalized
violation, then the total normalized violation, then band error and dimension.
This avoids an opaque weighted sum.

Structural failures remain fatal: missing or contradictory layout metadata,
an impossible symmetry orbit, projector rank loss, non-Hermitian output, or a
numerically singular downfolding path must not be converted into a warning.

## Commands And Artifacts

`kp inspect` and `kp project` call the same deterministic selection engine.
`kp inspect` is a dry run that reports candidate closures and projected
quality when the necessary inputs are available.  `kp project` reruns or
verifies the same resolution and writes the selected projector.  Running
`kp inspect` first is useful but not required.

The canonical projection output contains `low_energy_selection.json` and
`low_energy_selection.md`.  The terminal and report state:

- PASS or WARN selection status;
- resolved k=0 row and per-group minimum-|Q| references;
- selected fixed block-band indices and their symmetry-orbit relations;
- resolved continuum-sector assignment;
- orbital, physical-layer, and spin weights at the reference block;
- band RMS/max error, subspace overlap, and symmetry residual;
- rejected candidates and their threshold violations;
- confirmation that the selected indices are applied unchanged to every Q
  and every k.

Layer, orbital, and spin weights describe the selected states; they are not
ordinary-user selection inputs.  Delocalized Gamma states are reported as
weights rather than assigned a false unique physical-layer label.

## Representative Regression Cases

The automatic resolver is tested against the current curated contracts:

- MoTe2 K spinless: one local state per active layer;
- MoTe2 K spinful: the current two-state spinful selection;
- MgI2 M spinless and spinful: one-state and Kramers-paired behavior;
- MgI2 Gamma: the current four-state joint closure;
- ZrS2 Gamma: the current eight-state joint closure;
- 1+2 A-AB MoTe2 K-A/K-B: inactive physical layers and a one-orbital active
  sector;
- 1+2 A-AB MoTe2 Gamma: multilayer joint-block row ordering;
- PtSe2 Gamma: bands 54 and 55 as a time-reversal-exchanged two-sector orbit.

Unit tests use synthetic matrices for deterministic selection, closure,
fallback ranking, and reporting.  External-data tests compare the automatically
resolved selections and projection metrics with the curated material cases.

