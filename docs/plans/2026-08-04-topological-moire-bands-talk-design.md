# Topological Moire Bands Talk Design

## Purpose and constraints

- Deliverable: an English 16:9 academic presentation in PPTX and PDF.
- Location: `paper/moirekp/talk/`.
- Timing: 20 minutes total, budgeted as 15 minutes of prepared remarks and 5 minutes for questions.
- Audience: condensed-matter and computational-materials researchers with general familiarity with moire physics, but not necessarily with MoireKP internals.
- Objective: connect a high-throughput map of twisted semiconductors to automated, symmetry-constrained continuum-model construction.

## Narrative choice

The talk uses a discovery-to-modeling narrative:

1. Twisting opens a large design space for flat and topological bands.
2. High-throughput calculations turn the design space into an angle-resolved map.
3. Valley character and stacking symmetry organize that map.
4. The map identifies promising systems, but large first-principles Hamiltonians remain difficult to use for topology, geometry, and many-body studies.
5. MoireKP converts those Hamiltonians into compact matrix-faithful continuum models.
6. K-, M-, and Gamma-valley examples establish validation across distinct symmetry settings.

The audience should leave with one sentence: high-throughput mapping tells us where to look; MoireKP produces the compact models needed to explain and exploit what we find.

## Slide blueprint

### Main talk

1. **Title** - topic, presenter, affiliation, and two-source framing.
2. **Twist turns semiconductors into a design space** - atomic-to-moire scale hierarchy and the moire-length formula.
3. **The search space outgrows one-material-at-a-time studies** - material, stacking, angle, and valley axes with the database scale: 43 monolayers, 91 bilayer prototypes, and more than 1,000 moire band structures.
4. **A hierarchical workflow makes the landscape computable** - screening, stacking enumeration, GSFE/MLFF relaxation, DFT/TB, TAPW, topology, and quantum geometry.
5. **Valley character organizes the global map** - cropped high-throughput bandwidth/topology heat map with a direct legend.
6. **Bandwidth follows a valley-dependent hierarchy** - Gamma nearly quadratic, K stacking-sensitive, and M material-specific; include `W(theta) proportional to theta^2` and a compact comparison graphic.
7. **Twisting generates topology selectively** - Gamma Z2, K valley Chern, and M nonsymmorphic regimes, with stacking as the control knob.
8. **A map identifies targets, but not a usable low-energy theory** - contrast large source Hamiltonians, band-only fits, and matrix-faithful continuum models.
9. **MoireKP automates the first-principles-to-continuum reduction** - source projection, low-energy basis, symmetry transfer, allowed terms, and matrix least squares.
10. **One workflow covers K, M, and Gamma valleys** - MoTe2 K and MgI2 M/Gamma validation figures with concise metrics and observables.
11. **From catalogue to predictive models** - three takeaways and a question slide.

### Backup slides

12. **BiTeCl: termination and stacking tune topology**.
13. **Quantum-geometry screening highlights many-body candidates**.
14. **MoireKP equations and validation details**.

## Formula policy

Only formulas that directly support the story appear in the main talk:

- Moire scale: `L_m = a / [2 sin(theta/2)]`, with the small-angle approximation `L_m approximately a/theta`.
- Folding-dominated bandwidth: `W(theta) proportional to theta^2` for the regular Gamma-valley regime.
- Matrix-level construction: `H_cont(k) = sum_alpha c_alpha B_alpha(k)` and `c* = argmin_c sum_k ||H_eff(k) - sum_alpha c_alpha B_alpha(k)||_F^2`.

Detailed projection and symmetry equations are confined to the final backup slide.

## Visual system

- Format: 16:9 widescreen.
- Rhythm: dark title and conclusion slides, light analytical slides.
- Dominant colors: graphite `151820` and warm off-white `F6F3EC`.
- Scientific accents: Berry-curvature magenta `D84A8A`, topology cyan `19A7AE`, and flat-band gold `E7A93B`.
- Typography: Aptos Display or Arial for headings and Aptos/Arial for body text to ensure cross-platform rendering.
- Motif: a faint reciprocal-space hexagon/Q-lattice pattern and small valley badges using consistent Gamma/K/M colors.
- Every slide carries a figure, diagram, equation, or data visualization; no slide is a plain bullet list.
- Sources are shown in a small, high-contrast footer on every slide that uses published or manuscript figures.

Public KITP moire talks are used only as pacing and layout references. Their useful conventions are large action-oriented titles, one claim per slide, figure-led explanations, and compact citations. All scientific figures in the final deck come from the two user-provided local paper directories or are newly drawn schematic elements.

## Quality gates

1. Extract presentation text and check order, spelling, symbols, placeholders, and citations.
2. Export PPTX to PDF and render every page to images.
3. Inspect for overlap, clipping, illegible labels, inconsistent margins, weak contrast, and over-dense panels.
4. Perform at least one fix-and-rerender cycle.
5. Use an independent visual-review agent on the rendered slides and resolve all actionable findings.

