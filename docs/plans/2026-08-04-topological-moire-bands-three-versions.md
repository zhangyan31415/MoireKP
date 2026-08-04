# Three-Version Topological Moire Bands Talk Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build three scientifically audited, visually distinct English PPTX/PDF decks for the same 20-minute Topological Moire Bands with MoireKP talk.

**Architecture:** Shared source-data preparation, formulas, conceptual AI imagery, and a common slide-content manifest live under `paper/moirekp/talk/three_versions/shared/`. Each visual direction has an independent PptxGenJS builder, renderer output, tests, and final artifacts under its own variant directory so aesthetic decisions cannot collapse into simple recoloring.

**Tech Stack:** PptxGenJS, Node.js, Python, NumPy, Matplotlib, Pillow, LaTeX/Ghostscript, python-pptx, pypdf, built-in image generation, and independent visual-review agents.

---

### Task 1: Create the shared content and provenance contract

**Files:**
- Create: `paper/moirekp/talk/three_versions/shared/content.json`
- Create: `paper/moirekp/talk/three_versions/shared/provenance.json`
- Create: `paper/moirekp/talk/three_versions/tests/test_shared_contract.py`

**Step 1: Write failing contract tests**

Assert that the shared story contains eleven main slides and two backup slides, that every numerical slide has a local source, and that forbidden sources/claims (`tapw_proxy_from_model`, draft runtime benchmark, topology-field validation) never appear in main-slide evidence.

**Step 2: Run the contract tests and verify failure**

Run: `pytest -q paper/moirekp/talk/three_versions/tests/test_shared_contract.py`

Expected: FAIL because the manifests do not yet exist.

**Step 3: Implement the manifests**

Record titles, timing, claims, source paths, allowed AI-asset roles, and variant-specific visual intents.

**Step 4: Run the tests and verify success**

Expected: all shared-contract tests pass.

### Task 2: Generate and validate shared conceptual imagery

**Files:**
- Create: `paper/moirekp/talk/three_versions/shared/assets/ai/tmd_moire_hero.png`
- Create: `paper/moirekp/talk/three_versions/shared/assets/ai/isospectral_states.png`
- Create: `paper/moirekp/talk/three_versions/shared/assets/ai/finite_q_reduction.png`
- Create: `paper/moirekp/talk/three_versions/shared/assets/ai/prompts.json`
- Test: `paper/moirekp/talk/three_versions/tests/test_ai_assets.py`

**Step 1: Generate one asset per prompt with the built-in image-generation workflow**

Use the approved dark scientific-editorial palette. Include no text, equations, numerical values, axes, plots, or watermarks.

**Step 2: Copy final assets into the workspace**

Do not leave project-referenced images only under the generated-image cache.

**Step 3: Write asset tests**

Verify each PNG exists, is at least 1600 pixels wide, opens successfully, and has a matching prompt/provenance record.

**Step 4: Inspect the images visually**

Reject scientifically misleading TMD structure, fake data, or excessive cyberpunk styling; perform one targeted regeneration where necessary.

### Task 3: Replot the traceable MoTe2 validation evidence

**Files:**
- Create: `paper/moirekp/talk/three_versions/shared/prepare_data_assets.py`
- Create: `paper/moirekp/talk/three_versions/shared/assets/data/mote2_target_residual.png`
- Create: `paper/moirekp/talk/three_versions/shared/assets/data/mote2_error_fingerprint.png`
- Create: `paper/moirekp/talk/three_versions/shared/assets/data/mote2_residual_multiples.png`
- Create: `paper/moirekp/talk/three_versions/shared/assets/data/mote2_full_eight_bands.png`
- Test: `paper/moirekp/talk/three_versions/tests/test_validation_metrics.py`

**Step 1: Write failing metric tests**

Compute the target-band, top-three, and top-eight aligned statistics from `band.npz`. Assert rounded RMS values `0.35`, `0.41`, and `0.95 meV`, and top-eight maximum `2.11 meV`.

**Step 2: Run tests and verify failure**

Expected: FAIL because the shared plotting API does not exist.

**Step 3: Implement the plotting script**

Create separate transparent/dark-compatible outputs for the three treatments. Use direct labels and common k-path ticks; do not fabricate or smooth values.

**Step 4: Run tests and inspect every plot**

Expected: metric tests pass and plots remain readable when placed at 40-55% slide width.

### Task 4: Build Direction A — Moire Night

**Files:**
- Create: `paper/moirekp/talk/three_versions/A_moire_night/build.js`
- Create: `paper/moirekp/talk/three_versions/A_moire_night/render.py`
- Create: `paper/moirekp/talk/three_versions/A_moire_night/Topological_Moire_Bands_Moire_Night.pptx`
- Create: `paper/moirekp/talk/three_versions/A_moire_night/Topological_Moire_Bands_Moire_Night.pdf`
- Create: `paper/moirekp/talk/three_versions/A_moire_night/rendered/`

**Step 1: Write structural tests**

Assert 13 slides, 16:9 dimensions, required titles, dark palette usage, and absence of legacy rail/sidebar/takeaway motifs.

**Step 2: Implement the eleven main slides and two backups**

Use full-bleed imagery, luminous reciprocal-space motifs, transparent plots, and the target-band/residual validation treatment.

**Step 3: Build, render, and run tests**

Expected: valid PPTX, 13-page PDF, and 13 rendered slide PNGs.

**Step 4: Complete spec and visual review**

Fix all actionable issues and rerender.

### Task 5: Build Direction B — Neo-Editorial

**Files:**
- Create: `paper/moirekp/talk/three_versions/B_neo_editorial/build.js`
- Create: `paper/moirekp/talk/three_versions/B_neo_editorial/render.py`
- Create: `paper/moirekp/talk/three_versions/B_neo_editorial/Topological_Moire_Bands_Neo_Editorial.pptx`
- Create: `paper/moirekp/talk/three_versions/B_neo_editorial/Topological_Moire_Bands_Neo_Editorial.pdf`
- Create: `paper/moirekp/talk/three_versions/B_neo_editorial/rendered/`

**Step 1: Write structural tests**

Assert 13 slides, sand/black/Klein-blue fields, asymmetric layouts, and the absence of rounded-card grids and the Direction A palette/motif as the dominant system.

**Step 2: Implement the deck**

Use large editorial crops, hard rectangular fields, oversized words/numbers, and the error-fingerprint validation treatment.

**Step 3: Build, render, and run tests**

Expected: valid PPTX/PDF and full rendered output.

**Step 4: Complete spec and visual review**

Fix all actionable issues and rerender.

### Task 6: Build Direction C — K-space Cartography

**Files:**
- Create: `paper/moirekp/talk/three_versions/C_kspace_cartography/build.js`
- Create: `paper/moirekp/talk/three_versions/C_kspace_cartography/render.py`
- Create: `paper/moirekp/talk/three_versions/C_kspace_cartography/Topological_Moire_Bands_KSpace_Cartography.pptx`
- Create: `paper/moirekp/talk/three_versions/C_kspace_cartography/Topological_Moire_Bands_KSpace_Cartography.pdf`
- Create: `paper/moirekp/talk/three_versions/C_kspace_cartography/rendered/`

**Step 1: Write structural tests**

Assert 13 slides, location breadcrumbs, recurring map/hexagon coordinates, and distinct purple/aqua/magenta behavior.

**Step 2: Implement the deck**

Use zoom transitions, contour fields, candidate-focus windows, and residual-small-multiple validation.

**Step 3: Build, render, and run tests**

Expected: valid PPTX/PDF and full rendered output.

**Step 4: Complete spec and visual review**

Fix all actionable issues and rerender.

### Task 7: Cross-version scientific and presentation QA

**Files:**
- Create: `paper/moirekp/talk/three_versions/tests/test_all_decks.py`
- Create: `paper/moirekp/talk/three_versions/README.md`

**Step 1: Validate all packages and extracted text**

Check slide count, dimensions, formula/text presence, citations, and absence of placeholders and forbidden claims.

**Step 2: Compare the three contact sheets**

Require visibly different palettes, layout rhythms, validation treatments, and recurring motifs.

**Step 3: Request independent scientific and visual reviews**

Use separate reviewers for source integrity and room-scale aesthetics. Correct all important findings.

**Step 4: Run the final verification suite**

Run: `pytest -q paper/moirekp/talk/three_versions/tests`

Expected: all tests pass; each PPTX and PDF has thirteen slides/pages.

**Step 5: Document outputs**

List all PPTX/PDF paths, style summaries, generated-image prompts, source-data provenance, and the recommended viewing order.

