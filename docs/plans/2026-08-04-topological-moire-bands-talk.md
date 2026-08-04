# Topological Moire Bands Talk Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create an English 16:9 academic talk and matching PDF that connect high-throughput mapping of topological moire bands with automated continuum-model construction in MoireKP.

**Architecture:** A reproducible PptxGenJS build script will compose editable text and shapes with rasterized manuscript figures. Shared slide-layout helpers will enforce the approved visual system. A PDF export and PNG renders will be generated for independent visual QA, followed by at least one correction cycle.

**Tech Stack:** Node.js, PptxGenJS, Python/Pillow/Matplotlib, Ghostscript, LibreOffice when available, `python-pptx`, and `pypdf`.

---

### Task 1: Prepare report assets

**Files:**
- Create: `paper/moirekp/talk/package.json`
- Create: `paper/moirekp/talk/prepare_assets.py`
- Create: `paper/moirekp/talk/assets/`

**Step 1: Record the required Node dependency**

Create a minimal package manifest containing `pptxgenjs` and a `build` command.

**Step 2: Render the manuscript figures**

Use Ghostscript at presentation resolution to rasterize:

- `paper/moirekp/arxiv_references/2607.25172/source/fig_workflow_all.pdf`
- `paper/moirekp/arxiv_references/2607.25172/source/fig_band_width_moire_HSP.pdf`
- `paper/moirekp/arxiv_references/2607.25172/source/fig_BiTeCl_moire.pdf`
- `paper/moirekp/arxiv_references/2607.25172/supplement_0804/delta_g.pdf`
- selected MoireKP workflow, valley-selection, and validation figures.

**Step 3: Generate presentation-native scientific graphics**

Create transparent PNG/SVG assets for the moire-length formula, bandwidth scaling, matrix-fit equation, hexagonal reciprocal-space motif, and valley hierarchy.

**Step 4: Verify the assets**

Run:

```bash
python paper/moirekp/talk/prepare_assets.py
python - <<'PY'
from pathlib import Path
from PIL import Image
root = Path('paper/moirekp/talk/assets')
files = sorted(root.glob('*.png'))
assert files
for path in files:
    with Image.open(path) as image:
        assert image.width >= 600 and image.height >= 300, path
print(f'{len(files)} assets verified')
PY
```

Expected: all generated image assets open successfully and meet minimum dimensions.

### Task 2: Build the presentation

**Files:**
- Create: `paper/moirekp/talk/build_talk.js`
- Create: `paper/moirekp/talk/Topological_Moire_Bands_with_MoireKP.pptx`

**Step 1: Implement shared layout helpers**

Define the palette, typography, title block, footer/citation line, valley badge, image-contain/image-cover helpers, and slide-number treatment.

**Step 2: Implement the 11 main slides**

Follow the approved blueprint with one claim per slide, large titles, concise labels, editable formulas when practical, and source figures from the two local manuscripts.

**Step 3: Implement three backup slides**

Add BiTeCl, quantum geometry, and MoireKP-equation backup slides after the questions slide and label them clearly as backup material.

**Step 4: Build the PPTX**

Run:

```bash
cd paper/moirekp/talk
npm install --ignore-scripts
npm run build
```

Expected: `Topological_Moire_Bands_with_MoireKP.pptx` is generated without PptxGenJS errors.

### Task 3: Verify content and package integrity

**Files:**
- Inspect: `paper/moirekp/talk/Topological_Moire_Bands_with_MoireKP.pptx`

**Step 1: Validate the ZIP package**

Run:

```bash
unzip -t paper/moirekp/talk/Topological_Moire_Bands_with_MoireKP.pptx
```

Expected: no package errors.

**Step 2: Extract slide text**

Use `python-pptx` to print the text from all slides and verify slide count, order, title wording, formula symbols, citations, and the absence of placeholders.

**Step 3: Check presentation metadata and dimensions**

Verify a 16:9 page size, 14 slides, and editable slide titles/body text.

### Task 4: Export PDF and render slides

**Files:**
- Create: `paper/moirekp/talk/Topological_Moire_Bands_with_MoireKP.pdf`
- Create: `paper/moirekp/talk/rendered/slide-*.png`
- Create: `paper/moirekp/talk/rendered/contact-sheet.jpg`

**Step 1: Export PPTX to PDF**

Use LibreOffice Impress if available. If it is unavailable, generate the PDF from the same slide-layout source and disclose the fallback in build notes.

**Step 2: Render the PDF**

Use Ghostscript to render every page to PNG at 150-180 dpi.

**Step 3: Create a contact sheet**

Compose a labeled contact sheet with Pillow for fast whole-deck inspection.

### Task 5: Visual QA and correction cycle

**Files:**
- Modify as needed: `paper/moirekp/talk/build_talk.js`
- Regenerate: PPTX, PDF, and rendered images.

**Step 1: Inspect every rendered slide**

Check overlap, clipping, contrast, margins, text density, citation collisions, panel legibility, and visual rhythm.

**Step 2: Request independent visual review**

Provide the contact sheet and selected full-resolution slides to a fresh review agent and ask it to report all issues, including minor ones.

**Step 3: Fix all actionable issues**

Adjust the layout or asset preparation and rebuild the deck.

**Step 4: Rerender and reverify**

Repeat the complete text/package checks and visual inspection. Do not deliver until a fresh pass finds no unresolved formatting defects.

### Task 6: Final handoff

**Files:**
- Final: `paper/moirekp/talk/Topological_Moire_Bands_with_MoireKP.pptx`
- Final: `paper/moirekp/talk/Topological_Moire_Bands_with_MoireKP.pdf`
- Supporting: `paper/moirekp/talk/build_talk.js`
- Supporting: `paper/moirekp/talk/prepare_assets.py`

**Step 1: Confirm final artifacts**

Report file sizes, slide/page counts, and the exact verification commands that passed.

**Step 2: Summarize assumptions**

State that the title slide uses Yan Zhang and Institute of Physics, CAS as the editable default presenter identity.

