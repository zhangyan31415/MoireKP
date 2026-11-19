# count43_spin_up_projected

This directory contains a NumPy-only evaluator for `count43_spin_up_projected`.

## Requirements

Python 3 and NumPy.

## Run

```bash
python evaluate.py
```

This writes:

- `kpoints.npy`
- `bands.npy`
- `bands.pdf`
- `kdist.npy`
- `kpath_ticks.json`

## Edit the k path

Open `evaluate.py` and edit:

- `HIGH_SYMMETRY_POINTS`
- `KPATH`
- `POINTS_PER_SEGMENT`
- `WINDOW_BANDS`
- `WINDOW_EDGE`

## Files

- `evaluate.py`: standalone evaluator
- `model_data.npz`: arrays used by the evaluator
- `MODEL.md`: model formula and basis notes

- `model.toml` + `terms.csv`: authoritative expandable physical model
- `q_points.csv`: validated default finite Q realization
- `physical_model.py`: NumPy route evaluator (`load_model('.')`)
- `symmetry.toml`: sector-aware affine Q actions, when available
