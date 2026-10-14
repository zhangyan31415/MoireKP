# Final cleanup code review

Reviewed the current shared checkout diff and new model responsibility modules. Review focused on the scientific computation route, explicit failure behavior, response cache identity, and symmetry-to-model artifact geometry.

Two concrete artifact-reader issues were found and corrected after authorization:

1. `configuration._load_q_sets_from_symmetry_artifact` previously fell through from an absent packed canonical Q pair to `q_model.files`. The current symmetry producer labels those `.npy` files `raw_model_q`; they are saved before canonicalization. Returning them bypassed `_load_model_q_sets`' requirement for canonical geometry. The reader now supplies only certified/hash-verified packed canonical arrays. Missing canonical arrays produce the existing explicit `rerun kp symm` error. Explicit/manual symmetry sources retain their existing Q input route.
2. `_load_symmetry_manifest` previously reconstructed operations and a zero-angle frame from a metadata-less `representations.npz`. Current packed output always includes `__metadata_json__`. This historical inference branch now raises an explicit error instead of fabricating scientific provenance.

The two regression tests failed before the fixes with `DID NOT RAISE`. After the fixes, canonical artifact tests plus complete response/compiler tests passed: **283 passed in 7.85 s** on the final rerun after removing the now-dead helper chain. Deleted unused `_load_symmetry_artifact_manifest` and `_packed_symmetry_operation_metadata`; a repository/test search confirmed no remaining callers. The general operation metadata helpers still have legitimate current template/configuration callers and are retained.

The `manifest.json`/`summary.json` dual reader is not an old-format migration: the current producer writes the exact same serialized payload to both files. Those names do not select different mathematical implementations. The packed archive's embedded metadata is another current representation of the same output, rather than a metadata inference fallback.

The response production route uses current mathematical names and explicit SVD/ridge/physical minimum-norm policies. Compiler errors and certification failures propagate. Local versus general Reynolds coordinates are selected from route availability and owner-vocabulary injectivity; an ensuing certification failure cannot trigger an oracle. Independent dense/symbolic mathematical references remain separate. Cache identity hashes installed implementation content and numerical library releases, includes the shared installed `tapw.identity` dependency, and does not query Git or use a guessed version.

No additional concrete numerical regression was found in this review. This is a code and small-matrix regression review; real-material and installed-wheel acceptance remain distinct evidence supplied by the parent task. Existing raw/canonical Q residual correction, phase construction and numerical tolerances are unchanged.
