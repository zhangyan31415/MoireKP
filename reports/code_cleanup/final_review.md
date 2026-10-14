# Final cleanup code review

Reviewed the current shared checkout diff and new model responsibility modules. Review focused on the scientific computation route, explicit failure behavior, response cache identity, and symmetry-to-model artifact geometry.

Two concrete artifact-reader issues were found and corrected after authorization:

1. `configuration._load_q_sets_from_symmetry_artifact` previously fell through from an absent packed canonical Q pair to `q_model.files`. The current symmetry producer labels those `.npy` files `raw_model_q`; they are saved before canonicalization. Returning them bypassed `_load_model_q_sets`' requirement for canonical geometry. The reader now supplies only certified/hash-verified packed canonical arrays. Missing canonical arrays produce the existing explicit `rerun kp symm` error. Explicit/manual symmetry sources retain their existing Q input route.
2. `_load_symmetry_manifest` previously reconstructed operations and a zero-angle frame from a metadata-less `representations.npz`. Current packed output always includes `__metadata_json__`. This historical inference branch now raises an explicit error instead of fabricating scientific provenance.

The two regression tests failed before the fixes with `DID NOT RAISE`. After the fixes, canonical artifact tests plus complete response/compiler tests passed: **283 passed in 7.85 s** on the final rerun after removing the now-dead helper chain. Deleted unused `_load_symmetry_artifact_manifest` and `_packed_symmetry_operation_metadata`; a repository/test search confirmed no remaining callers. The general operation metadata helpers still have legitimate current template/configuration callers and are retained.

The `manifest.json`/`summary.json` dual reader is not an old-format migration: the current producer writes the exact same serialized payload to both files. Those names do not select different mathematical implementations. The packed archive's embedded metadata is another current representation of the same output, rather than a metadata inference fallback.

The response production route uses current mathematical names and explicit SVD/ridge/physical minimum-norm policies. Compiler errors and certification failures propagate. Local versus general Reynolds coordinates are selected from route availability and owner-vocabulary injectivity; an ensuing certification failure cannot trigger an oracle. Independent dense/symbolic mathematical references remain separate. Cache identity hashes installed implementation content and numerical library releases, includes the shared installed `tapw.identity` dependency, and does not query Git or use a guessed version.

No additional concrete numerical regression was found in this review. This is a code and small-matrix regression review; real-material and installed-wheel acceptance remain distinct evidence supplied by the parent task. Existing raw/canonical Q residual correction, phase construction and numerical tolerances are unchanged.

## Producer-consumer coordinate format regression

The real published ZrS2 M model run exposed a missed cross-module serialization contract: the symmetry producer wrote `right_handed_model_cartesian_reciprocal_v1`, while the response consumer's literal had been renamed to the untagged value. This literal is the single current stored coordinate discriminator, rather than an algorithm development label. The mismatch was introduced by the cleanup.

The current tag is now defined once as `COORDINATE_CONVENTION` in existing `symmetry/q_canonicalization.py`. Both the symmetry producer and response consumer import it directly. No alternate value, alias parser or migration branch is accepted. The mathematical axes, origin, Q geometry and numerical tolerances are unchanged.

The existing minimal actual `kp project`/`kp symm` integration test now passes the resulting archive metadata to the model's `_exactified_polynomial_origin` and constructs its polynomial coordinates; it no longer merely checks a duplicated expected literal. This test reproduced the production exception before the fix. The full actual-producer, canonical-artifact, storage, response and compiler suites then passed **374 tests in 11.34 s**. The restored current literal appears at exactly one production definition. Other renamed response normalization/center/action conventions were checked across producers and consumers; no further literal mismatch was found.

Final source change is limited to the shared geometry constant and its two direct imports/emission: `symmetry/q_canonicalization.py`, `symmetry/projection.py`, `model/response_basis.py`. Existing synthetic model fixtures were updated to the single current stored convention. A new installed build must rerun the model stage against the already-certified Heff/Q/representation artifacts; projection and symmetry need no recomputation for this serialization correction.
