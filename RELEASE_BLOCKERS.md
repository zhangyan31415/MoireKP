# Release Blockers

The following P0 metadata must be resolved before publishing an archival release.

| Item | Status | Required action |
| --- | --- | --- |
| License | TBD | Confirm the project license and add authoritative repository metadata before documenting it in README files. |
| DOI | TBD | Mint or identify the release DOI for the code and release-facing datasets. |
| Data URL | TBD | Publish or identify stable public URLs for the datasets listed in `examples/data-manifest.yaml`. |
| Data checksums | Pending external archive | Replace every `pending_external` marker in `examples/data-manifest.yaml` with either a public archive reference plus checksum metadata or a deliberate exclusion note. |
| Clean-clone smoke | Defined, not archival proof | Keep clean-clone smoke limited to metadata tests and CLI help; TAPW/KP numerical example commands remain external-data workflows until the dataset archive exists. |

Do not infer or copy a license claim from older notes. The README files intentionally avoid a license declaration until the release metadata is confirmed.

The data manifest intentionally uses `license: null`, `doi: null`, and
`data_url: null` while these blockers are unresolved. Local filesystem paths,
untracked generated outputs, and unpublished OpenMX/TAPW/KP artifacts are not
valid substitutes for release metadata.
