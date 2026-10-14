# Release Blockers

The following release metadata records what is resolved and what still blocks an archival release.

| Item | Status | Required action |
| --- | --- | --- |
| Software license | Resolved | MoireKP software is licensed under `LGPL-3.0-or-later`; authoritative terms are in `COPYRIGHT`, `COPYING.LESSER`, and `COPYING`. |
| Dataset licenses | TBD | Confirm the license and provenance of every external dataset listed in `examples/data-manifest.yaml`; do not infer these terms from the software license. |
| DOI | TBD | Mint or identify the release DOI for the code and release-facing datasets. |
| Data URL | TBD | Publish or identify stable public URLs for the datasets listed in `examples/data-manifest.yaml`. |
| Data checksums | Pending external archive | Replace every `pending_external` marker in `examples/data-manifest.yaml` with either a public archive reference plus checksum metadata or a deliberate exclusion note. |
| Clean-clone smoke | Defined, not archival proof | Keep clean-clone smoke limited to metadata tests and CLI help; TAPW/KP numerical example commands remain external-data workflows until the dataset archive exists. |

The data manifest intentionally uses `license: null`, `doi: null`, and
`data_url: null` for external datasets while these blockers are unresolved.
The repository's software license does not fill those dataset fields. Local filesystem paths,
untracked generated outputs, and unpublished OpenMX/TAPW/KP artifacts are not
valid substitutes for release metadata.
