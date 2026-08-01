# MoireKP LGPL-3.0-or-later Licensing Design

## Objective

Publish the MoireKP software under `LGPL-3.0-or-later` with authoritative
repository and Python package metadata, while keeping external dataset
licensing independent from the software license.

## Copyright and license decisions

- Software license: `LGPL-3.0-or-later`.
- Copyright notice:

  ```text
  Copyright (C) 2026 Yan Zhang, Jiabin Yu, and Quansheng Wu
  ```

- The copyright names and order follow the manuscript author block.
- The GNU license texts remain verbatim. Project copyright information is kept
  in a separate `COPYRIGHT` file and is not substituted into the Free Software
  Foundation's license text.
- Existing source files will not receive a bulk copyright-header rewrite.
  New files should use:

  ```text
  SPDX-License-Identifier: LGPL-3.0-or-later
  ```

## Repository files and package metadata

1. Add `COPYING` containing the complete GNU General Public License version 3.
2. Add `COPYING.LESSER` containing the complete GNU Lesser General Public
   License version 3.
3. Add `COPYRIGHT` with the project name, the approved copyright notice, the
   SPDX identifier, and a pointer to both GNU license files.
4. Use PEP 639 metadata in `pyproject.toml`:

   ```toml
   license = "LGPL-3.0-or-later"
   license-files = ["COPYING", "COPYING.LESSER", "COPYRIGHT"]
   ```

5. Raise the build requirement to `setuptools>=77`, the first setuptools
   series with support for PEP 639 SPDX expressions and `project.license-files`.
6. Add concise License sections to `README.md` and `README.zh.md`.

## Software and dataset boundary

The LGPL declaration covers the MoireKP/TAPW/KP source code and the
repository-authored documentation, configuration files, and small examples
distributed with the software package. It does not assign LGPL terms to
external OpenMX matrices, TAPW arrays, generated KP outputs, or other archived
datasets.

Release documentation and tests will distinguish the resolved software
license from still-pending dataset license metadata. Dataset `license` fields
in `examples/data-manifest.yaml` remain unresolved until the corresponding
data owners and archive terms are confirmed.

## Validation

- Check the exact SPDX expression in `pyproject.toml`.
- Check that `COPYING`, `COPYING.LESSER`, and `COPYRIGHT` are tracked and
  included in both sdist and wheel artifacts.
- Check that wheel metadata contains `License-Expression:
  LGPL-3.0-or-later` and the expected `License-File` entries.
- Check that README and release-blocker language distinguishes software from
  dataset licensing.
- Run the release-contract tests and the default non-slow, non-external-data
  test suite.

## Non-goals

- Do not license external datasets as LGPL.
- Do not alter manuscript files.
- Do not infer additional copyright holders from Git commit metadata.
- Do not bulk-add personal copyright notices to historical source files.
