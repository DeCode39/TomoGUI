# Changelog

Versions follow Semantic Versioning. Git tags use `vMAJOR.MINOR.PATCH`.

## [0.2.0] - 2026-11-19

- Keep smooth sinusoidal checkerboards and stage the default 8/4/2 checker-count
  sets. Enforce two model-grid cells per axis per checker and half-domain maximum
  lengths, with live validity feedback and read-only run-plan errors.
- Validate Kd lobe widths; preserve original wavelengths by default, with an
  explicit, recorded fitting option when the grid/domain requires it.
- Make the actual-data baseline ABS-only through relocation, joint inversion,
  and final relocation. Preserve reference observations and pre-omission quality.
- Add adjustable CS/CR pair previews and staged ABS+CR -> ABS+CR+CS -> ABS+new CR
  refinement from the completed baseline model and observed catalogue.
- Default to both-pick top-quality CR with <10 km 3D source separation; CS uses
  <30 degree station azimuth difference and <3 km baseline event displacement.
- Add paired 8/4/2 ABS-control versus differential resolution suites, shared
  noisy absolute data, relative-anisotropy scoring and explicit acceptance gates.
- Fingerprint staged/scored inputs and outputs, reject stale acceptance, and
  rebuild pairs from completed observed-data handoffs at each external stage.
- Expand isolated pair-selection, sinusoid, staging and scoring regression tests;
  update public-repository installation and scientific workflow documentation.

Validation uses isolated catalogues and explicit recovered-model fixtures.
An installed TomoATT forward-mode smoke test also verifies a mixed ABS/CS/CR file.
Production MPI/HPC convergence and scientific resolution still require actual
solver runs and review for each dataset.

## [0.1.0] - 2026-11-01

First GitHub release (alpha).

- Automatic input recognition, read-only source previews, isolated src_rec
  preparation, and numeric/formula/alphabetical quality-to-weight conversion.
- Geographic region filtering with editable bounds and draggable corners.
- Checkerboard counts and HDF5 previews, adjustable grid estimates, and the
  existing dual-layer anisotropy preset.
- Read-only run-folder planning and automatic three-wavelength/eight-coefficient
  Kd staging, weighted scoring and coefficient handoff.
- Checkerboard validation before actual inversion; local MPI and Slurm runners.
- Integrated headless HDF5 reader, cross-file math, slices, masks, 3D samples,
  and PNG export.
- Bundled data-free workflow resources and `tomogui --init NEW_PROJECT` so a
  separate research workspace is not required.
- WSL/Linux installation guides, dependency/execution diagnostics, portable
  generated runners, GitHub CI and version-tagged release assets.

Validation covers synthetic workflows, installed-package initialization,
viewer normalization, and run generation. Production MPI/HPC inversions have
not been validated as part of this release.
