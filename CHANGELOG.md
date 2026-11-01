# Changelog

Versions follow Semantic Versioning. Git tags use `vMAJOR.MINOR.PATCH`.

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
