# Workflow and scientific behavior

## Project and data

Create a new project using `tomogui --init NEW_DIRECTORY`. This copies a versioned,
data-free snapshot of the reused workflow helpers. It never overwrites an
existing directory. Projects from the original research workspace still work;
they use their existing scripts/templates rather than replacing them on launch.

Place inputs inside the project. The Data tab shows a bounded, read-only source
preview, detects the structure, and prepares an isolated copy under `.tomogui/`.
The original file is not rewritten. Catalog conversion may need station metadata.
Event-only tables can be previewed but cannot create missing arrival picks.

The preparation workflow accepts absolute 8/9-field receiver rows. Differential
and extended layouts are rejected explicitly. Some PyTomoATT versions omit
receiver indentation; owned copies are normalized for the legacy helpers.

Quality conversion supports ordered numeric comparisons, scalar formulas with
`q`, `min`, `max`, `abs` and `clip`, and case-insensitive grade labels. Grade
weights are editable; there is no universal scientific A/B/C/D weight scale.
Set an explicit policy for missing values and unmatched grades. Formulas do not
execute arbitrary Python. Custom conversions affect pick weights, preserve
event-weight calculations and bypass automatic CWB-code remapping for that job.

## Region and grids

Drag the rectangle corners or edit bounds. Changes affect the selection preview;
Apply region runs the existing single-region scanner and writes final src_rec
files and metadata. P-only filtering disables the S-pick minimum and vice versa.
The original single-region defaults are used for combined-phase filtering.

Grid estimates use the phase-specific final src_rec receiver spacing, not the
unfiltered input footprint. Defaults retain a lateral multiplier of 1, depth
spacing of 1 km, the existing lateral buffer and an inversion grid at about
75% of the stored grid counts using the original rounding. Receiver elevations
extend the depth range. Missing station-spacing estimates use the original
25-points/degree fallback. Tune spacing and scale explicitly in Run setup.

## Models

The current checkerboard defaults are `[2,2,2]`, velocity perturbation `0.2`,
anisotropy `0.1` and direction `60` degrees. The actual model is built through
the maintained template and PyTomoATT; previews show HDF5 fields rather than an
illustrative pattern. Counts and amplitudes can be changed. The fixed dual-layer
anisotropy preset leaves baseline velocity unchanged.

## Run planning and generation

Preview run folders resolves the layout, phase grids, weighting/output choices,
and job counts without creating run/model artifacts. Generate run folders writes
a new artifact tree and manifest under `.tomogui/runs/`. Run generation omits
pick-weight columns by default unless retention is selected; custom quality
conversion enables retention explicitly. Original source/event weights retain
the existing helper's behavior.

### 1. Resolution Kd estimation

The current JpGU coefficient tuple and resolution sweep wavelength definitions
are read from the reused helpers. The preset is 8 coefficients across 3
sinusoidal wavelengths per phase (24 inversions and 3 signal tasks). Sinusoidal
helpers retain their 3% amplitude, azimuths, bottom buffer, seeds and wave weights.
Each coefficient at a given phase/wavelength shares one noisy forward dataset.
Slurm inversion tasks depend on the corresponding signal job and wait for a
nonempty noisy src_rec file.

After actual recovered models exist, Score completed models reuses the existing
windowed-resolution metric. It combines wavelength scores using their original
weights, checks matching physical grids, and applies the best complete weighted
score per phase when the whole phase sweep is complete. Complete alternatives
remain selectable. Partial results cannot silently become the final selection.

### 2. Checkerboard validation

The selected coefficient is applied to the separately chosen checkerboard or
dual-layer validation case. Inspect the recovered HDF5 model. Accept recovery
requires the output to exist and be newer than the coefficient selection.

### 3. Actual-data inversion

The GUI allows this stage only after coefficient selection and accepted
validation. The maintained joint-step iteration counts and relocation handoff
tags are preserved. A successful Slurm submission means submitted, not completed;
completion is determined from output models.

The optional extra analysis-fields checkbox enables `output_source_field`,
`output_kernel`, `output_in_process` and `output_in_process_data` in actual-data
inversion YAMLs via `--enable-azcov-outputs`. These support later coverage,
ray tracing and kernel/sensitivity analysis. It uses more disk space and I/O;
the off default is retained.

## Execution and troubleshooting

Local MPI and Slurm commands run in the environment hosting the Python server.
Set `TOMOATT_BIN` before launching it. Generated runners use the generating
Python interpreter; `TOMOGUI_PYTHON` can select another interpreter after moving
files. New bundled Slurm templates omit site-specific partition/account values.
See INSTALLATION.md for MPI compatibility, cluster modules and optional PyGMT.

Jobs are serialized, recorded persistently and cancellable in Jobs & logs.
A browser NetworkError means no usable response reached the browser. Confirm
the server is alive and use Reconnect. The client refreshes stale session tokens
and polls compact job summaries. It does not blindly retry a job-creation POST
after a network failure because the server may already have accepted it.

## HDF5 inspection and limits

The reused headless viewer normalizes native grids, including `(dep,lat,lon)`
local mask files, and checks physical grid compatibility in cross-file math.
The UI provides orthogonal slices, two-model comparison, masks, sampled 3D points
and PNG export. Map slices preserve geographic aspect. Embedded VTK/PyVista is
not used; no Qt installation is needed.

This alpha does not include all original desktop viewer features: oblique/four-
model layouts, direction-arrow overlays, depth videos, and all export variants
remain outside this browser UI. There is no general custom-shape editor, broad
region search UI, SSH/rsync profile manager or full QC gallery. The offline map
outline is Taiwan-specific; data elsewhere still have real coordinates.

Project/run manifests currently contain absolute paths; keep them at the original
location for GUI execution/scoring. Generated solver input paths are relative.
Optional postprocessing helpers can skip missing products or continue after
plot failures; inspect each log. This release's tests do not run production MPI
or HPC inversions.

## Developer checks

```bash
python -m pip install -e ".[workflow,dev]"
python -m unittest discover -s tests -v
python tests/serve_fixture.py
```

The optional Playwright `tests/browser.cjs` targets the isolated fixture server
on port 8767 and covers real background jobs, source/folder previews, dragging,
model rebuilding, grid tuning, viewer modes, PNG export and reconnect behavior.
Install Node.js and Playwright only for browser development checks. The other
opt-in acceptance scripts expect original research inputs and are not required
for installation or CI. Synthetic perfect-recovery fixtures used by unit tests
are explicitly test data, not claimed inversion outputs.
