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

Checkerboards remain smooth products of three sinusoids. The preview defaults
are `[2,2,2]`, velocity perturbation `0.2`, anisotropy `0.1` and direction `60`
degrees. Counts mean signed sinusoidal lobes along each axis. Zero crossings
align with the active grid endpoints; depth perturbations exclude air cells.
There are no blockwise-constant anomalies. Previews show actual HDF5 fields.

Each lobe spans at least two fine model-grid **cells** on each axis (2 x 2 x 2
minimum volume), and at most half the corresponding full grid-domain span.
Cells are intervals between nodes: 8 checkers need at least 16 active intervals,
or 17 nodes per axis. Counts below 2 are rejected. Invalid presets are disabled
in the preview, and generation checks all default validation sets: 8, 4 and 2
checkers per axis. Refine the model grid if the 8-checker set does not fit.
Inversion-grid scale does not change these fine-grid sizing limits.

The fixed dual-layer anisotropy preset leaves baseline velocity unchanged.

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

Nominal sinusoidal lobe widths must also meet the two-cell/half-span limits.
The default wavelengths are unchanged. If they cannot fit, generation stops;
adjust the grid/domain or explicitly enable recorded wavelength fitting.
Fitting applies a common horizontal and a common depth scale to the three
original wavelength sets, preserving their ratios, amplitudes, seeds and wave
weights. Both original and effective values are stored in the run manifest.
If no common scale fits, refine the grid. Fitting never changes the sinusoidal
shape into constant blocks.

After actual recovered models exist, Score completed models reuses the existing
windowed-resolution metric. It combines wavelength scores using their original
weights, checks matching physical grids, and applies the best complete weighted
score per phase when the whole phase sweep is complete. Complete alternatives
remain selectable. Partial results cannot silently become the final selection.

### 2. Checkerboard validation

The selected coefficient is applied to all three sinusoidal checkerboard sets
(8/4/2) and any additional custom-count or dual-layer validation case. Inspect
the recovered HDF5 models. Accept recovery requires every output to exist and
be newer than the coefficient selection.

### 3. Actual-data inversion

The GUI allows this stage only after coefficient selection and accepted
validation. The maintained joint-step iteration counts and relocation handoff
tags are preserved. A successful Slurm submission means submitted, not completed;
completion is determined from output models.

The baseline is explicitly ABS-only in both model-update and relocation flags.
It runs relocation -> joint inversion -> relocation. CS/CR refinement is a
separate experiment; the original absolute run remains available for comparison.

The optional extra analysis-fields checkbox enables `output_source_field`,
`output_kernel`, `output_in_process` and `output_in_process_data` in actual-data
inversion YAMLs via `--enable-azcov-outputs`. These support later coverage,
ray tracing and kernel/sensitivity analysis. It uses more disk space and I/O;
the off default is retained.

### 4. Optional CS/CR refinement and new resolution tests

Complete the absolute-only baseline, then enable CS/CR refinement. Preview
eligible pairs before staging. The initial model is baseline step 2's
`final_model.h5`, and the starting data are baseline step 3's final `_obs.dat`.
Observed files include origin-time-adjusted absolute observations; synthetic
predictions are never substituted for observations. Persistent event IDs match
the original filtered input, saved before baseline weight columns are omitted.

| Selection | Default | Editable control |
| --- | --- | --- |
| CR quality | Both picks have saved pick weight 1.0 | Highest-quality pick weight |
| CR source separation | Strictly <10 km in 3D | CR 3D source separation |
| CS station azimuth gap at source | Circular difference strictly <30 degrees | CS azimuth gap |
| CS event displacement | Original to final baseline position strictly <3 km in 3D | CS baseline displacement |

The Tibet study supplies the 10 km and 30 degree defaults:
[Zhang et al., 2025, section 2, p. 704](https://jingchen.netlify.app/publication/2025zhangxuezhen_pwave/2025zhangxuezhen_pwave.pdf).
Using a 3D CR distance and the adjustable 3 km baseline displacement threshold
are this workflow's explicit choices. Distance is
`hypot(great-circle epicentral separation, depth difference)` in kilometres.
CS uses PyTomoATT's pair-generation function with updated station azimuths;
CR uses a spatial index and an exact 3D filter because the available PyTomoATT
CR distance option is horizontal. Pairs always share the phase. CR shares the
station, and CS shares the source.

CR quality comes from the saved filtered pick weights, independent of the
baseline's weight-retention setting. Missing grades are excluded unless
explicitly treated as top quality. Both quality selection and any numerical
grade-to-weight conversion should be reviewed for the dataset. Differential
weights average the two current absolute pick weights; event weights retain
their existing meanings. All absolute rows are retained. Pair limits bound
preview/generation cost; exceeding the budget stops the job.

| Stage | Source file contents | Active use |
| --- | --- | --- |
| 1: relocation | ABS + newly generated CR | Relocation uses ABS+CR |
| 2: joint inversion | ABS + regenerated CR + CS | Model update uses ABS+CS; relocation uses ABS+CR |
| 3: relocation | ABS + regenerated CR | Relocation uses ABS+CR |

The files are `1_src_rec_files/step1.dat`, `step2.dat`, `step3.dat`, with
`pair_report_step*.json` recording counts, settings and observed-input hashes.
At each boundary pairs are rebuilt from the preceding completed `_obs.dat`;
old differential rows are discarded. CR geometry and CS azimuths use the latest
catalogue. CS event eligibility is frozen from the original-to-final baseline
displacement, providing a consistent selection throughout the experiment.
Unchecking CS or CR removes that family from pair generation and solver flags.
Pairs are regenerated at the three external stage boundaries, not every inner
iteration of the joint solver.

Prepare also stages a new resolution suite around the baseline final model:
8/4/2 sinusoidal perturbations, each with one ABS control and one CS/CR branch.
The two branches share the same noisy **absolute** forward dataset; differences
are derived afterward. All six cases per phase run the full three-stage chain.
Noise seeds are fixed, and perturbations are relative to the baseline velocity
and anisotropy, including existing nonzero xi/eta. The selected Kd is reused
for this comparison; a new coefficient sweep is not silently performed.

Run paired resolution, score all completed cases, inspect velocity/anisotropy
recovery and masks in the viewer, then accept paired recovery. Acceptance is a
human resolution judgement, not an automatic score threshold. Only then can
the actual CS/CR refinement execute. Inputs, completion records and scored
outputs are fingerprinted; changes require restaging or rescoring/acceptance.
Changing a condition in the form requires preparing a new experiment. Previous
experiment folders are retained. A new selected baseline Kd invalidates the
attached differential experiment, as does restarting the absolute baseline.
Restarting paired resolution clears its previous scores and acceptance.
Slurm branches depend on their shared signal
job. A submission alone cannot satisfy the completion/acceptance gates.

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
location for GUI execution/scoring. Differential suites share some inputs using
absolute paths; keep the complete staged tree at that location on the execution
machine. Stage on the cluster, or use the same mounted path, for Slurm execution.
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
browser check, `tests/browser_differential.cjs`, runs after `browser.cjs` against
the same isolated fixture. It creates clearly labelled fixture solver outputs
and tests CS/CR previews, staging, settings persistence and the resolution gate.
Set `TOMOGUI_PYTHON` if Python is not on PATH. No production workspace is used.
The other
opt-in acceptance scripts expect original research inputs and are not required
for installation or CI. Synthetic perfect-recovery fixtures used by unit tests
are explicitly test data, not claimed inversion outputs.
