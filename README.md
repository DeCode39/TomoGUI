# TomoGUI

A clean, local browser interface for preparing and inspecting TomoATT projects.
TomoGUI runs Python on Linux or WSL and opens at **http://127.0.0.1:8766**.
The TomoATT solver is a separate installation.

**Status:** v0.1.0, alpha. Releases use versioned Git tags and GitHub release
assets. Nothing is published to PyPI by this project.

## What you can do

- Inspect a source file before preparation. Recognize absolute `src_rec`, SALUTE,
  CWB and ISC-style arrival tables automatically.
- Map numeric quality, formulas, or alphabetical grades to pick weights.
- Drag region rectangle corners or edit geographic/depth bounds and filters.
- Preview actual checkerboard HDF5 models and change their counts/amplitudes.
- Tune model/inversion grid estimates while retaining the workflow defaults.
- Preview run folders before writing them. Follow resolution Kd estimation,
  checkerboard validation, then actual-data inversion.
- Explore HDF5 slices, two-model comparisons, masks, cross-file expressions,
  sampled 3D volumes, and PNG exports with the reused headless HDF5 viewer.

Required preparation scripts, model helpers and data-free templates are included.
**You do not need the original research workspace.** Research catalogs, model
volumes, solver binaries and inversion results are not included.

## Requirements

| Requirement | Needed for | How it is supplied |
| --- | --- | --- |
| Linux or Ubuntu on WSL 2; Bash | Server and generated runners | OS; WSL instructions below |
| Python 3.10+ (3.11 recommended), pip, venv | Application | Install with your OS package manager or Conda |
| Git | Installing from a release tag | OS package manager |
| NumPy, SciPy, pandas, h5py, PyYAML, Matplotlib | Preparation and viewer | Installed by pip with TomoGUI |
| PyTomoATT 0.2.8 or 0.3.x | Region/model/run preparation | Installed by the `[workflow]` extra |
| Web browser | Interface | Windows browser works with a WSL server |
| Your existing TomoATT executable and matching MPI runtime | Actual local solver execution | Existing TomoATT installation; set `TOMOATT_BIN` |
| Slurm `sbatch`/`srun` | Cluster submissions only | Cluster installation |
| GMT + PyGMT | Legacy `plot_output.py` plots only | Optional; see installation guide |

PyTomoATT and all its declared Python dependencies are installed by the commands
below. A TomoATT solver installation alone does not supply these Python packages.
Qt, Node.js, VTK, PyVista, Docker and a GPU are not required for normal use.

## Install on Linux

Ubuntu/Debian example (use your distribution's equivalent package commands):

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip ca-certificates
python3 --version  # must be 3.10 or newer
```

Authenticate Git first if the repository is private. For example, install `gh`
with your OS package manager, run `gh auth login`, then `gh auth setup-git`.
Alternatively, use an SSH clone with your existing GitHub SSH key.

```bash
git clone https://github.com/DeCode39/TomoGUI.git
cd TomoGUI
git switch --detach v0.1.0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[workflow]"
tomogui --version
```

Use a new project directory, outside the source checkout:

```bash
tomogui --init "$HOME/tomogui-projects/example"
# Set this to your already-installed executable before launching the server:
export TOMOATT_BIN="/path/to/TomoATT/build/bin/TOMOATT"
test -x "$TOMOATT_BIN"
tomogui --workspace "$HOME/tomogui-projects/example" --check
tomogui --workspace "$HOME/tomogui-projects/example"
```

Copy your arrival files and station metadata into the project's `data/` folder,
then choose them in the Data tab. File browsing is restricted to the selected
project. `--init` refuses to overwrite an existing directory.

## Install on Windows with WSL 2

Run the following in **Administrator PowerShell** if Ubuntu WSL is not installed:

```powershell
wsl --install -d Ubuntu
```

Restart Windows if prompted, open Ubuntu, and create your Linux user. Run the
Linux installation commands above **inside Ubuntu**, including the Python venv
and pip installation. A Windows Python environment cannot be reused in WSL.
For better filesystem performance, keep the checkout and new projects under
`$HOME` in Linux. Existing Windows data is available under `/mnt/c`, `/mnt/e`, etc.

Launch the server from Ubuntu:

```bash
cd "$HOME/TomoGUI"  # or your actual clone location
source .venv/bin/activate
export TOMOATT_BIN="/path/to/your/Linux/TOMOATT"
tomogui --workspace "$HOME/tomogui-projects/example" --no-browser
```

Open **http://127.0.0.1:8766** in your Windows browser. Leave the Ubuntu process
running. The Environment dialog should show WSL and a Linux Python executable.
Linux symlinks to TomoATT cannot be followed by native Windows Python.

See [the complete installation guide](docs/INSTALLATION.md) for Conda, optional
plotting dependencies, MPI/Slurm requirements, verification and troubleshooting.

## Install a release asset instead of cloning

Download the wheel from the matching [GitHub release](https://github.com/DeCode39/TomoGUI/releases)
and activate a Python venv, then run:

```bash
python -m pip install './tomogui-0.1.0-py3-none-any.whl[workflow]'
tomogui --init "$HOME/tomogui-projects/example"
```

For a source archive, extract it, enter its top-level directory, and use
`python -m pip install ".[workflow]"`. Dependencies are installed normally; do
not use `--no-deps` for a new installation. Verify downloaded assets against the
release's `SHA256SUMS` file with `sha256sum -c SHA256SUMS` (download both archives
if checking the entire checksums file).

## Scientific defaults and run sequence

The bundled defaults preserve the maintained research helpers:

- Checkerboard counts `[2, 2, 2]`, velocity amplitude `0.2`, anisotropy `0.1`,
  direction `60` degrees; the existing dual-layer anisotropy preset keeps
  velocity unchanged.
- Model-grid lateral scale `1`, depth spacing `1 km`, inversion counts at `75%`.
  Phase-specific final src_rec spacing is used; controls allow explicit tuning.
- Import preserves weights. Run generation omits receiver/pick weight columns
  unless selected; a custom quality conversion explicitly enables retention.
  Event-weight semantics remain unchanged.
- Kd estimation automatically stages 3 sinusoidal wavelengths and 8 coefficients
  per phase: 24 inversion cases plus 3 shared signal/noise calculations. The
  original 3% sinusoidal model settings, wave weights and seeds are reused.
- Scoring applies the highest complete weighted score per phase. Review
  checkerboard recovery and accept it before launching actual-data inversion.
  No coefficient is selected from missing or fabricated solver outputs.

New project artifacts and logs go in `.tomogui/`; original inputs are not edited.
Run-folder previews create no run artifacts. Generated runners use the current
Python environment, with `TOMOGUI_PYTHON` available when moving runs to another
machine. Inspect generated Slurm resource requests before submitting to a cluster.

The extra analysis outputs option enables wavefields, kernels and intermediate
inversion data for later azimuth coverage, ray tracing and sensitivity analysis.
It increases storage and I/O and remains off by default.

See [workflow details](docs/WORKFLOW.md) for weighting, grids, execution, outputs
and current limitations. Existing workflow workspaces can still be opened with
`tomogui --workspace /path/to/workspace`; their existing helpers are used.

## Releases and development

Use a release tag for reproducible installs; `main` contains development changes.
See [release instructions](release/README.md) and [the changelog](CHANGELOG.md).
Upgrades do not overwrite helpers in existing projects; create a new project for
new bundled template versions and explicitly move/copy your data as needed.

```bash
python -m pip install -e ".[workflow,dev]"
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
```

Tests initialize isolated projects from packaged resources. No original research
workspace or TomoATT solver execution is needed. Optional browser checks are
documented in `docs/WORKFLOW.md`. Production MPI/HPC inversions are not covered
by this first release's validation.

## License and upstream projects

GPL version 3; see [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
The reused workflow file provenance is included in the package.

- [TomoATT](https://github.com/TomoATT/TomoATT)
- [PyTomoATT installation](https://tomoatt.github.io/PyTomoATT/installation.html)
