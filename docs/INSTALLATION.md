# Installation on Linux and WSL

TomoATT is assumed to be installed and working. TomoGUI does not build or replace
the solver. Python libraries, Git, a package environment and optional plotting
tools are separate requirements.

## 1. Operating-system prerequisites

Supported target: Linux, including Ubuntu in WSL 2. Python 3.10 or newer is
required; Python 3.11 is the recommended scientific environment.

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip ca-certificates
python3 --version
```

If the OS ships Python older than 3.10, use a newer supported OS Python or create
the Conda environment below. Do not replace the system Python installation.
Binary wheels normally supply the NumPy/SciPy/HDF5 components. On an unsupported
architecture where pip must compile them, use Conda or install the compiler and
native development libraries required by those upstream packages.

For WSL, install Ubuntu using `wsl --install -d Ubuntu` in Administrator
PowerShell, restart if requested, and run all remaining commands in Ubuntu.
Keep source/projects in the Linux home directory when possible. Windows drives
are mounted as `/mnt/c`, `/mnt/e`, etc.; no drive letters are used in Linux paths.

## 2. Download a versioned source release

For private repository access using GitHub CLI:

```bash
sudo apt install -y gh
gh auth login
gh auth setup-git
git clone https://github.com/DeCode39/TomoGUI.git
cd TomoGUI
git switch --detach v0.1.0
```

GitHub SSH authentication or downloading a source archive from the release page
are alternatives. A private repository requires explicit access; installing
TomoATT does not grant access to TomoGUI. If you already have the checkout, run
`git fetch --tags` before selecting a release tag. Local changes should be
committed or saved before changing versions.

## 3. Install Python dependencies

### Option A: venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[workflow]"
python -m pip check
```

`[workflow]` installs PyTomoATT as well as TomoGUI's declared dependencies.
PyTomoATT pulls its own requirements, including pyproj, ruamel-yaml,
scikit-learn, tqdm and xarray. Run these commands in the venv; do not use
`sudo pip`, `--no-deps`, or a Windows `.venv` from WSL.

### Option B: Conda / Miniforge

If Conda is not installed, obtain an installer for your Linux architecture from
the official Miniforge release page:
https://github.com/conda-forge/miniforge/releases . Install it according to the
upstream instructions and initialize your shell. Then:

```bash
conda create -n tomogui -c conda-forge python=3.11 pip
conda activate tomogui
python -m pip install ".[workflow]"
python -m pip check
```

You can also activate the existing scientific environment used with TomoATT,
then install TomoGUI and its declared dependencies in it. Use a separate
environment if changing dependencies could affect another project.

## 4. Create a project and check the installation

```bash
tomogui --version
tomogui --init "$HOME/tomogui-projects/example"
export TOMOATT_BIN="/absolute/path/to/TomoATT/build/bin/TOMOATT"
test -x "$TOMOATT_BIN"
tomogui --workspace "$HOME/tomogui-projects/example" --check
tomogui --workspace "$HOME/tomogui-projects/example" --no-browser
```

Open http://127.0.0.1:8766 in a browser. For WSL this can be a normal Windows
browser. Keep the server terminal open; Ctrl+C stops the server. Activate the
same Python environment and set `TOMOATT_BIN` again in each new shell, or add
the export to your own shell configuration if appropriate.

Copy input data inside the new project's `data/` folder. Existing `src_rec`
files are accepted directly; SALUTE/CWB catalogs may also need a station file.
An event-location table without arrival picks cannot create travel-time data.
Use the source preview before importing.

`--check` reports the selected project, Python platform/executable, scientific
Python dependencies, TomoATT path, Bash, MPI, and Slurm availability. Its readiness
flags distinguish preparation from solver execution. A missing solver or Slurm
does not prevent preparation or viewing; missing workflow Python dependencies
produce a nonzero check exit status. The check does not execute the solver or
prove MPI binary compatibility.

## 5. MPI and Slurm

### Local MPI

Use the **same MPI implementation and compatible libraries used to build
TomoATT**. Do not install another implementation blindly into the environment.
On a compatible OpenMPI-based Ubuntu installation, the runtime package is
`openmpi-bin`; MPICH-based builds require their matching MPICH runtime instead.
Your cluster's module system may supply both the solver and MPI.

```bash
command -v mpirun
mpirun --version
test -x "$TOMOATT_BIN"
```

TomoGUI's local runner uses `mpirun` by default. `MPI_LAUNCHER` overrides the
launcher; `SLURM_NTASKS` can override generated task counts. Do not launch an
MPI inversion just to test a GUI installation: preparation and viewing work
without a running solver.

### Slurm

Slurm is required only for cluster submission, not on a workstation used to
prepare folders. Launch TomoGUI where `sbatch` and `srun` are available when
using its Slurm buttons. Packaged templates contain no site-specific account or
partition. Review/edit `run_tomogui_slurm.sh` for your account, partition, time,
memory and task limits before submission. Existing legacy workspaces retain
their own template directives.

Generated jobs inherit the submitting environment and use the Python executable
that generated them. Set `TOMOGUI_PYTHON` to the target cluster's Python when
moving run folders between machines. Activate/load the required MPI modules
before submission. `SLURM_MPI_TYPE` overrides the default `pmix` mode; use a
plugin supported by your cluster (`srun --mpi=list`). Never add `set -u` ahead of
Conda activation hooks.

The run manifest currently stores absolute paths, so the GUI project must remain
at its original location for execution/scoring. Generated solver input paths are
relative within the staged layout. Cross-machine GUI manifest migration is not
implemented in v0.1.0.

## Optional legacy plotting tools

The integrated viewer and Matplotlib QC do not require GMT or Qt. Some copied
`plot_output.py` helpers use PyGMT and the GMT shared library. To use them, a
Conda environment is the most direct installation route:

```bash
conda create -n tomogui-plots -c conda-forge python=3.11 pip pygmt
conda activate tomogui-plots
python -m pip install ".[workflow]"
python -c "import pygmt; pygmt.show_versions()"
```

Install FFmpeg with your package manager only if using video-producing
postprocessing helpers. Cartopy/Shapely support optional geographic plotting;
they are not required by the browser's bundled Taiwan coastline. Some optional
plotting features download their own geographic assets.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `tomogui: command not found` | Activate the venv/Conda environment; try `python -m tomogui --version` |
| `externally-managed-environment` from pip | Create and activate a venv; do not install into the OS Python |
| Missing PyTomoATT or scientific modules | Run `python -m pip install ".[workflow]"` and `python -m pip check` |
| Existing directory rejected by `--init` | Choose a new directory; initialization never overwrites a project |
| Missing Scripts/templates in `--workspace` | First run `--init NEW_PROJECT`, or select a full legacy workspace |
| Windows `WinError 1920` on a TomoATT link | Run Python inside WSL; use Linux `/mnt/...` paths |
| Browser NetworkError | Check the server terminal, its port, and WSL; use Reconnect or refresh |
| Address already in use | Stop the previous server or use `--port 8767` and open that port |
| pyproj / PROJ database warning | Remove stale `PROJ_LIB`/`PROJ_DATA` overrides; reinstall pyproj in the active environment if needed |
| Slurm command missing | Use preparation/local mode, or run on a cluster login node with Slurm tools |
| No region passes filters | Inspect pick minima, selected phases, central area and all-azimuth-bin requirement |
| GMT import/library error | Install the optional GMT/PyGMT environment; core GUI viewing still works |

For a remote Linux server, keep the loopback binding and forward the port from
your own machine: `ssh -L 8766:127.0.0.1:8766 user@server`. Start TomoGUI with
`--no-browser` on that server, then open the forwarded local address. Do not
expose this local file application as an unauthenticated public web service.

Upstream references: [PyTomoATT installation](https://tomoatt.github.io/PyTomoATT/installation.html),
[TomoATT](https://github.com/TomoATT/TomoATT),
[Miniforge](https://github.com/conda-forge/miniforge).
