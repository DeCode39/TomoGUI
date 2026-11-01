#!/bin/bash
#SBATCH -J tomoatt_inversion
#SBATCH -n 8
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH -o tomoatt_%j.out
#SBATCH -e tomoatt_%j.err

set -eo pipefail
ulimit -l unlimited

# Use the Python/MPI environment from which this job is submitted.

TOMOATT_BIN="${TOMOATT_BIN:-$HOME/TomoATT/build/bin/TOMOATT}"
NTASKS="${SLURM_NTASKS:-8}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

run_optional_python() {
  local script="$1"
  shift || true
  if [[ ! -f "$script" ]]; then
    echo "Skipping missing script: $script"
    return 0
  fi
  echo "Running python $script $*"
  if ! python "$script" "$@"; then
    echo "Warning: $script failed; continuing."
  fi
}

# Step 1: Generate necessary input files
python prepare_input_files.py



# Step 2: Do joint inversion
srun --mpi=pmix -n "${NTASKS}" "${TOMOATT_BIN}" -i 3_input_params/input_params_joint_step1.yaml
srun --mpi=pmix -n "${NTASKS}" "${TOMOATT_BIN}" -i 3_input_params/input_params_joint_step2.yaml
srun --mpi=pmix -n "${NTASKS}" "${TOMOATT_BIN}" -i 3_input_params/input_params_joint_step3.yaml

# Step 3 (Optional): Plot and run extra diagnostics
run_optional_python plot_output.py
run_optional_python run_postprocess.py
