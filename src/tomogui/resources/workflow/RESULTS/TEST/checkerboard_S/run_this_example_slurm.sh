#!/bin/bash
#SBATCH -J tomoatt_eg1
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

# Step 1: Generate necessary input files
python prepare_input_files.py --n-pert 2,2,2

# Step 2: Run forward modeling
srun --mpi=pmix -n "${NTASKS}" "${TOMOATT_BIN}" -i 3_input_params/input_params_signal.yaml

# Step 3: Assign data noise to the observational data
python assign_gaussian_noise.py

# Step 4: Do inversion
srun --mpi=pmix -n "${NTASKS}" "${TOMOATT_BIN}" -i 3_input_params/input_params_inv.yaml

# Step 5 (Optional): Plot the results
python plot_output.py

mv 2_models 2_models_ckb_222
mv img img_222_gd
mv OUTPUT_FILES OUTPUT_FILES_222_gd

# Repeat for 4 x 4 x 4
# Step 1: Generate necessary input files
python prepare_input_files.py --n-pert 4,4,4

# Step 2: Run forward modeling
srun --mpi=pmix -n "${NTASKS}" "${TOMOATT_BIN}" -i 3_input_params/input_params_signal.yaml

# Step 3: Assign data noise to the observational data
python assign_gaussian_noise.py

# Step 4: Do inversion
srun --mpi=pmix -n "${NTASKS}" "${TOMOATT_BIN}" -i 3_input_params/input_params_inv.yaml

# Step 5 (Optional): Plot the results
python plot_output.py

mv 2_models 2_models_ckb_444
mv img img_444_gd
mv OUTPUT_FILES OUTPUT_FILES_444_gd

# Repeat for 8 x 8 x 8
# Step 1: Generate necessary input files
python prepare_input_files.py --n-pert 8,8,8

# Step 2: Run forward modeling
srun --mpi=pmix -n "${NTASKS}" "${TOMOATT_BIN}" -i 3_input_params/input_params_signal.yaml

# Step 3: Assign data noise to the observational data
python assign_gaussian_noise.py

# Step 4: Do inversion
srun --mpi=pmix -n "${NTASKS}" "${TOMOATT_BIN}" -i 3_input_params/input_params_inv.yaml

# Step 5 (Optional): Plot the results
python plot_output.py

mv 2_models 2_models_ckb_888
mv img img_888_gd
mv OUTPUT_FILES OUTPUT_FILES_888_gd
