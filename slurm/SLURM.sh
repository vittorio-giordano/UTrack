#!/usr/bin/env bash
#SBATCH --job-name=UTrack_Global
#SBATCH --partition=partition_name #specify the partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --threads-per-core=1
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@institution.edu
#SBATCH --output=/log/global_%j.log
#SBATCH --error=/log/global_%j.err

# SETUP 
echo "Job running on node: $SLURMD_NODENAME"
echo "Resources: $SLURM_CPUS_PER_TASK CPUs, $SLURM_MEM_PER_NODE GB RAM"

cd "$SLURM_SUBMIT_DIR"

mkdir -p log

module purge
#module load miniconda3 #change to your conda module version if needed
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate utrack_env # Activate the conda environment, change to your environment name

# Threading configuration for Numba and OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# EXECUTION
YEAR=2024
MONTH=05

CONFIG_NAME=${1:-"config_bwdmix24.yml"}
CONFIG_FILE="config/$CONFIG_NAME"
PYTHON_SCRIPT=${2:-"src/UTrack_SPINUP_tracking_indic_diagn_interp.py"}

echo "----------------------------------------------------------------"
echo "UTrack Global Simulation started"
echo "Script: $PYTHON_SCRIPT"
echo "Configuration: $CONFIG_FILE | Date: $YEAR-$MONTH"
echo "----------------------------------------------------------------"

$CONDA_PREFIX/bin/time -v $CONDA_PREFIX/bin/python $PYTHON_SCRIPT \
    --config $CONFIG_FILE \
    --year $YEAR \
    --month $MONTH

echo "----------------------------------------------------------------"
echo "Job completed."
echo "----------------------------------------------------------------"

echo -e "\n=== SLURM DIAGNOSTICS: QUEUE STATUS (sinfo) ===" >&2
sinfo >&2