#!/bin/bash
#SBATCH --account=project_2018026
#SBATCH --job-name=loanng_pairs
#SBATCH --output=/scratch/project_2018026/loanng/pairs_%j.out
#SBATCH --time=00:10:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=small

module load python-data
module load gcc/11.3.0 openmpi/4.1.4

VENV="$HOME/supercomputing-excercises/week7/.venv"
export PATH="$VENV/bin:$PATH"

echo "=== Job ${SLURM_JOB_ID} | ranks=${SLURM_NTASKS} | $(date) ==="

srun --mpi=pmix_v3 "$VENV/bin/python" -u analysis_mpi.py \
    /projappl/project_2018026/super_data

echo "=== Done $(date) ==="
