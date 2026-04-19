#!/bin/bash
#SBATCH --account=project_2018026
#SBATCH --job-name=loanng_mpi_analysis
#SBATCH --output=/scratch/project_2018026/loanng/mpi_analysis_%j.out
#SBATCH --time=00:15:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=small

module load python-data
module load gcc/11.3.0 openmpi/4.1.4

VENV="$HOME/supercomputing-excercises/week6/.venv"
export PATH="$VENV/bin:$PATH"

srun --mpi=pmix_v3 "$VENV/bin/python" -u analysis_mpi.py
