#!/bin/bash
#SBATCH --account=project_2018026
#SBATCH --job-name=loanng_location_test
#SBATCH --output=/scratch/project_2018026/loanng/location_test_%j.out
#SBATCH --time=00:30:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=small

module load python-data
module load gcc/11.3.0 openmpi/4.1.4

VENV="$HOME/supercomputing-excercises/week6/.venv"
export PATH="$VENV/bin:$PATH"
PYTHON="$VENV/bin/python"

PROJAPPL_DATA="/projappl/project_2018026/super_data"
SCRATCH_DATA="/scratch/project_2018026/loanng/super_data"
LOCAL_DATA="$LOCAL_SCRATCH/super_data"

echo "============================================================"
echo "  DATA LOCATION COMPARISON TEST"
echo "  Job ID: $SLURM_JOB_ID"
echo "  MPI ranks: $SLURM_NTASKS"
echo "============================================================"

# ---- Test 1: projappl (parallel filesystem, persistent) ----
echo ""
echo ">>> TEST 1: Data on projappl"
srun --mpi=pmix_v3 "$PYTHON" -u analysis_mpi.py "$PROJAPPL_DATA"

# ---- Test 2: scratch (Lustre parallel filesystem, fast I/O) ----
echo ""
echo ">>> TEST 2: Data on scratch"
if [ ! -d "$SCRATCH_DATA" ]; then
    echo "Copying data to scratch..."
    mkdir -p "$SCRATCH_DATA"
    cp "$PROJAPPL_DATA"/*.npy "$SCRATCH_DATA"/
fi
srun --mpi=pmix_v3 "$PYTHON" -u analysis_mpi.py "$SCRATCH_DATA"

# ---- Test 3: local NVMe ($LOCAL_SCRATCH on compute node) ----
echo ""
echo ">>> TEST 3: Data on LOCAL_SCRATCH (NVMe)"
mkdir -p "$LOCAL_DATA"
cp "$PROJAPPL_DATA"/*.npy "$LOCAL_DATA"/
srun --mpi=pmix_v3 "$PYTHON" -u analysis_mpi.py "$LOCAL_DATA"

echo ""
echo "============================================================"
echo "  ALL TESTS COMPLETE"
echo "============================================================"
