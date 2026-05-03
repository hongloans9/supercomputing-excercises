"""
analysis_mpi.py — Find file pairs in super_data/ using MPI (mpi4py).

Each .npy file is fingerprinted with a SHA-256 hash of its raw bytes.
Two files sharing the same hash are a "pair" (byte-identical content).

Run on Puhti:
  srun --mpi=pmix_v3 python analysis_mpi.py [data_folder]
"""

import os
import sys
import hashlib
import numpy as np
from mpi4py import MPI
from collections import defaultdict

DATA_FOLDER = sys.argv[1] if len(sys.argv) > 1 else "/projappl/project_2018026/super_data"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    all_npy = sorted(f for f in os.listdir(DATA_FOLDER) if f.endswith(".npy"))
else:
    all_npy = None
all_npy = comm.bcast(all_npy, root=0)

N        = len(all_npy)
my_files = [all_npy[i] for i in range(N) if i % size == rank]

comm.Barrier()
t0 = MPI.Wtime()

local_results = []
for fname in my_files:
    arr    = np.load(os.path.join(DATA_FOLDER, fname))
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    local_results.append((fname, digest))

comm.Barrier()
elapsed = MPI.Wtime() - t0

all_results = comm.gather(local_results, root=0)

if rank == 0:
    results = [item for chunk in all_results for item in chunk]

    by_hash = defaultdict(list)
    for fname, digest in results:
        by_hash[digest].append(fname)

    pairs = {h: files for h, files in by_hash.items() if len(files) >= 2}

    print("=== ANALYSIS RESULT ===")
    print(f"Processed files : {N}")
    print(f"Pairs found     : {len(pairs)}")
    print(f"Time taken      : {elapsed:.2f} seconds")
    print(f"Time taken      : {elapsed / 60:.2f} minutes")

    if pairs:
        print("\nPaired files:")
        for h, files in pairs.items():
            print(f"  [{h[:12]}...]  {', '.join(files)}")
