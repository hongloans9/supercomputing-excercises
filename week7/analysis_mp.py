"""
analysis_mp.py — Find file pairs in super_data/ using multiprocessing.

Each .npy file is reduced to a single summary value (array sum).
Two files with the same sum are a "pair".

Uses multiprocessing.Pool to process all files in parallel across
available CPU cores.
"""

import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from collections import defaultdict

DATA_FOLDER = "super_data"
N_WORKERS   = cpu_count()


def file_sum(fname):
    arr = np.load(os.path.join(DATA_FOLDER, fname))
    return fname, float(arr.sum())


if __name__ == "__main__":
    all_npy = sorted(f for f in os.listdir(DATA_FOLDER) if f.endswith(".npy"))
    N = len(all_npy)

    t0 = time.perf_counter()

    with Pool(N_WORKERS) as pool:
        results = pool.map(file_sum, all_npy)

    elapsed = time.perf_counter() - t0

    by_sum = defaultdict(list)
    for fname, s in results:
        by_sum[s].append(fname)

    pairs = {s: files for s, files in by_sum.items() if len(files) >= 2}

    print("=== ANALYSIS RESULT ===")
    print(f"Processed files : {N}")
    print(f"Pairs found     : {len(pairs)}")
    print(f"Time taken      : {elapsed:.2f} seconds")
    print(f"Time taken      : {elapsed / 60:.2f} minutes")

    if pairs:
        print("\nPaired files:")
        for s, files in pairs.items():
            print(f"  sum={s:.6f} → {', '.join(files)}")
