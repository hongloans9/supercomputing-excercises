"""
analysis_mpi.py — Heavy numerical analysis on .npy files using MPI (mpi4py).
Run on the supercomputer with:  srun python analysis_mpi.py [data_folder]

Each MPI rank processes a subset of the files; rank 0 gathers
and prints the final report.
"""
import os
import sys
import time
import numpy as np
from mpi4py import MPI

DATA_FOLDER = sys.argv[1] if len(sys.argv) > 1 else "/projappl/project_2018026/super_data"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def analyze_file(fname):
    """Run a computationally heavy analysis pipeline on one .npy file."""
    path = os.path.join(DATA_FOLDER, fname)
    arr = np.load(path)

    # --- 1) FFT & power spectrum ---
    fft = np.fft.fft2(arr)
    power = np.abs(fft) ** 2
    mean_power = power.mean()

    # --- 2) SVD on 1000x1000 sub-blocks ---
    n = arr.shape[0] // 2
    all_S = []
    for r in range(2):
        for c in range(2):
            block = arr[r*n:(r+1)*n, c*n:(c+1)*n]
            U, S, Vt = np.linalg.svd(block, full_matrices=False)
            all_S.append(S)
    S_combined = np.sort(np.concatenate(all_S))[::-1]

    # --- 3) SVD compression error estimate ---
    block = arr[:n, :n]
    U, S, Vt = np.linalg.svd(block, full_matrices=False)
    compression = {}
    for k in [10, 50]:
        recon = (U[:, :k] * S[:k]) @ Vt[:k, :]
        compression[k] = np.linalg.norm(block - recon) / np.linalg.norm(block)

    # --- 4) Eigenvalue decomposition of row-covariance ---
    cov_rows = arr @ arr.T / arr.shape[1]
    eig_rows = np.linalg.eigvalsh(cov_rows)

    # --- 5) Matrix norms ---
    fro_norm = np.linalg.norm(arr, "fro")
    nuc_norm = np.sum(S_combined)

    # --- 6) Sorting + percentiles ---
    sorted_flat = np.sort(arr, axis=None)
    percs = np.percentile(arr, list(range(0, 101)))
    iqr = percs[75] - percs[25]

    # --- 7) Histogram + Shannon entropy ---
    hist_vals, bin_edges = np.histogram(arr, bins=1000)
    probs = hist_vals / hist_vals.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))

    # --- 8) Basic statistics ---
    max_val = arr.max()
    min_val = arr.min()
    mean_val = arr.mean()
    std_val = arr.std()
    skewness = np.mean(((arr - mean_val) / std_val) ** 3)
    kurtosis = np.mean(((arr - mean_val) / std_val) ** 4) - 3

    return {
        "file": fname,
        "max": float(max_val),
        "min": float(min_val),
        "mean": float(mean_val),
        "std": float(std_val),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "entropy": float(entropy),
        "iqr": float(iqr),
        "mean_power": float(mean_power),
        "fro_norm": float(fro_norm),
        "nuc_norm": float(nuc_norm),
        "top_singular": float(S_combined[0]),
        "rank50_err": float(compression[50]),
        "eig_row_max": float(eig_rows[-1]),
    }


# --- Distribute files across ranks ---
if rank == 0:
    all_npy = sorted(f for f in os.listdir(DATA_FOLDER) if f.endswith(".npy"))
    # Split file list into `size` roughly-equal chunks
    chunks = [[] for _ in range(size)]
    for i, f in enumerate(all_npy):
        chunks[i % size].append(f)
else:
    chunks = None

my_files = comm.scatter(chunks, root=0)

if rank == 0:
    print(f"Total files: {len(all_npy)}, MPI ranks: {size}")
    print(f"Rank 0 processing {len(my_files)} files\n")

comm.Barrier()
t0 = MPI.Wtime()

# --- Each rank analyses its share ---
local_results = []
for fname in my_files:
    result = analyze_file(fname)
    local_results.append(result)

comm.Barrier()
elapsed = MPI.Wtime() - t0

# --- Gather all results to rank 0 ---
all_results = comm.gather(local_results, root=0)

if rank == 0:
    # Flatten the list of lists
    results = []
    for chunk in all_results:
        results.extend(chunk)
    results.sort(key=lambda r: r["file"])

    n = len(results)
    print("=" * 100)
    print("  ANALYSIS REPORT — super_data (MPI)")
    print("=" * 100)
    print(f"{'File':<22} | {'Max':>8} | {'Min':>8} | {'Mean':>8} | "
          f"{'Std':>8} | {'Entropy':>8} | {'TopSV':>10} | {'Rank50Err':>10}")
    print("-" * 100)

    for r in results:
        print(f"{r['file']:<22} | {r['max']:>8.4f} | {r['min']:>8.4f} | "
              f"{r['mean']:>8.4f} | {r['std']:>8.4f} | {r['entropy']:>8.4f} | "
              f"{r['top_singular']:>10.2f} | {r['rank50_err']:>10.6f}")

    print("-" * 100)
    print(f"\nSummary across {n} files:")
    print(f"  Sum of max values    : {sum(r['max'] for r in results):.4f}")
    print(f"  Sum of min values    : {sum(r['min'] for r in results):.4f}")
    print(f"  Average mean         : {sum(r['mean'] for r in results) / n:.6f}")
    print(f"  Average std          : {sum(r['std'] for r in results) / n:.6f}")
    print(f"  Average entropy      : {sum(r['entropy'] for r in results) / n:.4f} bits")
    print(f"  Average skewness     : {sum(r['skewness'] for r in results) / n:.6f}")
    print(f"  Average kurtosis     : {sum(r['kurtosis'] for r in results) / n:.6f}")
    print(f"  Brightest file       : {max(results, key=lambda r: r['max'])['file']}")
    print(f"  Darkest file         : {min(results, key=lambda r: r['min'])['file']}")
    print(f"  Highest entropy      : {max(results, key=lambda r: r['entropy'])['file']}")
    print(f"  Largest top SV       : {max(results, key=lambda r: r['top_singular'])['file']}")

    print(f"\n  MPI ranks used       : {size}")
    print(f"  Total elapsed time   : {elapsed:.2f} seconds")
    print("=" * 100)
