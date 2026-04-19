"""
analysis_mp.py — Heavy numerical analysis on .npy files in super_data/.
Uses Python multiprocessing to parallelise across CPU cores.

Restrict NumPy internal threading so that multiprocessing provides
the parallelism (avoids thread contention).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import numpy as np
from multiprocessing import Pool, cpu_count


DATA_FOLDER = "super_data"
N_WORKERS = cpu_count()


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

    # --- 3) SVD compression error estimate (from largest block) ---
    U, S, Vt = all_S[0], all_S[0], all_S[0]  # reuse last block
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

    # --- 9) Sorting + percentiles ---
    sorted_flat = np.sort(arr, axis=None)
    percs = np.percentile(arr, list(range(0, 101)))
    iqr = percs[75] - percs[25]

    # --- 10) Histogram + Shannon entropy ---
    hist_vals, bin_edges = np.histogram(arr, bins=1000)
    probs = hist_vals / hist_vals.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))

    # --- 11) Basic statistics ---
    max_val = arr.max()
    min_val = arr.min()
    mean_val = arr.mean()
    std_val = arr.std()
    skewness = np.mean(((arr - mean_val) / std_val) ** 3)
    kurtosis = np.mean(((arr - mean_val) / std_val) ** 4) - 3

    return {
        "file": fname,
        "shape": arr.shape,
        "max": max_val,
        "min": min_val,
        "mean": mean_val,
        "std": std_val,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "entropy": entropy,
        "iqr": iqr,
        "mean_power": mean_power,
        "fro_norm": fro_norm,
        "nuc_norm": nuc_norm,
        "top_singular": S[0],
        "rank50_err": compression[50],
        "eig_row_max": eig_rows[-1],
    }


def print_report(results, elapsed):
    """Print a formatted summary of the analysis."""
    print("=" * 100)
    print("  ANALYSIS REPORT — super_data")
    print("=" * 100)
    print(f"{'File':<22} | {'Max':>8} | {'Min':>8} | {'Mean':>8} | "
          f"{'Std':>8} | {'Entropy':>8} | {'TopSV':>10} | {'Rank50Err':>10}")
    print("-" * 100)

    for r in results:
        print(f"{r['file']:<22} | {r['max']:>8.4f} | {r['min']:>8.4f} | "
              f"{r['mean']:>8.4f} | {r['std']:>8.4f} | {r['entropy']:>8.4f} | "
              f"{r['top_singular']:>10.2f} | {r['rank50_err']:>10.6f}")

    print("-" * 100)
    n = len(results)
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

    print(f"\n  Workers used         : {N_WORKERS}")
    print(f"  Total elapsed time   : {elapsed:.2f} seconds")
    print("=" * 100)


if __name__ == "__main__":
    all_npy = sorted(f for f in os.listdir(DATA_FOLDER) if f.endswith(".npy"))
    print(f"Found {len(all_npy)} .npy files in {DATA_FOLDER}/")
    print(f"Using {N_WORKERS} worker processes\n")

    t0 = time.time()

    with Pool(N_WORKERS) as pool:
        results = pool.map(analyze_file, all_npy)

    elapsed = time.time() - t0
    print_report(results, elapsed)
