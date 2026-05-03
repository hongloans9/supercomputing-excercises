import os
import numpy as np
import time

DATA_FOLDER = "super_data"
SAMPLE_SIZE = 50_000

all_npy = sorted(f for f in os.listdir(DATA_FOLDER) if f.endswith(".npy"))
N = len(all_npy)
print(f"Files: {N}, sample size: {SAMPLE_SIZE}\n")

print("Loading samples...")
t0 = time.perf_counter()
rng = np.random.default_rng(seed=0)
first = np.load(os.path.join(DATA_FOLDER, all_npy[0])).ravel()
idx = rng.choice(len(first), SAMPLE_SIZE, replace=False)

samples = np.empty((N, SAMPLE_SIZE), dtype=np.float32)
for i, fname in enumerate(all_npy):
    arr = np.load(os.path.join(DATA_FOLDER, fname)).ravel()
    samples[i] = arr[idx].astype(np.float32)
print(f"  done in {time.perf_counter()-t0:.1f}s\n")

# Full correlation matrix via BLAS
mat = samples - samples.mean(axis=1, keepdims=True)
mat /= (mat.std(axis=1, keepdims=True) + 1e-8)
corr = (mat @ mat.T) / SAMPLE_SIZE

rows, cols = np.triu_indices(N, k=1)
all_r = corr[rows, cols]

print("Threshold scan:")
for thresh in [0.9999, 0.999, 0.99, 0.98, 0.95, 0.9, 0.8, 0.5]:
    n = int(np.sum(np.abs(all_r) > thresh))
    print(f"  |r| > {thresh:.4f} : {n:5d} pairs")

print("\nTop 20 most correlated pairs:")
top20 = np.argsort(np.abs(all_r))[::-1][:20]
for k in top20:
    i, j = int(rows[k]), int(cols[k])
    print(f"  {all_npy[i]}  vs  {all_npy[j]}   r={all_r[k]:+.6f}")
