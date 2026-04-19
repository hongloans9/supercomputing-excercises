"""
Generate synthetic .npy files in super_data/ for the analysis exercise.
Creates 80 files of size 2000x2000 (float64) with random data.
"""
import os
import numpy as np

DATA_FOLDER = "super_data"
N_FILES = 80
SHAPE = (2000, 2000)

os.makedirs(DATA_FOLDER, exist_ok=True)

for i in range(N_FILES):
    fname = f"image_{i:04d}.npy"
    arr = np.random.random(SHAPE).astype(np.float64)
    np.save(os.path.join(DATA_FOLDER, fname), arr)
    print(f"Created {fname}  ({arr.nbytes / 1e6:.1f} MB)")

print(f"\nDone. {N_FILES} files in {DATA_FOLDER}/")
