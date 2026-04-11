import os
import random
import numpy as np

DATA_FOLDER = "/projappl/project_2018026/super_data"
N_FILES = 10

all_npy = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".npy")]
chosen = random.sample(all_npy, N_FILES)

print("=== ANALYSIS REPORT LOANNGUYEN ===")
print(f"{'File':<30} | {'Max':>10} | {'Min':>10} | {'Mean':>10} | {'Std':>10}")
print("-" * 80)

max_values = []
min_values = []
mean_values = []
std_values = []

for fname in chosen:
    arr = np.load(os.path.join(DATA_FOLDER, fname))
    max_val  = arr.max()
    min_val  = arr.min()
    mean_val = arr.mean()
    std_val  = arr.std()

    max_values.append(max_val)
    min_values.append(min_val)
    mean_values.append(mean_val)
    std_values.append(std_val)

    print(f"{fname:<30} | {max_val:>10.3f} | {min_val:>10.3f} | {mean_val:>10.3f} | {std_val:>10.3f}")

print("-" * 80)
print(f"\nSummary across {N_FILES} files:")
print(f"  Sum of max values  : {sum(max_values):.3f}")
print(f"  Sum of min values  : {sum(min_values):.3f}")
print(f"  Sum of mean values : {sum(mean_values):.3f}")
print(f"  Average std        : {sum(std_values)/N_FILES:.3f}")
print(f"  Brightest image    : {chosen[max_values.index(max(max_values))]}")
print(f"  Darkest image      : {chosen[min_values.index(min(min_values))]}")
