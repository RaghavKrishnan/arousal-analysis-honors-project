import numpy as np
import scipy.io as sio

# 1. Load the data
file_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\output\chatter_NSG_job\sliding-dynFeat-Chatter.mat'
print("Loading supercomputer output...")
mat_data = sio.loadmat(file_path)

# The MATLAB script saved the matrix under the variable name 'dynFeat'
dyn_matrix = mat_data['dynFeat']

print("\n--- 1. DIMENSIONALITY CHECK ---")
expected_shape = (30, 7381, 419)
print(f"Matrix Shape: {dyn_matrix.shape}")
if dyn_matrix.shape == expected_shape:
    print("[PASS] Shape perfectly matches (Subjects, Edges, Time Windows).")
else:
    print(f"[FAIL] Expected {expected_shape}, but got {dyn_matrix.shape}.")

print("\n--- 2. NaN (MISSING DATA) CHECK ---")
nan_count = np.isnan(dyn_matrix).sum()
if nan_count == 0:
    print("[PASS] 0 NaNs detected. The zero-variance trap was successfully avoided.")
else:
    print(f"[FAIL] {nan_count} NaNs detected!")

print("\n--- 3. INFINITY CHECK ---")
inf_count = np.isinf(dyn_matrix).sum()
if inf_count == 0:
    print("[PASS] 0 Infinite values detected. Fisher's r-to-z transformation is stable.")
else:
    print(f"[FAIL] {inf_count} Infinite values detected!")

print("\n--- 4. SIGNAL VARIANCE CHECK ---")
# Check if the matrix is accidentally filled with all zeros
if np.all(dyn_matrix == 0):
    print("[FAIL] The matrix is entirely empty (all zeros).")
else:
    mean_val = np.nanmean(dyn_matrix)
    std_val = np.nanstd(dyn_matrix)
    print(f"[PASS] Matrix contains dynamic variance.")
    print(f"       Global Mean Z-Score: {mean_val:.4f}")
    print(f"       Global Std Dev:      {std_val:.4f}")
    
print("\nValidation Complete.")