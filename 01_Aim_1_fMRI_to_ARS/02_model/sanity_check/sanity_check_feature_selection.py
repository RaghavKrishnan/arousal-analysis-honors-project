import numpy as np
import scipy.io as sio

# 1. Load the NSG output file using SciPy
file_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\output\Chatter_NSG_Job\sliding-dynFeat-Chatter.mat'
print("Loading 496MB matrix with SciPy... this may take a moment.")
data = sio.loadmat(file_path)
dynFC = data['dynFeat']

# 2. Dimensionality Test
print("\n--- DIMENSIONALITY ---")
print(f"Matrix Shape: {dynFC.shape}")
n_subj, n_edges, n_time = dynFC.shape

if n_edges == 7381:
    print("[PASS] Exactly 7,381 functional connections detected.")
else:
    print(f"[FAIL] Expected 7381 edges, but found {n_edges}.")

# 3. Missing Value (NaN) Test
print("\n--- DATA INTEGRITY ---")
nan_count = np.isnan(dynFC).sum()
if nan_count == 0:
    print("[PASS] 0 NaNs detected. Matrix is mathematically sound.")
else:
    print(f"[FAIL] {nan_count} NaNs detected. The SVR will crash if fed this data.")

# 4. Fisher Z-Transform Bounds Test
print("\n--- DISTRIBUTION BOUNDS ---")
max_val = np.nanmax(dynFC)
min_val = np.nanmin(dynFC)
mean_val = np.nanmean(dynFC)

print(f"Global Maximum: {max_val:.3f}")
print(f"Global Minimum: {min_val:.3f}")
print(f"Global Mean:    {mean_val:.3f}")

if max_val < 5.0 and min_val > -5.0:
    print("[PASS] Fisher Z-scores are within normal physiological bounds.")
else:
    print("[WARNING] Extreme values detected. Check for noise spikes or perfect correlations.")