import numpy as np

# Point to your pristine data
npy_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\05_extracted_features\Chatter_Cohort_3D_Master.npy'
raw_data = np.load(npy_path)

print("\n--- INFINITY TEST ---")
inf_count = np.isinf(raw_data).sum()
if inf_count == 0:
    print("[PASS] 0 Infinite values detected.")
else:
    print(f"[FAIL] {inf_count} Infinite values detected. MATLAB will crash.")

print("\n--- Z-SCORE INTEGRITY TEST ---")
# Your data shape is (122, 464, 30): (ROIs, Timepoints, Subjects)
# The Z-scoring was applied across the time axis (axis=1)
means = np.mean(raw_data, axis=1)
stds = np.std(raw_data, axis=1)

# Check if means are mathematically close to 0 and stds are close to 1
mean_check = np.allclose(means, 0, atol=1e-5)
std_check = np.allclose(stds, 1, atol=1e-2)

if mean_check and std_check:
    print("[PASS] All regions correctly normalized (Mean = 0, Std = 1).")
else:
    print("[WARNING] Z-scoring deviations detected. Normalization failed.")
    print(f"  Worst Mean deviation from 0: {np.max(np.abs(means)):.6f}")
    print(f"  Worst Std deviation from 1:  {np.max(np.abs(stds - 1)):.6f}")