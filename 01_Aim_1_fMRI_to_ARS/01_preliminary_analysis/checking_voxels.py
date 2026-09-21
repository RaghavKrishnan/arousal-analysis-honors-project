import numpy as np

# 1. Load the compressed archive
file_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\sub-S01_raw_voxels_per_parcel.npz'
raw_data = np.load(file_path)

# 2. Define the region you want to investigate (e.g., Left Thalamus is 118)
target_roi = 'ROI_115'

if target_roi in raw_data:
    voxel_matrix = raw_data[target_roi]
    
    print(f"\n--- DATA FOR {target_roi} ---")
    print(f"Matrix Shape: {voxel_matrix.shape}")
    print(f"Total Voxels in this parcel: {voxel_matrix.shape[0]}")
    print(f"Total Timepoints recorded: {voxel_matrix.shape[1]}\n")
    
    # 3. Print the raw magnetic intensities for the very first voxel across the first 20 timepoints
    print("Sample Data (Voxel 1, Timepoints 1-20):")
    print(voxel_matrix[0, :20])
    
else:
    print(f"{target_roi} is completely empty (0 voxels found).")