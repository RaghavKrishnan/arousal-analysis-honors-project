import numpy as np
import scipy.io as sio
import os

# 1. Define the exact file paths
input_npy = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\05_extracted_features\Superhero\Superhero_Cohort_3D_Master.npy'
output_mat = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\05_extracted_features\Superhero\Superhero_Cohort_3D_Master.mat'

# 2. Load the 3D numpy array
print("Loading Python tensor...")
tensor = np.load(input_npy)

# 3. Save as a MATLAB .mat file
# The dictionary key 'master_array' is critical, as it becomes the variable name in MATLAB
print("Converting to MATLAB format...")
sio.savemat(output_mat, {'master_array': tensor})

print(f"Conversion complete! File saved to: {output_mat}")