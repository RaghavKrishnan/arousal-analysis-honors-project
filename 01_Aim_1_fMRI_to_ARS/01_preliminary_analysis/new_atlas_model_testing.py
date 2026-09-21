import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.stats import zscore
from nilearn import image, plotting
from nilearn.connectome import ConnectivityMeasure

# Suppress the divide-by-zero warnings from empty regions
warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():

    # 1. The input fMRI data
    fmri_img_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\fmri_subject_data\sub-01\sub-S01_ses-1_task-BigBuckBunny_space-MNI_desc-ppres_bold.nii'
    print(f"Successfully loaded the fMRI data from: {fmri_img_path}") 

    # 2. Loading the Local 2mm Brainnetome Atlas
    bn_atlas_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\workspace_folder_when_running_AFNI\BN_Atlas_246_2mm.nii'
    print(f"Successfully loaded the Brainnetome Atlas from: {bn_atlas_path}")
    bn_atlas = image.load_img(bn_atlas_path)
    
    # 3. Loading the Local 2mm Yeo Atlas (Pre-split into 114 ROIs)
    yeo_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\workspace_folder_when_running_AFNI\Yeo2011_17Networks_N1000.split_components.FSL_MNI152_2mm.nii'
    print(f"Fetching the local 2mm Yeo 114 Atlas from: {yeo_path}")
    yeo_114_rois = image.load_img(yeo_path)

    # 4. Building the Combined 122-ROI Atlas
    print("Combining Brainnetome Subcortical ROIs and Yeo 114 ROIs...")
    
    # Resample Brainnetome to perfectly match the local Yeo 2mm grid affine
    bn_resampled = image.resample_to_img(bn_atlas, yeo_114_rois, interpolation='nearest')

    yeo_data = yeo_114_rois.get_fdata()
    bn_data = bn_resampled.get_fdata()
    
    # Ensure base data is strictly int32 to avoid downstream visualization errors
    custom_data = np.copy(yeo_data).astype(np.int32)
    
    # STRICT PRECEDENCE RULE: 
    # Only assign subcortical values where the Yeo atlas is completely empty (yeo_data == 0).
    # This mathematically prevents Brainnetome from overwriting cortical voxels.
    empty_space_mask = (yeo_data == 0)
    
    # Left hemisphere subcortical mappings
    custom_data[np.isin(bn_data, [211, 213]) & empty_space_mask] = 115 # L Amygdala
    custom_data[np.isin(bn_data, [215, 217]) & empty_space_mask] = 116 # L Hippocampus
    custom_data[np.isin(bn_data, [219, 221, 223, 225, 227, 229]) & empty_space_mask] = 117 # L Basal Ganglia
    custom_data[np.isin(bn_data, [231, 233, 235, 237, 239, 241, 243, 245]) & empty_space_mask] = 118 # L Thalamus

    # Right hemisphere subcortical mappings
    custom_data[np.isin(bn_data, [212, 214]) & empty_space_mask] = 119 # R Amygdala
    custom_data[np.isin(bn_data, [216, 218]) & empty_space_mask] = 120 # R Hippocampus
    custom_data[np.isin(bn_data, [220, 222, 224, 226, 228, 230]) & empty_space_mask] = 121 # R Basal Ganglia
    custom_data[np.isin(bn_data, [232, 234, 236, 238, 240, 242, 244, 246]) & empty_space_mask] = 122 # R Thalamus
     
    # Save the corrected 122-ROI atlas
    new_122_atlas = image.new_img_like(yeo_114_rois, custom_data)
    atlas_save_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\custom_122_atlas.nii.gz'
    new_122_atlas.to_filename(atlas_save_path)
    print(f"New 122-ROI hybrid atlas safely generated and saved to: {atlas_save_path}")

    # 5. Manual Masker Engine
    print("Bypassing nilearn Masker to force strict 122-ROI extraction...")

    fmri_img = image.load_img(fmri_img_path)
    fmri_data = fmri_img.get_fdata()
    num_timepoints = fmri_data.shape[3]

    print("Aligning atlas to fMRI grid...")
    atlas_aligned = image.resample_to_img(new_122_atlas, fmri_img, interpolation='nearest')
    atlas_data = atlas_aligned.get_fdata()

    # Create the empty placeholder spreadsheet for the time series data
    time_series = np.zeros((num_timepoints, 122))
# Create the empty placeholder spreadsheet for the time series data
    time_series = np.zeros((num_timepoints, 122))

    # --- NEW: Dictionary to hold the raw, un-averaged voxel data ---
    raw_voxel_dict = {}

    # Manually extracting each region 
    print("Extracting signals for all 122 regions and logging raw voxel values...")
    for roi in range(1, 123):
        roi_mask = (atlas_data == roi)
        
        if np.any(roi_mask):
            # roi_voxels shape: (Number of Voxels in this ROI, Number of Timepoints)
            roi_voxels = fmri_data[roi_mask]
            
            # 1. Save the raw voxel array to our dictionary
            raw_voxel_dict[f'ROI_{roi}'] = roi_voxels
            
            # 2. Calculate the mean signal and drop it into its specific column
            time_series[:, roi - 1] = np.nanmean(roi_voxels, axis=0)

    # --- NEW INTERMEDIATE DIAGNOSTIC STEP: Save Raw Voxels ---
    raw_voxels_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\sub-S01_raw_voxels_per_parcel.npz'
    print(f"Saving raw voxel arrays for manual inspection to: {raw_voxels_path}")
    # np.savez_compressed saves the dictionary efficiently without crashing RAM
    np.savez_compressed(raw_voxels_path, **raw_voxel_dict)
    # ---------------------------------------------------------

    # --- SCANNER FOV DIAGNOSTIC ---
    missing_regions = []
    for col in range(122):
        if np.all(time_series[:, col] == 0):
            missing_regions.append(col + 1) # +1 to match exact anatomical labels
            
    print("\n--- MISSING REGIONS REPORT ---")
    print(f"Total missing regions: {len(missing_regions)}")
    print(f"Exact Region Numbers: {missing_regions}")
    print("------------------------------\n")

    print(f"Raw time series shape: {time_series.shape}")

    # 6. Standardizing the signals using z-score normalization
    print("Standardizing signals...")
    for col in range(122):
        if not np.all(time_series[:, col] == 0):
            time_series[:, col] = zscore(time_series[:, col], nan_policy='omit')

    # Cleaning up isolated NaNs to prevent matrix contamination
    time_series = np.nan_to_num(time_series, nan=0.0)
    print(f"Final standardized time series shape: {time_series.shape}")

    # 7. Computing the functional connectivity matrix
    print("Computing the functional connectivity matrix...")
    correlation_measure = ConnectivityMeasure(kind='correlation')   
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    print(f"Functional connectivity matrix computed. Shape: {correlation_matrix.shape}")

    # Replace NaNs generated by missing region standard deviations with 0
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    print("NaN values replaced with 0. Any remaining NaNs:", np.isnan(correlation_matrix).any())

    # 8. Save the raw numerical data array
    print("Saving static matrix data...")
    save_path_data = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\results\Aim_1_graphs\sub-S01_static_matrix.npy'
    os.makedirs(os.path.dirname(save_path_data), exist_ok=True)
    np.save(save_path_data, correlation_matrix)
    
    # 9. Generate and save the visual heatmap
    print("Generating heatmap image...")
    save_path_img = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\results\Aim_1_graphs\sub-S01_static_heatmap.png'
    
    display = plotting.plot_matrix(
        correlation_matrix, 
        colorbar=True, 
        vmax=0.8, 
        vmin=-0.8, 
        title="Subject 01 - Static Connectivity (Custom 122 ROI atlas)"
    )
    plt.savefig(save_path_img, bbox_inches='tight', dpi=300)
    
    print("The data and heatmap have been saved.")

if __name__ == "__main__":
    main()