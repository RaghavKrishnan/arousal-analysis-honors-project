import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.stats import zscore
from nilearn import datasets, plotting, image
from nilearn.connectome import ConnectivityMeasure
from nilearn.regions import connected_label_regions

# Suppress the divide-by-zero warnings from empty regions
warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():

    #1. The input data
    fmri_img_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\sub-S01_ses-1_task-BigBuckBunny_space-MNI_desc-ppres_bold.nii'
    print("Successfully loaded the fMRI data from:", fmri_img_path) 

    #Loading Brainnetome Atlas
    bn_atlas_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\BN_Atlas_246_2mm.nii'
    print("Successfully loaded the Brainnetome Atlas from:", bn_atlas_path)
    bn_atlas = image.load_img(bn_atlas_path)
    
    #2. Load the Yeo atlas
    print("Fetching the Yeo Atlas")
    yeo_atlas = datasets.fetch_atlas_yeo_2011(n_networks=17, thickness = 'thick')

    #3. Convert the 17 networks to 114 ROIs
    print("Separating disconnected networks into the 114 ROIs")
    yeo_114_rois = connected_label_regions(yeo_atlas.maps, min_size=100)

    #Building combined atlas with both Brainnetome and Yeo 114 ROIs
    print("Combining Brainnetome Atlas and Yeo 114 ROIs into a single atlas")
    bn_resampled = image.resample_to_img(bn_atlas, yeo_114_rois, interpolation='nearest')

    yeo_data = yeo_114_rois.get_fdata()
    bn_data = bn_resampled.get_fdata()
    custom_data = np.copy(yeo_data)
    
    #Left hemisphere
    custom_data[np.isin(bn_data, [211, 213])] = 115 
    custom_data[np.isin(bn_data, [215, 217])] = 116 
    custom_data[np.isin(bn_data, [219, 221, 223, 225, 227, 229])] = 117 
    custom_data[np.isin(bn_data, [231, 233, 235, 237, 239, 241, 243, 245])] = 118 

    # Right hemisphere
    custom_data[np.isin(bn_data, [212, 214])] = 119 
    custom_data[np.isin(bn_data, [216, 218])] = 120 
    custom_data[np.isin(bn_data, [220, 222, 224, 226, 228, 230])] = 121 
    custom_data[np.isin(bn_data, [232, 234, 236, 238, 240, 242, 244, 246])] = 122 
     
    new_122_atlas = image.new_img_like(yeo_114_rois, custom_data)

    new_122_atlas.to_filename(r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\custom_122_atlas.nii.gz')


    # Creating a manual masker engine since the nilearn masker does not support the 122-ROI atlas
    print("Bypassing nilearn Masker to force strict 122-ROI extraction...")

    # Loading the fMRI image
    fmri_img = image.load_img(fmri_img_path)
    fmri_data = fmri_img.get_fdata()
    num_timepoints = fmri_data.shape[3]

    # Aligning the atlas to the fMRI grid
    print("Aligning atlas to fMRI grid...")
    atlas_aligned = image.resample_to_img(new_122_atlas, fmri_img, interpolation='nearest')
    atlas_data = atlas_aligned.get_fdata()

    # Create the empty placeholder spreadsheet for the time series data
    time_series = np.zeros((num_timepoints, 122))

    # Manually extracting each region 
    print("Extracting signals for all 122 regions...")
    for roi in range(1, 123):
        # Find the 3D coordinates for the specific region
        roi_mask = (atlas_data == roi)
        
        # If the region exists in the scanner's field of view...
        if np.any(roi_mask):
            # Grab all voxels inside this region across all timepoints
            roi_voxels = fmri_data[roi_mask]
            # Calculate the mean signal and drop it into its specific column
            time_series[:, roi - 1] = np.nanmean(roi_voxels, axis=0)
        
        # If the scanner clipped it, the column safely remains 0.0

    
    # --- SCANNER FOV DIAGNOSTIC ---
    missing_regions = []
    for col in range(122):
        #If all numbers in this column are zero, then, 
        if np.all(time_series[:, col] == 0):
            missing_regions.append(col + 1) # +1 because Python counts from 0
            
    print("\n--- MISSING REGIONS REPORT ---")
    print(f"Total missing regions: {len(missing_regions)}")
    print(f"Exact Region Numbers: {missing_regions}")
    print("------------------------------\n")




    print(f"Raw time series shape: {time_series.shape}")

    # Standardizing the signals using z-score normalization while ignoring NaNs
    print("Standardizing signals...")
    for col in range(122):
        
        if not np.all(time_series[:, col] == 0):
            time_series[:, col] = zscore(time_series[:, col], nan_policy='omit')

    # Cleaning up 
    time_series = np.nan_to_num(time_series, nan=0.0)
    print(f"Final standardized time series shape: {time_series.shape}")

    #6. Computing the functional connectivity matrix
    print("Computing the functional connectivity matrix...")
    correlation_measure = ConnectivityMeasure(kind='correlation')   
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    print(f"Functional connectivity matrix computed. Shape: {correlation_matrix.shape}")

    # Replace NaNs with 0
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    print("NaN values replaced with 0. Any remaining NaNs:", np.isnan(correlation_matrix).any())

   # 7. Save the raw numerical data array
    print("Saving static matrix data...")
    save_path_data = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\results\Aim_1_graphs\sub-S01_static_matrix.npy'
    os.makedirs(os.path.dirname(save_path_data), exist_ok=True)
    np.save(save_path_data, correlation_matrix)
    
    # 8. Generate and save the visual heatmap
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