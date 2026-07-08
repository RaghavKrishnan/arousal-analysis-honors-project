import os
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.regions import connected_label_regions
from nilearn import plotting

def main():

    #1. The input data
    fmri_img_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\sub-S01_ses-1_task-BigBuckBunny_space-MNI_desc-ppres_bold.nii'
    print("Successfully loaded the fMRI data from:", fmri_img_path) 


    #2. Load the atlas
    print("Fetching the Yeo Atlas")
    yeo_atlas = datasets.fetch_atlas_yeo_2011(n_networks=17, thickness = 'thick')

    #3. Convert networks to 114 ROIs
    print("Separating disconnected networks into the 114 ROIs")
    # 'thick_17' pulls the 17-network parcellation fitted to thick cortex segmentations.
    # connected_label_regions breaks it down into the 114 localized ROIs.
    yeo_114_rois = connected_label_regions(yeo_atlas.maps, min_size=100)

    #4. Creating the Masker engine
    print("Intializing the NiftiLabelsMasker")
    masker = NiftiLabelsMasker(labels_img=yeo_114_rois, standardize='zscore_sample', memory='nilearn_cache', verbose=5)

    #5. Dimensionality reduction: Extracting the time series from the fMRI data
    print("Extracting the time series from the fMRI data")
    time_series = masker.fit_transform(fmri_img_path)
    print("Extraction complete. Time series shape:", time_series.shape)

    #6. Computing the functional connectivity matrix
    print("Computing the functional connectivity matrix")
    correlation_measure = ConnectivityMeasure(kind='correlation')   
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    print("Functional connectivity matrix computed. Shape:", correlation_matrix.shape)

   # 7. Save the raw numerical data array
    print("Saving static matrix data...")
    save_path_data = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\results\Aim_1_graphs\sub-S01_static_matrix.npy'
    
    # NEW: Tell Python to create the 'Aim_1_graphs' folder if it is missing
    os.makedirs(os.path.dirname(save_path_data), exist_ok=True)
    
    np.save(save_path_data, correlation_matrix)
    
    # 8. Generate and save the visual heatmap
    print("Generating heatmap image...")
    save_path_img = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\results\Aim_1_graphs\sub-S01_static_heatmap.png'
    
    # plot_matrix creates a standard neuroimaging heatmap
    display = plotting.plot_matrix(
        correlation_matrix, 
        colorbar=True, 
        vmax=0.8, 
        vmin=-0.8, 
        title="Subject 01 - Static Connectivity (Yeo 114)"
    )
    plt.savefig(save_path_img, bbox_inches='tight', dpi=300)
    
    print("The data and heatmap have been saved.")

if __name__ == "__main__":
    main()


