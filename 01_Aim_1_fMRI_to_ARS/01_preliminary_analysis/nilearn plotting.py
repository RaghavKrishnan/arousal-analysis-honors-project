from nilearn import plotting
import matplotlib.pyplot as plt

#1 Import the files 
fmri_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\sub-S01_ses-1_task-BigBuckBunny_space-MNI_desc-ppres_bold.nii'
mask_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\missing_parcels_mask.nii.gz'
output_png = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\FOV_cutoff_proof.png'

print("Rendering brain slices...")

# 2. Tell Nilearn to plot the mask in solid red over the grayscale brain scan
# display_mode='x' gives us the Sagittal (side-profile) view
plotting.plot_roi(
    roi_img=mask_path, 
    bg_img=fmri_path,
    display_mode='x', 
    cut_coords=5,        # Show 5 different side-profile slices
    title="Scanner FOV Cutoff: Missing Regions 96-114",
    cmap='Reds',         # Make the missing regions bright red
    alpha=0.8            # Make it slightly transparent
)

# 3. Save it as a normal picture file
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"Success! Open this file in Windows: {output_png}")