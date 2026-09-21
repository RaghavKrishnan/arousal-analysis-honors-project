import numpy as np
from nilearn import image, plotting, datasets
import matplotlib.pyplot as plt

# 1. Define File Paths
fmri_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\sub-S01_ses-1_task-BigBuckBunny_space-MNI_desc-ppres_bold.nii'
atlas_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\custom_122_atlas.nii.gz' 
output_png = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS\04_data\Ventral_Parcels_FOV_Cutoff.png'

print("Step 1: Loading complete MNI152 standard brain template...")
standard_brain = datasets.load_mni152_template()

print("Step 2: Isolating Ventral Parcels 96-114 from the atlas...")
atlas_img = image.load_img(atlas_path)
atlas_data = atlas_img.get_fdata()
# Isolate regions 96 to 114 inclusive as a binary mask (int32 format)
mask_data = np.logical_and(atlas_data >= 96, atlas_data <= 114).astype(np.int32)
mask_img = image.new_img_like(atlas_img, mask_data)

print("Step 3: Extracting non-demeaned volume from functional time-series...")
# Extract the first volume to get valid non-zero spatial intensities for the FOV outline
functional_vol = image.index_img(fmri_path, 0)

print("Step 4: Generating multi-slice axial visualization...")
# Initialize the figure with the complete standard anatomical brain as the canvas
display = plotting.plot_anat(
    standard_brain,
    display_mode='z',
    cut_coords=[-45, -35, -25, -15, -5],
    title="Verification: Missing Parcels 96-114 (Red) vs. Subject Functional FOV (Heat)"
)

# Overlay the isolated ventral parcels (safely bounded inside standard MNI space)
display.add_overlay(
    mask_img,
    cmap='Reds',
    transparency=0.5,
    vmin=0.1,
    vmax=1.0  # Hardcoded max prevents scaling math crashes on empty lower slices
)

# Overlay the subject's functional volume to visually demonstrate where it cuts off
display.add_overlay(
    functional_vol,
    cmap='magma',
    transparency=0.5,
    threshold=100  # Filters out the scanner's empty background noise
)

# Save high-resolution graphic for analysis
plt.savefig(output_png, dpi=300, bbox_inches='tight')
display.close()
print(f"\n[SUCCESS] Verification image saved to: {output_png}")