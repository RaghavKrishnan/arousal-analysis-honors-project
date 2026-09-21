import nibabel as nib
from nilearn import plotting

underlay = nib.load(r'C:\Users\YourName\Desktop\Atlas_Files\MNI152_T1_2mm_Brain.nii')
overlay = nib.load(r'C:\Users\YourName\Desktop\Atlas_Files\Yeo2011_17Networks_N1000.split_components.FSL_MNI152_2mm.nii.gz')

print("Underlay shape:", underlay.shape, "| affine:\n", underlay.affine)
print("Overlay shape:", overlay.shape, "| affine:\n", overlay.affine)

plotting.plot_roi(overlay, bg_img=underlay, title="17Networks split-components on MNI152")
plotting.show()