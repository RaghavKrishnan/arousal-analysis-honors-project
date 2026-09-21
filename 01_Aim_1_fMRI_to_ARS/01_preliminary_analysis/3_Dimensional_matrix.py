import os
import glob
import numpy as np
import warnings
import gc
from scipy.stats import zscore
from nilearn import image

warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    
    MOVIE_NAME = "Superhero" 

    # 2. Define Local Directories
    base_dir = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS'
    data_dir = os.path.join(base_dir, '04_data') 
    
    # Dynamically point to the correct movie folder
    movie_dir = os.path.join(data_dir, 'fmri_subject_data', MOVIE_NAME)
    
    # Route output to a dedicated subfolder based on the movie name
    output_dir = os.path.join(base_dir, '05_extracted_features', MOVIE_NAME)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Load the Master Atlas
    atlas_path = os.path.join(base_dir, r'04_data\atlas_considered_for_use\custom_122_atlas.nii')
    print("==========================================")
    print(f"Initializing feature extraction for: {MOVIE_NAME}")
    print("==========================================\n")
    print(f"Loading master atlas from: {atlas_path}")
    master_atlas = image.load_img(atlas_path)

    # 4. Dynamically discover the files
    print(f"\nScanning directory for valid '{MOVIE_NAME}' fMRI files...")
    search_pattern = os.path.join(movie_dir, '**', f'*task-{MOVIE_NAME}*space-MNI*bold.nii*')
    
    raw_matches = glob.glob(search_pattern, recursive=True)
    valid_files = [f for f in raw_matches if os.path.isfile(f) and not f.endswith('.part')]
    valid_files.sort()
    
    if not valid_files:
        print(f"\n[ERROR] No valid files found for {MOVIE_NAME}. Did the download script finish successfully?")
        return

    print(f"Discovered {len(valid_files)} valid '{MOVIE_NAME}' files. Beginning direct-read extraction...")
    
    all_subjects_data = []
    processed_subject_ids = []

    # 5. Process Each Subject
    for file_idx, fmri_path in enumerate(valid_files):
        filename = os.path.basename(fmri_path)
        subject_id = filename.split('_')[0] 
        
        print(f"\n    -> [{file_idx + 1}/{len(valid_files)}] Processing {subject_id}...")

        try:
            # --- DIRECT READ FROM COMPRESSED FILE ---
            fmri_img = image.load_img(fmri_path)
            fmri_data = fmri_img.get_fdata()
            num_timepoints = fmri_data.shape[3]

            atlas_aligned = image.resample_to_img(master_atlas, fmri_img, interpolation='nearest')
            atlas_data = atlas_aligned.get_fdata()

            temp_time_series = np.zeros((num_timepoints, 122))
            empty_rois = [] # Diagnostic tracker for missing regions

            for roi in range(1, 123):
                roi_mask = (atlas_data == roi)
                if np.any(roi_mask):
                    temp_time_series[:, roi - 1] = np.nanmean(fmri_data[roi_mask], axis=0)
                else:
                    empty_rois.append(roi) # Log completely empty masks

            if empty_rois:
                print(f"        [WARNING] {subject_id} has empty ROIs! Regions {empty_rois} will flatline in MATLAB.")

            for col in range(122):
                if not np.all(temp_time_series[:, col] == 0):
                    temp_time_series[:, col] = zscore(temp_time_series[:, col], nan_policy='omit')

            temp_time_series = np.nan_to_num(temp_time_series, nan=0.0)
            parcel_x_time = temp_time_series.T 
            
            subject_out_path = os.path.join(output_dir, f"{subject_id}_task-{MOVIE_NAME}_matrix.npy")
            np.save(subject_out_path, parcel_x_time)
            
            all_subjects_data.append(parcel_x_time)
            processed_subject_ids.append(subject_id)
            
            # --- MEMORY FLUSH (No Deletion) ---
            fmri_img.uncache()
            del fmri_data
            del fmri_img
            del atlas_aligned
            del atlas_data
            gc.collect()
            print(f"        [Success] Extracted {subject_id} and cleared RAM.")
            
        except Exception as e:
            print(f"        [!] Error processing {subject_id}: {e}")

    print("\n--------------------------------------------------")
    if not all_subjects_data:
        print("No data processed. Check file paths and downloads.")
        return

    # 6. Save the intermediate archive
    print("Saving intermediate archive of all un-truncated 2D matrices...")
    matrices_dict = {subj: matrix for subj, matrix in zip(processed_subject_ids, all_subjects_data)}
    intermediate_out_path = os.path.join(output_dir, f"{MOVIE_NAME}_All_2D_Matrices_Archive.npz")
    np.savez_compressed(intermediate_out_path, **matrices_dict)
    
    # 7. Constructing the Final 3D Array
    print("Constructing the 3D array across all participants...")
    min_timepoints = min([matrix.shape[1] for matrix in all_subjects_data])
    
    truncated_matrices = []
    for matrix in all_subjects_data:
        truncated_matrix = matrix[:, :min_timepoints]
        truncated_matrices.append(truncated_matrix)

    master_3d_array = np.dstack(truncated_matrices)
    master_out_path = os.path.join(output_dir, f"{MOVIE_NAME}_Cohort_3D_Master.npy")
    np.save(master_out_path, master_3d_array)
    
    subject_log_path = os.path.join(output_dir, f"{MOVIE_NAME}_Subject_Index_Key.txt")
    with open(subject_log_path, 'w') as f:
        for idx, subj in enumerate(processed_subject_ids):
            f.write(f"Z-Axis Index {idx}: {subj}\n")

    print(f"\nSUCCESS: Pipeline Complete for {MOVIE_NAME}.")
    print(f"Final 3D Dimensionality: {master_3d_array.shape}")
    print(f"Data saved to: {output_dir}")

if __name__ == "__main__":
    main()