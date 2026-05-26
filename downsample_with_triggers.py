import pandas as pd
import numpy as np
import os
import glob

# --- 1. CONFIGURATION ---
# The folder with your sorted raw .tsv files
RAW_SORTED_DIR = 'output_sorted_by_movie/'

# A new folder for the correctly sliced 1Hz output
OUTPUT_SLICED_DIR = 'downsampled_sliced_1Hz/'

# The movies we will process
MOVIES_TO_PROCESS = [
    "BigBuckBunny",
    "Chatter",
    "TearsOfSteel",
    "TheSecretNumber"
]

# --- 2. TRIGGER-FINDING FUNCTION (ROBUST) ---
def find_master_movie_indices(tsv_path, trigger_column_index=7):
    """
    Finds the start and end index of the *longest* 0->5 / 5->0 trigger pair.
    """
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        if trigger_column_index >= len(df.columns):
            print(f"    - WARNING: 'H' column (index 7) not found in reference file. Skipping movie.")
            return None, None
            
        trigger = df.iloc[:, trigger_column_index]
    except Exception as e:
        print(f"    - ERROR reading {tsv_path}: {e}")
        return None, None

    starts = (trigger == 5) & (trigger.shift(1) == 0)
    ends = (trigger == 0) & (trigger.shift(1) == 5)
    
    start_indices = starts[starts].index
    end_indices = ends[ends].index

    if len(start_indices) == 0 or len(end_indices) == 0:
        print(f"    - WARNING: No 0->5 or 5->0 triggers found in reference file. Skipping movie.")
        return None, None

    longest_duration = 0
    best_start = -1
    best_end = -1

    for start_idx in start_indices:
        possible_ends = end_indices[end_indices > start_idx]
        if len(possible_ends) > 0:
            end_idx = possible_ends[0]
            duration = end_idx - start_idx
            
            if duration > longest_duration:
                longest_duration = duration
                best_start = start_idx
                best_end = end_idx

    if best_start == -1:
        print(f"    - WARNING: No valid end trigger found after a start trigger in reference file. Skipping movie.")
        return None, None

    return best_start, best_end

# --- 3. DOWNSAMPLING FUNCTION ---
def downsample_sliced_data(tsv_path, start_index, end_index, original_hz=1000):
    """Loads a raw TSV, extracts the EDA signal, slices it, and downsamples to 1Hz."""
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        
        # Robustly find the SC column
        if 'SC' in df.columns:
            eda_signal = df['SC'].values
        elif df.shape[1] >= 3:
            eda_signal = df.iloc[:, 2].values
        else:
            print(f"    - ERROR: No 'SC' column found in {os.path.basename(tsv_path)}. Skipping file.")
            return None
            
        # --- CRITICAL STEP ---
        # Slice the raw signal *before* downsampling using the master indices
        sliced_eda_signal = eda_signal[start_index:end_index]
        
        # Now downsample the correctly sliced data
        window_size = int(original_hz / 1) # 1000
        downsampled_signal = [
            np.mean(sliced_eda_signal[i : i + window_size])
            for i in range(0, len(sliced_eda_signal), window_size)
        ]
        
        downsampled_df = pd.DataFrame({
            "Second": np.arange(len(downsampled_signal)),
            "SkinConductance": downsampled_signal
        })
        return downsampled_df
        
    except Exception as e:
        print(f"    - ERROR during downsampling {os.path.basename(tsv_path)}: {e}")
        return None

# --- 4. NEW MAIN SCRIPT (TWO-STAGE PROCESS) ---
def main():
    if not os.path.exists(OUTPUT_SLICED_DIR):
        os.makedirs(OUTPUT_SLICED_DIR)

    # --- STAGE 1: Find the Master Timings for each movie ---
    print("--- STAGE 1: Finding Master Timings for each movie ---")
    movie_timings = {}
    
    for movie in MOVIES_TO_PROCESS:
        movie_folder_path = os.path.join(RAW_SORTED_DIR, movie)
        if not os.path.exists(movie_folder_path):
            print(f"  - WARNING: Folder for '{movie}' not found. Skipping.")
            continue
        
        # Find a reference file (the first one)
        file_list = glob.glob(os.path.join(movie_folder_path, '*_physio*.tsv'))
        if not file_list:
            print(f"  - WARNING: No physio files found for '{movie}'. Skipping.")
            continue
            
        reference_file = file_list[0]
        print(f"  - Analyzing {movie} using reference file: {os.path.basename(reference_file)}")
        
        # Get the master start/end indices
        start_idx, end_idx = find_master_movie_indices(reference_file)
        
        if start_idx is not None:
            duration_sec = (end_idx - start_idx) / 1000.0
            print(f"    - Found timing: Start Index {start_idx}, End Index {end_idx} (Duration: {duration_sec:.1f}s)")
            movie_timings[movie] = (start_idx, end_idx)
        else:
            print(f"    - ERROR: Could not find valid triggers in reference file. This movie will be skipped.")

    # --- STAGE 2: Process all files using the Master Timings ---
    print("\n--- STAGE 2: Processing all files using Master Timings ---")
    
    for movie_name, timings in movie_timings.items():
        start_idx, end_idx = timings
        print(f"\nProcessing Movie: {movie_name} (using index range {start_idx} to {end_idx})")
        
        movie_folder_path = os.path.join(RAW_SORTED_DIR, movie_name)
        output_movie_folder = os.path.join(OUTPUT_SLICED_DIR, movie_name)
        if not os.path.exists(output_movie_folder):
            os.makedirs(output_movie_folder)
            
        # Get ALL participant files for this movie
        participant_files = glob.glob(os.path.join(movie_folder_path, '*_physio*.tsv'))
        
        for p_file in participant_files:
            filename = os.path.basename(p_file)
            print(f"  - Slicing & Downsampling: {filename}")
            
            # 2. Downsample *only* that slice using the master timings
            downsampled_data = downsample_sliced_data(p_file, start_idx, end_idx)
            
            if downsampled_data is not None:
                # 3. Save the new, correct file
                output_filename = f"downsampled_1Hz_SLICED_{filename.replace('.tsv', '.csv')}"
                output_path = os.path.join(output_movie_folder, output_filename)
                downsampled_data.to_csv(output_path, index=False)
                print(f"    - Success: Saved new 1Hz file with {len(downsampled_data)} seconds.")
            else:
                print(f"    - ERROR: Failed to process this file.")

    print("\n--- Final Sliced Downsampling Complete ---")
    print(f"All new files are in the '{OUTPUT_SLICED_DIR}' folder.")

if __name__ == "__main__":
    main()