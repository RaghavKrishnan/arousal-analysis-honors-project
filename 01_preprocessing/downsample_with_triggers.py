"""
downsample_with_triggers.py
Location: 01_preprocessing/
"""
import pandas as pd
import numpy as np
import os
import glob

# --- 1. CONFIGURATION ---
RAW_SORTED_DIR   = '../data/output_sorted_by_movie/'
OUTPUT_SLICED_DIR = '../results/downsampled_sliced_1Hz/'

MOVIES_TO_PROCESS = [
    "BigBuckBunny", "Chatter", "TearsOfSteel", "TheSecretNumber",
    "AfterTheRain", "BetweenViewings", "FirstBite", "LessonLearned",
    "Payload", "Sintel", "Spaceman", "Superhero", "ToClaireFromSonny", "YouAgain"
]

# --- 2. TRIGGER-FINDING FUNCTION ---
def find_master_movie_indices(tsv_path, trigger_column_index=7):
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        if trigger_column_index >= len(df.columns):
            print(f"    - WARNING: Column index {trigger_column_index} not found.")
            return None, None
        trigger = df.iloc[:, trigger_column_index]
    except Exception as e:
        print(f"    - ERROR reading {tsv_path}: {e}")
        return None, None

    starts = (trigger == 5) & (trigger.shift(1) == 0)
    ends   = (trigger == 0) & (trigger.shift(1) == 5)

    start_indices = starts[starts].index
    end_indices   = ends[ends].index

    if len(start_indices) == 0 or len(end_indices) == 0:
        print(f"    - WARNING: No valid triggers found.")
        return None, None

    longest_duration = 0
    best_start, best_end = -1, -1

    for start_idx in start_indices:
        possible_ends = end_indices[end_indices > start_idx]
        if len(possible_ends) > 0:
            end_idx  = possible_ends[0]
            duration = end_idx - start_idx
            if duration > longest_duration:
                longest_duration = duration
                best_start = start_idx
                best_end   = end_idx

    if best_start == -1:
        return None, None

    return best_start, best_end

# --- 3. DOWNSAMPLING FUNCTION ---
def downsample_sliced_data(tsv_path, start_index, end_index, original_hz=1000):
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        if 'SC' in df.columns:
            eda_signal = df['SC'].values
        elif df.shape[1] >= 3:
            eda_signal = df.iloc[:, 2].values
        else:
            return None

        sliced_eda_signal = eda_signal[start_index:end_index]
        window_size = int(original_hz / 1)
        downsampled_signal = [
            np.mean(sliced_eda_signal[i : i + window_size])
            for i in range(0, len(sliced_eda_signal), window_size)
        ]

        return pd.DataFrame({
            "Second": np.arange(len(downsampled_signal)),
            "SkinConductance": downsampled_signal
        })

    except Exception as e:
        print(f"    - ERROR: {e}")
        return None

# --- 4. MAIN ---
def main():
    os.makedirs(OUTPUT_SLICED_DIR, exist_ok=True)

    print("--- STAGE 1: Finding Master Timings ---")
    movie_timings = {}

    for movie in MOVIES_TO_PROCESS:
        movie_folder_path = os.path.join(RAW_SORTED_DIR, movie)
        if not os.path.exists(movie_folder_path):
            print(f"  - WARNING: Folder for '{movie}' not found. Skipping.")
            continue

        file_list = glob.glob(os.path.join(movie_folder_path, '*_physio*.tsv'))
        if not file_list:
            print(f"  - WARNING: No physio files found for '{movie}'.")
            continue

        reference_file = file_list[0]
        print(f"  - {movie}: using {os.path.basename(reference_file)}")

        start_idx, end_idx = find_master_movie_indices(reference_file)
        if start_idx is not None:
            duration_sec = (end_idx - start_idx) / 1000.0
            print(f"    -> {start_idx} to {end_idx} ({duration_sec:.1f}s)")
            movie_timings[movie] = (start_idx, end_idx)
        else:
            print(f"    -> ERROR: Could not find triggers.")

    print("\n--- STAGE 2: Processing all files ---")
    for movie_name, timings in movie_timings.items():
        start_idx, end_idx = timings
        print(f"\nProcessing: {movie_name}")

        movie_folder_path  = os.path.join(RAW_SORTED_DIR, movie_name)
        output_movie_folder = os.path.join(OUTPUT_SLICED_DIR, movie_name)
        os.makedirs(output_movie_folder, exist_ok=True)

        participant_files = glob.glob(
            os.path.join(movie_folder_path, '*_physio*.tsv')
        )

        for p_file in participant_files:
            filename = os.path.basename(p_file)
            print(f"  - {filename}")
            downsampled_data = downsample_sliced_data(p_file, start_idx, end_idx)
            if downsampled_data is not None:
                output_filename = f"downsampled_1Hz_SLICED_{filename.replace('.tsv', '.csv')}"
                output_path = os.path.join(output_movie_folder, output_filename)
                downsampled_data.to_csv(output_path, index=False)
                print(f"    -> Saved ({len(downsampled_data)} seconds)")
            else:
                print(f"    -> ERROR: Failed.")

    print("\n--- Complete ---")

if __name__ == "__main__":
    main()
