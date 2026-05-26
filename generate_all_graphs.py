"""
generate_all_graphs.py
======================
Generates Intense Emotion vs. Mean Skin Conductance dual-axis plots
for ALL movies in output_sorted_by_movie/.

HOW TO RUN:
    1. Place this script in your 'Honors Work Arousal Analysis' folder
       (same level as annotation_files/ and output_sorted_by_movie/)
    2. Open a terminal / command prompt in that folder
    3. Run:  python generate_all_graphs.py

OUTPUT:
    A folder called 'all_movies_graphs/' will be created containing:
      - One dual-axis PNG plot per movie
      - correlation_summary.csv with r and p values for all movies
"""

import pandas as pd
import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# =============================================================================
# CONFIGURATION — edit these paths if needed
# =============================================================================
DATA_DIR       = 'output_sorted_by_movie/'   # folder with all 14 movie subfolders
ANNOTATION_DIR = 'annotation_files/'          # folder with Annot_Movie_stim.tsv files
OUTPUT_DIR     = 'all_movies_graphs/'         # where graphs and CSV will be saved


# =============================================================================
# STEP 1 — AUTO-DETECT TRIGGER COLUMN
# =============================================================================
def detect_trigger_column(tsv_path):
    """
    Scans every column in a physio TSV file looking for the one that contains
    the movie trigger signal (transitions between 0 and 5).

    The trigger works like this:
      - Signal is 0 before the movie starts
      - Jumps to 5 when the movie begins (0 -> 5 = movie start)
      - Drops back to 0 when the movie ends  (5 -> 0 = movie end)

    Returns the column index of the trigger column, or None if not found.
    """
    try:
        df = pd.read_csv(tsv_path, sep='\t')
    except Exception as e:
        print(f"      ERROR reading file for trigger detection: {e}")
        return None

    best_col_idx = None
    best_duration = 0

    for col_idx in range(len(df.columns)):
        col = df.iloc[:, col_idx]

        # Only check columns that actually contain both 0 and 5
        unique_vals = set(col.unique())
        if 0 not in unique_vals or 5 not in unique_vals:
            continue

        # Find 0->5 (start) and 5->0 (end) transitions
        starts = (col == 5) & (col.shift(1) == 0)
        ends   = (col == 0) & (col.shift(1) == 5)

        start_indices = starts[starts].index.tolist()
        end_indices   = ends[ends].index.tolist()

        if not start_indices or not end_indices:
            continue

        # Find the longest contiguous 0->5...5->0 segment in this column
        for s in start_indices:
            valid_ends = [e for e in end_indices if e > s]
            if valid_ends:
                duration = valid_ends[0] - s
                if duration > best_duration:
                    best_duration = duration
                    best_col_idx  = col_idx

    return best_col_idx


# =============================================================================
# STEP 2 — FIND MOVIE START AND END SAMPLE INDICES
# =============================================================================
def find_movie_indices(tsv_path, trigger_col_idx):
    """
    Using the already-detected trigger column, finds the exact sample index
    where the movie starts and where it ends.

    Returns (start_index, end_index) or (None, None) on failure.
    """
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        trigger = df.iloc[:, trigger_col_idx]
    except Exception as e:
        print(f"      ERROR reading trigger column: {e}")
        return None, None

    starts = (trigger == 5) & (trigger.shift(1) == 0)
    ends   = (trigger == 0) & (trigger.shift(1) == 5)

    start_indices = starts[starts].index.tolist()
    end_indices   = ends[ends].index.tolist()

    if not start_indices or not end_indices:
        return None, None

    best_start, best_end, best_duration = -1, -1, 0

    for s in start_indices:
        valid_ends = [e for e in end_indices if e > s]
        if valid_ends:
            duration = valid_ends[0] - s
            if duration > best_duration:
                best_duration = duration
                best_start    = s
                best_end      = valid_ends[0]

    if best_start == -1:
        return None, None

    return best_start, best_end


# =============================================================================
# STEP 3 — PROCESS ALL PARTICIPANTS' SC DATA FOR ONE MOVIE
# =============================================================================
def process_sc_for_movie(movie_name):
    """
    For a given movie:
      1. Finds all participant physio TSV files
      2. Auto-detects the trigger column from the first file
      3. Finds the movie window (start and end indices)
      4. For every participant: slices the SC signal to the movie window,
         then downsamples from 1000 Hz to 1 Hz (one value per second)
      5. Averages across all participants

    Returns a 1D numpy array of mean SC (one value per second), or None on failure.
    """
    movie_folder = os.path.join(DATA_DIR, movie_name)
    participant_files = sorted(glob.glob(os.path.join(movie_folder, '*physio*.tsv')))

    if not participant_files:
        print(f"    WARNING: No physio TSV files found in '{movie_folder}'")
        return None

    print(f"    Found {len(participant_files)} participant files.")

    # --- Detect trigger column from the first file ---
    ref_file = participant_files[0]
    print(f"    Auto-detecting trigger column using: {os.path.basename(ref_file)}")
    trigger_col = detect_trigger_column(ref_file)

    if trigger_col is None:
        print(f"    ERROR: Could not detect trigger column. Skipping {movie_name}.")
        return None

    print(f"    Trigger column: index {trigger_col}")

    # --- Get master start/end indices ---
    start_idx, end_idx = find_movie_indices(ref_file, trigger_col)

    if start_idx is None:
        print(f"    ERROR: Could not find valid trigger window. Skipping {movie_name}.")
        return None

    duration_sec = (end_idx - start_idx) / 1000.0
    print(f"    Movie window: samples {start_idx} to {end_idx} = {duration_sec:.1f} seconds")

    # --- Process every participant ---
    all_sc = []

    for p_file in participant_files:
        try:
            df = pd.read_csv(p_file, sep='\t')

            # Find the SC (skin conductance) column
            # Priority: look for a column literally named 'SC',
            # otherwise fall back to the 3rd column (index 2) which is
            # the standard position in Emo-FilM physio files.
            if 'SC' in df.columns:
                eda_raw = df['SC'].values
            elif df.shape[1] >= 3:
                eda_raw = df.iloc[:, 2].values
            else:
                print(f"      SKIP: Cannot find SC column in {os.path.basename(p_file)}")
                continue

            # Slice to the movie window using master indices
            sliced = eda_raw[start_idx:end_idx]

            # Downsample: average every 1000 consecutive samples -> 1 second
            # (the physio data is recorded at 1000 Hz)
            downsampled = [
                np.mean(sliced[i : i + 1000])
                for i in range(0, len(sliced), 1000)
            ]
            all_sc.append(downsampled)

        except Exception as e:
            print(f"      SKIP: Error in {os.path.basename(p_file)}: {e}")
            continue

    if not all_sc:
        print(f"    ERROR: No valid SC data extracted for {movie_name}.")
        return None

    # --- Average across participants ---
    # Trim to the shortest participant to handle minor length differences
    min_len = min(len(sc) for sc in all_sc)
    mean_sc = np.mean(np.array([sc[:min_len] for sc in all_sc]), axis=0)

    print(f"    Successfully processed {len(all_sc)}/{len(participant_files)} participants.")
    return mean_sc


# =============================================================================
# STEP 4 — LOAD ANNOTATION (IntenseEmotion)
# =============================================================================
def load_intense_emotion(movie_name):
    """
    Loads the IntenseEmotion consensus annotation timeseries for a movie
    from annotation_files/Annot_{movie_name}_stim.tsv

    Returns a 1D numpy array (one value per second), or None on failure.
    """
    tsv_path  = os.path.join(ANNOTATION_DIR, f'Annot_{movie_name}_stim.tsv')
    json_path = os.path.join(ANNOTATION_DIR, f'Annot_{movie_name}_stim.json')

    if not os.path.exists(tsv_path):
        print(f"    WARNING: Annotation file not found: {tsv_path}")
        return None
    if not os.path.exists(json_path):
        print(f"    WARNING: Annotation JSON not found: {json_path}")
        return None

    try:
        annotations_df = pd.read_csv(tsv_path, sep='\t', header=None)
        with open(json_path, 'r') as f:
            item_labels = json.load(f)

        annotations_df.columns = item_labels['Columns']

        if 'IntenseEmotion' not in annotations_df.columns:
            # Print available columns so you can debug if annotation name differs
            print(f"    WARNING: 'IntenseEmotion' not found.")
            print(f"    Available columns: {list(annotations_df.columns)}")
            return None

        return annotations_df['IntenseEmotion'].values

    except Exception as e:
        print(f"    ERROR loading annotation for {movie_name}: {e}")
        return None


# =============================================================================
# STEP 5 — GENERATE THE DUAL-AXIS GRAPH
# =============================================================================
def generate_dual_axis_plot(movie_name, emotion_data, sc_data, r_val, p_val):
    """
    Produces the dual-axis time series graph matching the TearsOfSteel style:
      - Red line (left axis):  Intense Emotion
      - Blue line (right axis): Mean Skin Conductance (μS)
      - Title shows movie name, r, and p values
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    time_sec = np.arange(len(emotion_data))

    # --- Left axis: Intense Emotion (red) ---
    color_emotion = 'tab:red'
    ax1.set_xlabel('Time (Seconds)', fontsize=12)
    ax1.set_ylabel('Intense Emotion', color=color_emotion, fontsize=12)
    ax1.plot(time_sec, emotion_data,
             color=color_emotion, linewidth=2, label='Intense Emotion')
    ax1.tick_params(axis='y', labelcolor=color_emotion)

    # --- Right axis: Mean SC (blue) ---
    ax2 = ax1.twinx()
    color_sc = 'tab:blue'
    ax2.set_ylabel('Mean Skin Conductance (μS)', color=color_sc, fontsize=12)
    ax2.plot(time_sec, sc_data,
             color=color_sc, linewidth=2, label='Mean SC')
    ax2.tick_params(axis='y', labelcolor=color_sc)

    # --- Title: format p-value neatly ---
    if p_val < 0.001:
        p_str = "< 0.001"
    else:
        p_str = f"{p_val:.3f}"

    plt.title(
        f'{movie_name}: Intense Emotion vs. Skin Conductance  '
        f'(r = {r_val:.3f}, p {p_str})',
        fontsize=13,
        pad=12
    )

    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"{movie_name}_dual_axis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    Graph saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Auto-discover all movie folders ---
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data folder '{DATA_DIR}' not found.")
        print("Make sure you are running this script from inside 'Honors Work Arousal Analysis/'")
        return

    movie_folders = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    if not movie_folders:
        print(f"ERROR: No subfolders found in '{DATA_DIR}'.")
        return

    print(f"Found {len(movie_folders)} movie(s): {', '.join(movie_folders)}")
    print(f"Output will be saved to: '{OUTPUT_DIR}'\n")

    results = []

    for movie in movie_folders:
        print(f"\n{'='*60}")
        print(f"  MOVIE: {movie}")
        print(f"{'='*60}")

        # --- Get skin conductance signal ---
        sc_signal = process_sc_for_movie(movie)

        # --- Get annotation signal ---
        emotion_signal = load_intense_emotion(movie)

        # --- Skip if either signal is missing ---
        if sc_signal is None:
            print(f"  -> SKIPPED: No SC data for {movie}.")
            continue
        if emotion_signal is None:
            print(f"  -> SKIPPED: No annotation data for {movie}.")
            continue

        # --- Align lengths ---
        # Annotation is at 1 Hz, SC is now at 1 Hz after downsampling.
        # Minor mismatches (1-2 seconds) happen due to rounding — trim to the shorter one.
        min_len = min(len(emotion_signal), len(sc_signal))
        emotion_aligned = emotion_signal[:min_len]
        sc_aligned      = sc_signal[:min_len]

        print(f"    Aligned length: {min_len} seconds")

        # --- Compute Pearson correlation ---
        r_val, p_val = pearsonr(emotion_aligned, sc_aligned)
        print(f"    Pearson r = {r_val:.4f},  p = {p_val:.6f}")

        # --- Generate graph ---
        generate_dual_axis_plot(movie, emotion_aligned, sc_aligned, r_val, p_val)

        results.append({
            "Movie":        movie,
            "Duration_Sec": min_len,
            "Pearson_r":    round(r_val, 4),
            "P_Value":      round(p_val, 6),
            "Significant":  "Yes" if p_val < 0.05 else "No"
        })

    # --- Print and save summary ---
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL MOVIES")
    print(f"{'='*60}")

    if results:
        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False))

        summary_path = os.path.join(OUTPUT_DIR, "correlation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary CSV saved to: {summary_path}")

        # Count significant results
        n_sig = summary_df["Significant"].value_counts().get("Yes", 0)
        print(f"Movies with significant correlation (p < 0.05): {n_sig}/{len(results)}")
    else:
        print("No movies were successfully processed.")
        print("Check that annotation_files/ and output_sorted_by_movie/ are present.")


if __name__ == "__main__":
    main()
