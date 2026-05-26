"""
generate_individual_graphs.py
==============================
Generates one dual-axis Intense Emotion vs. Skin Conductance graph
for EVERY INDIVIDUAL PARTICIPANT across EVERY MOVIE.

No averaging — each participant's raw SC timecourse is plotted directly
against the consensus IntenseEmotion annotation for that movie.

OUTPUT STRUCTURE:
    individual_graphs/
    ├── AfterTheRain/
    │   ├── sub-01_dual_axis.png
    │   ├── sub-03_dual_axis.png
    │   └── ...
    ├── BigBuckBunny/
    │   ├── sub-01_dual_axis.png
    │   └── ...
    └── ...
    individual_correlation_summary.csv  ← r and p for every participant x movie

HOW TO RUN:
    1. Place this script in 'Honors Work Arousal Analysis/' alongside
       annotation_files/ and output_sorted_by_movie/
    2. Activate your venv:   venv\\Scripts\\activate
    3. Run:                  python generate_individual_graphs.py
"""

import pandas as pd
import numpy as np
import os
import glob
import json
import re
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR       = 'output_sorted_by_movie/'
ANNOTATION_DIR = 'annotation_files/'
OUTPUT_DIR     = 'individual_graphs/'


# =============================================================================
# UTILITY — EXTRACT PARTICIPANT ID FROM FILENAME
# =============================================================================
def extract_participant_id(filepath):
    """
    Pulls the subject ID (e.g. 'sub-01') out of a filename like:
        sub-01_ses-3_task-ToClaireFromSonny_physio.tsv
    Returns 'sub-01', or the full filename stem if pattern not found.
    """
    filename = os.path.basename(filepath)
    match = re.match(r'(sub-\d+)', filename)
    if match:
        return match.group(1)
    # Fallback: use everything before the first underscore
    return filename.split('_')[0]


# =============================================================================
# TRIGGER DETECTION — AUTO-DETECT WHICH COLUMN IS THE MOVIE TRIGGER
# =============================================================================
def detect_trigger_column(tsv_path):
    """
    Scans every column looking for the one that transitions 0->5 (movie start)
    and 5->0 (movie end). Returns the column index with the longest such window,
    or None if not found.
    """
    try:
        df = pd.read_csv(tsv_path, sep='\t')
    except Exception as e:
        print(f"      ERROR reading file for trigger detection: {e}")
        return None

    best_col_idx  = None
    best_duration = 0

    for col_idx in range(len(df.columns)):
        col        = df.iloc[:, col_idx]
        unique_vals = set(col.unique())

        if 0 not in unique_vals or 5 not in unique_vals:
            continue

        starts = (col == 5) & (col.shift(1) == 0)
        ends   = (col == 0) & (col.shift(1) == 5)

        start_indices = starts[starts].index.tolist()
        end_indices   = ends[ends].index.tolist()

        if not start_indices or not end_indices:
            continue

        for s in start_indices:
            valid_ends = [e for e in end_indices if e > s]
            if valid_ends:
                duration = valid_ends[0] - s
                if duration > best_duration:
                    best_duration = duration
                    best_col_idx  = col_idx

    return best_col_idx


# =============================================================================
# FIND MOVIE START AND END INDICES FROM TRIGGER COLUMN
# =============================================================================
def find_movie_indices(tsv_path, trigger_col_idx):
    """
    Using the detected trigger column, returns the (start, end) sample indices
    for the longest 0->5...5->0 window. Returns (None, None) on failure.
    """
    try:
        df      = pd.read_csv(tsv_path, sep='\t')
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
# EXTRACT ONE PARTICIPANT'S SC SIGNAL FOR ONE MOVIE
# =============================================================================
def extract_participant_sc(tsv_path, start_idx, end_idx):
    """
    Reads a single participant's physio file, slices the SC signal to the
    movie window, and downsamples from 1000 Hz to 1 Hz.

    Returns a 1D list of floats (one value per second), or None on failure.
    """
    try:
        df = pd.read_csv(tsv_path, sep='\t')

        # Find the SC column — named 'SC' or fall back to 3rd column (index 2)
        if 'SC' in df.columns:
            eda_raw = df['SC'].values
        elif df.shape[1] >= 3:
            eda_raw = df.iloc[:, 2].values
        else:
            print(f"      SKIP: Cannot find SC column in {os.path.basename(tsv_path)}")
            return None

        # Slice to movie window
        sliced = eda_raw[start_idx:end_idx]

        # Downsample: average every 1000 samples (1000 Hz -> 1 Hz)
        downsampled = [
            np.mean(sliced[i : i + 1000])
            for i in range(0, len(sliced), 1000)
        ]

        return downsampled

    except Exception as e:
        print(f"      ERROR processing {os.path.basename(tsv_path)}: {e}")
        return None


# =============================================================================
# LOAD INTENSE EMOTION ANNOTATION FOR ONE MOVIE
# =============================================================================
def load_intense_emotion(movie_name):
    """
    Loads the IntenseEmotion consensus annotation timeseries from
    annotation_files/Annot_{movie_name}_stim.tsv

    Returns a 1D numpy array or None on failure.
    """
    tsv_path  = os.path.join(ANNOTATION_DIR, f'Annot_{movie_name}_stim.tsv')
    json_path = os.path.join(ANNOTATION_DIR, f'Annot_{movie_name}_stim.json')

    if not os.path.exists(tsv_path):
        print(f"    WARNING: Annotation TSV not found: {tsv_path}")
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
            print(f"    WARNING: 'IntenseEmotion' column not found.")
            print(f"    Available columns: {list(annotations_df.columns)}")
            return None

        return annotations_df['IntenseEmotion'].values

    except Exception as e:
        print(f"    ERROR loading annotation for {movie_name}: {e}")
        return None


# =============================================================================
# GENERATE ONE DUAL-AXIS GRAPH FOR ONE PARTICIPANT IN ONE MOVIE
# =============================================================================
def generate_graph(movie_name, participant_id, emotion_data, sc_data, r_val, p_val, save_path):
    """
    Produces a dual-axis time series graph:
      Red  (left axis):  Intense Emotion annotation
      Blue (right axis): This participant's Skin Conductance
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    time_sec = np.arange(len(emotion_data))

    # Left axis — Intense Emotion
    color_emotion = 'tab:red'
    ax1.set_xlabel('Time (Seconds)', fontsize=12)
    ax1.set_ylabel('Intense Emotion', color=color_emotion, fontsize=12)
    ax1.plot(time_sec, emotion_data,
             color=color_emotion, linewidth=1.8, label='Intense Emotion')
    ax1.tick_params(axis='y', labelcolor=color_emotion)

    # Right axis — Skin Conductance
    ax2 = ax1.twinx()
    color_sc = 'tab:blue'
    ax2.set_ylabel('Skin Conductance (μS)', color=color_sc, fontsize=12)
    ax2.plot(time_sec, sc_data,
             color=color_sc, linewidth=1.8, label='SC')
    ax2.tick_params(axis='y', labelcolor=color_sc)

    # Title
    p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
    plt.title(
        f'{movie_name} | {participant_id}:  '
        f'Intense Emotion vs. Skin Conductance  '
        f'(r = {r_val:.3f}, p {p_str})',
        fontsize=11,
        pad=10
    )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # dpi=150 instead of 300 — keeps file sizes reasonable across 400+ graphs
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Discover all movie folders ---
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: '{DATA_DIR}' not found.")
        print("Run this script from inside 'Honors Work Arousal Analysis/'")
        return

    movie_folders = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    if not movie_folders:
        print(f"ERROR: No movie subfolders found in '{DATA_DIR}'.")
        return

    print(f"Found {len(movie_folders)} movies: {', '.join(movie_folders)}")
    print(f"Output folder: '{OUTPUT_DIR}'\n")

    all_results = []
    total_graphs = 0
    total_skipped = 0

    # -------------------------------------------------------------------------
    # LOOP OVER EVERY MOVIE
    # -------------------------------------------------------------------------
    for movie in movie_folders:
        print(f"\n{'='*60}")
        print(f"  MOVIE: {movie}")
        print(f"{'='*60}")

        # Create output subfolder for this movie
        movie_output_dir = os.path.join(OUTPUT_DIR, movie)
        os.makedirs(movie_output_dir, exist_ok=True)

        # Load annotation once per movie (same for all participants)
        emotion_signal = load_intense_emotion(movie)
        if emotion_signal is None:
            print(f"  -> SKIPPING entire movie: no annotation data.")
            continue

        # Find all participant files for this movie
        movie_folder      = os.path.join(DATA_DIR, movie)
        participant_files = sorted(glob.glob(
            os.path.join(movie_folder, '*physio*.tsv')
        ))

        if not participant_files:
            print(f"  WARNING: No physio files found. Skipping.")
            continue

        print(f"  {len(participant_files)} participants found.")

        # Detect trigger column once using the first participant's file
        ref_file    = participant_files[0]
        trigger_col = detect_trigger_column(ref_file)

        if trigger_col is None:
            print(f"  ERROR: Could not detect trigger column. Skipping movie.")
            continue

        # Get master movie window indices
        start_idx, end_idx = find_movie_indices(ref_file, trigger_col)

        if start_idx is None:
            print(f"  ERROR: Could not find movie window. Skipping movie.")
            continue

        duration_sec = (end_idx - start_idx) / 1000.0
        print(f"  Trigger col: {trigger_col} | "
              f"Movie window: {start_idx}–{end_idx} ({duration_sec:.0f}s)")

        # ---------------------------------------------------------------------
        # LOOP OVER EVERY PARTICIPANT IN THIS MOVIE
        # ---------------------------------------------------------------------
        for p_file in participant_files:
            participant_id = extract_participant_id(p_file)

            sc_signal = extract_participant_sc(p_file, start_idx, end_idx)

            if sc_signal is None:
                print(f"    {participant_id}: SKIPPED (SC extraction failed)")
                total_skipped += 1
                continue

            # Align lengths
            min_len         = min(len(emotion_signal), len(sc_signal))
            emotion_aligned = emotion_signal[:min_len]
            sc_aligned      = np.array(sc_signal[:min_len])

            # Skip if signal is flat (all NaN or zero variance — can't correlate)
            if np.nanstd(sc_aligned) == 0 or np.nanstd(emotion_aligned) == 0:
                print(f"    {participant_id}: SKIPPED (flat signal, no variance)")
                total_skipped += 1
                continue

            # Replace any NaN values with the mean of the signal
            # (rare edge case from bad physio recording windows)
            sc_aligned      = np.where(np.isnan(sc_aligned),
                                       np.nanmean(sc_aligned), sc_aligned)
            emotion_aligned = np.where(np.isnan(emotion_aligned),
                                       np.nanmean(emotion_aligned), emotion_aligned)

            # Compute correlation
            r_val, p_val = pearsonr(emotion_aligned, sc_aligned)

            # Save graph
            save_path = os.path.join(
                movie_output_dir,
                f"{participant_id}_dual_axis.png"
            )
            generate_graph(movie, participant_id,
                           emotion_aligned, sc_aligned,
                           r_val, p_val, save_path)

            print(f"    {participant_id}: r = {r_val:.3f}, p = {p_val:.4f}  "
                  f"[{min_len}s] -> saved")

            all_results.append({
                "Movie":          movie,
                "Participant":    participant_id,
                "Duration_Sec":   min_len,
                "Pearson_r":      round(r_val, 4),
                "P_Value":        round(p_val, 6),
                "Significant":    "Yes" if p_val < 0.05 else "No"
            })

            total_graphs += 1

    # -------------------------------------------------------------------------
    # SAVE SUMMARY
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"Graphs generated: {total_graphs}")
    print(f"Skipped:          {total_skipped}")

    if all_results:
        summary_df   = pd.DataFrame(all_results)
        summary_path = os.path.join(OUTPUT_DIR, "individual_correlation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary CSV:      {summary_path}")

        # Quick overview stats
        print(f"\nPer-movie summary (mean r across participants):")
        movie_summary = (summary_df
                         .groupby("Movie")["Pearson_r"]
                         .agg(["mean", "min", "max", "count"])
                         .round(3))
        movie_summary.columns = ["Mean_r", "Min_r", "Max_r", "N_participants"]
        print(movie_summary.to_string())

        n_sig = (summary_df["Significant"] == "Yes").sum()
        print(f"\nSignificant correlations (p < 0.05): "
              f"{n_sig}/{len(all_results)} participant-movie pairs")
    else:
        print("No graphs were generated. Check your folder paths and file names.")


if __name__ == "__main__":
    main()
