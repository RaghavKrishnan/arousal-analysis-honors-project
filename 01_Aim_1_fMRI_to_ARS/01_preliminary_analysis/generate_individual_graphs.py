"""
generate_individual_graphs.py
Location: 02_preliminary_analysis/
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
DATA_DIR       = '../data/output_sorted_by_movie/'
ANNOTATION_DIR = '../data/annotation_files/'
OUTPUT_DIR     = '../results/individual_graphs/'


# =============================================================================
# UTILITIES
# =============================================================================
def extract_participant_id(filepath):
    filename = os.path.basename(filepath)
    match    = re.match(r'(sub-\d+)', filename)
    if match:
        return match.group(1)
    return filename.split('_')[0]


def detect_trigger_column(tsv_path):
    try:
        df = pd.read_csv(tsv_path, sep='\t')
    except Exception as e:
        print(f"      ERROR: {e}")
        return None

    best_col_idx  = None
    best_duration = 0

    for col_idx in range(len(df.columns)):
        col         = df.iloc[:, col_idx]
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


def find_movie_indices(tsv_path, trigger_col_idx):
    try:
        df      = pd.read_csv(tsv_path, sep='\t')
        trigger = df.iloc[:, trigger_col_idx]
    except Exception as e:
        print(f"      ERROR: {e}")
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


def extract_participant_sc(tsv_path, start_idx, end_idx):
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        if 'SC' in df.columns:
            eda_raw = df['SC'].values
        elif df.shape[1] >= 3:
            eda_raw = df.iloc[:, 2].values
        else:
            return None

        sliced = eda_raw[start_idx:end_idx]
        return [
            np.mean(sliced[i : i + 1000])
            for i in range(0, len(sliced), 1000)
        ]
    except Exception as e:
        print(f"      ERROR: {e}")
        return None


def load_intense_emotion(movie_name):
    tsv_path  = os.path.join(ANNOTATION_DIR, f'Annot_{movie_name}_stim.tsv')
    json_path = os.path.join(ANNOTATION_DIR, f'Annot_{movie_name}_stim.json')

    if not os.path.exists(tsv_path) or not os.path.exists(json_path):
        print(f"    WARNING: Annotation not found for {movie_name}")
        return None

    try:
        annotations_df = pd.read_csv(tsv_path, sep='\t', header=None)
        with open(json_path, 'r') as f:
            item_labels = json.load(f)
        annotations_df.columns = item_labels['Columns']
        if 'IntenseEmotion' not in annotations_df.columns:
            print(f"    WARNING: 'IntenseEmotion' not found. Available: {list(annotations_df.columns)}")
            return None
        return annotations_df['IntenseEmotion'].values
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def generate_graph(movie_name, participant_id, emotion_data, sc_data,
                   r_val, p_val, save_path):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    time_sec = np.arange(len(emotion_data))

    color_emotion = 'tab:red'
    ax1.set_xlabel('Time (Seconds)', fontsize=12)
    ax1.set_ylabel('Intense Emotion', color=color_emotion, fontsize=12)
    ax1.plot(time_sec, emotion_data, color=color_emotion, linewidth=1.8)
    ax1.tick_params(axis='y', labelcolor=color_emotion)

    ax2 = ax1.twinx()
    color_sc = 'tab:blue'
    ax2.set_ylabel('Skin Conductance (μS)', color=color_sc, fontsize=12)
    ax2.plot(time_sec, sc_data, color=color_sc, linewidth=1.8)
    ax2.tick_params(axis='y', labelcolor=color_sc)

    p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
    plt.title(
        f'{movie_name} | {participant_id}:  '
        f'Intense Emotion vs. Skin Conductance  '
        f'(r = {r_val:.3f}, p {p_str})',
        fontsize=11, pad=10
    )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: '{DATA_DIR}' not found. Run from inside 02_preliminary_analysis/")
        return

    movie_folders = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    print(f"Found {len(movie_folders)} movies: {', '.join(movie_folders)}\n")

    all_results   = []
    total_graphs  = 0
    total_skipped = 0

    for movie in movie_folders:
        print(f"\n{'='*60}\n  MOVIE: {movie}\n{'='*60}")

        movie_output_dir = os.path.join(OUTPUT_DIR, movie)
        os.makedirs(movie_output_dir, exist_ok=True)

        emotion_signal = load_intense_emotion(movie)
        if emotion_signal is None:
            print(f"  -> SKIPPING: no annotation.")
            continue

        movie_folder      = os.path.join(DATA_DIR, movie)
        participant_files = sorted(glob.glob(
            os.path.join(movie_folder, '*physio*.tsv')
        ))

        if not participant_files:
            print(f"  WARNING: No physio files found.")
            continue

        print(f"  {len(participant_files)} participants.")

        ref_file    = participant_files[0]
        trigger_col = detect_trigger_column(ref_file)
        if trigger_col is None:
            print(f"  ERROR: Trigger column not found.")
            continue

        start_idx, end_idx = find_movie_indices(ref_file, trigger_col)
        if start_idx is None:
            print(f"  ERROR: Movie window not found.")
            continue

        duration_sec = (end_idx - start_idx) / 1000.0
        print(f"  Trigger col: {trigger_col} | Window: {start_idx}–{end_idx} ({duration_sec:.0f}s)")

        for p_file in participant_files:
            participant_id = extract_participant_id(p_file)
            sc_signal      = extract_participant_sc(p_file, start_idx, end_idx)

            if sc_signal is None:
                print(f"    {participant_id}: SKIPPED (SC extraction failed)")
                total_skipped += 1
                continue

            min_len         = min(len(emotion_signal), len(sc_signal))
            emotion_aligned = emotion_signal[:min_len]
            sc_aligned      = np.array(sc_signal[:min_len])

            if np.nanstd(sc_aligned) == 0 or np.nanstd(emotion_aligned) == 0:
                print(f"    {participant_id}: SKIPPED (flat signal)")
                total_skipped += 1
                continue

            sc_aligned      = np.where(np.isnan(sc_aligned),
                                       np.nanmean(sc_aligned), sc_aligned)
            emotion_aligned = np.where(np.isnan(emotion_aligned),
                                       np.nanmean(emotion_aligned), emotion_aligned)

            r_val, p_val = pearsonr(emotion_aligned, sc_aligned)

            save_path = os.path.join(movie_output_dir,
                                     f"{participant_id}_dual_axis.png")
            generate_graph(movie, participant_id,
                           emotion_aligned, sc_aligned,
                           r_val, p_val, save_path)

            print(f"    {participant_id}: r = {r_val:.3f}, p = {p_val:.4f} [{min_len}s]")

            all_results.append({
                "Movie":        movie,
                "Participant":  participant_id,
                "Duration_Sec": min_len,
                "Pearson_r":    round(r_val, 4),
                "P_Value":      round(p_val, 6),
                "Significant":  "Yes" if p_val < 0.05 else "No"
            })
            total_graphs += 1

    print(f"\n{'='*60}\nDONE\n{'='*60}")
    print(f"Graphs generated: {total_graphs}")
    print(f"Skipped:          {total_skipped}")

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(
            os.path.join(OUTPUT_DIR, "individual_correlation_summary.csv"),
            index=False
        )
        print(f"\nPer-movie summary (mean r across participants):")
        movie_summary = (summary_df
                         .groupby("Movie")["Pearson_r"]
                         .agg(["mean", "min", "max", "count"])
                         .round(3))
        movie_summary.columns = ["Mean_r", "Min_r", "Max_r", "N_participants"]
        print(movie_summary.to_string())

        n_sig = (summary_df["Significant"] == "Yes").sum()
        print(f"\nSignificant (p < 0.05): {n_sig}/{len(all_results)}")

if __name__ == "__main__":
    main()
