import pandas as pd
import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- 1. CONFIGURATION ---
RAW_SORTED_DIR = 'output_sorted_by_movie/'
ANNOTATION_DIR = 'annotation_files/'
OUTPUT_DIR = 'emotion_sc_correlations_exact/'

# Map each movie to its specific trigger column index based on your previous pipeline
MOVIES_CONFIG = {
    "Sintel": 8,
    "Payload": 8,
    "BigBuckBunny": 7,
    "Chatter": 7,
    "TearsOfSteel": 7,
    "TheSecretNumber": 7
}

# --- 2. TRIGGER & PHYSIO PROCESSING ---
def find_master_movie_indices(tsv_path, trigger_column_index):
    """Finds the start and end index of the 0->5 / 5->0 trigger pair."""
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        if trigger_column_index >= len(df.columns):
            return None, None
        trigger = df.iloc[:, trigger_column_index]
    except Exception:
        return None, None

    starts = (trigger == 5) & (trigger.shift(1) == 0)
    ends = (trigger == 0) & (trigger.shift(1) == 5)
    
    start_indices = starts[starts].index
    end_indices = ends[ends].index

    if len(start_indices) == 0 or len(end_indices) == 0:
        return None, None

    longest_duration = 0
    best_start, best_end = -1, -1

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
        return None, None
    return best_start, best_end

def process_and_average_sc(movie_name, trigger_col_index):
    """Finds triggers, slices, downsamples, and averages SC for all participants."""
    movie_folder = os.path.join(RAW_SORTED_DIR, movie_name)
    if not os.path.exists(movie_folder):
        print(f"    - WARNING: Folder for '{movie_name}' not found in raw data.")
        return None

    participant_files = glob.glob(os.path.join(movie_folder, '*_physio*.tsv'))
    if not participant_files:
        return None

    # Find master indices using the first file as a reference
    start_idx, end_idx = find_master_movie_indices(participant_files[0], trigger_col_index)
    if start_idx is None:
        print(f"    - ERROR: Could not find valid triggers for {movie_name}.")
        return None

    all_participants_downsampled_sc = []

    for p_file in participant_files:
        try:
            df = pd.read_csv(p_file, sep='\t')
            if 'SC' in df.columns:
                eda_signal = df['SC'].values
            elif df.shape[1] >= 3:
                eda_signal = df.iloc[:, 2].values
            else:
                continue
                
            # EXACT SLICE USING TRIGGERS
            sliced_eda_signal = eda_signal[start_idx:end_idx]
            
            # DOWNSAMPLE TO 1HZ
            window_size = 1000
            downsampled_signal = [
                np.mean(sliced_eda_signal[i : i + window_size])
                for i in range(0, len(sliced_eda_signal), window_size)
            ]
            all_participants_downsampled_sc.append(downsampled_signal)
        except Exception:
            continue

    if not all_participants_downsampled_sc:
        return None

    # Find min length to align and average
    min_len = min(len(sc) for sc in all_participants_downsampled_sc)
    truncated_sc = [sc[:min_len] for sc in all_participants_downsampled_sc]
    mean_sc = np.mean(np.array(truncated_sc), axis=0)
    
    return mean_sc

# --- 3. ANNOTATION PROCESSING ---
def get_intense_emotion(movie_name):
    """Loads the annotation file and returns the IntenseEmotion series."""
    tsv_path = os.path.join(ANNOTATION_DIR, f'Annot_{movie_name}_stim.tsv')
    json_path = os.path.join(ANNOTATION_DIR, f'Annot_{movie_name}_stim.json')
    
    if not os.path.exists(tsv_path) or not os.path.exists(json_path):
        return None

    try:
        annotations_df = pd.read_csv(tsv_path, sep='\t', header=None)
        with open(json_path, 'r') as f:
            item_labels = json.load(f)
        
        annotations_df.columns = item_labels['Columns']
        if 'IntenseEmotion' in annotations_df.columns:
            return annotations_df['IntenseEmotion'].values
        return None
    except Exception:
        return None

# --- 4. VISUALIZATION ---
def generate_plots(movie_name, emotion_data, sc_data, correlation, p_val):
    """Generates dual-axis and scatter plots for the exact alignment."""
    # 1. Dual-Axis Time Series
    fig, ax1 = plt.subplots(figsize=(12, 6))
    time_seconds = np.arange(len(emotion_data))
    
    color1 = 'tab:red'
    ax1.set_xlabel('Time (Seconds)')
    ax1.set_ylabel('Intense Emotion', color=color1)
    ax1.plot(time_seconds, emotion_data, color=color1, linewidth=2, label='Intense Emotion')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel('Mean Skin Conductance ($\mu$S)', color=color2)
    ax2.plot(time_seconds, sc_data, color=color2, linewidth=2, label='Mean SC')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title(f'{movie_name}: Intense Emotion vs. Skin Conductance (r={correlation:.3f}, p={p_val:.3f})')
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{movie_name}_dual_axis.png"), dpi=300)
    plt.close()

    # 2. Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(emotion_data, sc_data, alpha=0.5, color='purple')
    z = np.polyfit(emotion_data, sc_data, 1)
    p = np.poly1d(z)
    plt.plot(emotion_data, p(emotion_data), "k--", alpha=0.8)
    
    plt.title(f'{movie_name} Scatter: Intense Emotion vs. SC')
    plt.xlabel('Intense Emotion Score')
    plt.ylabel('Mean Skin Conductance ($\mu$S)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{movie_name}_scatter.png"), dpi=300)
    plt.close()

# --- 5. MAIN EXECUTION ---
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("--- Starting End-to-End Exact Alignment & Correlation Analysis ---")
    results = []
    
    global_emotion = []
    global_sc = []

    for movie, trigger_col in MOVIES_CONFIG.items():
        print(f"\nProcessing: {movie} (Using Trigger Col: {trigger_col})...")
        
        sc_signal = process_and_average_sc(movie, trigger_col)
        emotion_signal = get_intense_emotion(movie)
        
        if emotion_signal is not None and sc_signal is not None:
            # Handle minor length mismatches due to downsampling math vs annotation lengths
            min_length = min(len(emotion_signal), len(sc_signal))
            emotion_aligned = emotion_signal[:min_length]
            sc_aligned = sc_signal[:min_length]
            
            global_emotion.extend(emotion_aligned)
            global_sc.extend(sc_aligned)
            
            r_val, p_val = pearsonr(emotion_aligned, sc_aligned)
            generate_plots(movie, emotion_aligned, sc_aligned, r_val, p_val)
            
            print(f"    - Aligned Length: {min_length} seconds.")
            print(f"    - Correlation (r): {r_val:.4f} (p: {p_val:.4f})")
            
            results.append({
                "Movie": movie,
                "Duration_Sec": min_length,
                "Pearson_r": r_val,
                "P_Value": p_val
            })
        else:
            print(f"    - Skipping {movie} due to missing or invalid data.")

    if global_emotion and global_sc:
        global_r, global_p = pearsonr(global_emotion, global_sc)
        print("\n--- GLOBAL CORRELATION (All Valid Movies) ---")
        print(f"Total Aligned Datapoints (Seconds): {len(global_emotion)}")
        print(f"Global Pearson (r): {global_r:.4f} (p-value: {global_p:.4f})")
        
        results.append({
            "Movie": "GLOBAL_ALL_MOVIES",
            "Duration_Sec": len(global_emotion),
            "Pearson_r": global_r,
            "P_Value": global_p
        })
        generate_plots("GLOBAL_ALL_MOVIES", np.array(global_emotion), np.array(global_sc), global_r, global_p)

    if results:
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(OUTPUT_DIR, "correlation_summary_exact.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary report saved to: {summary_path}")

if __name__ == "__main__":
    main()