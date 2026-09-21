"""
eda_quality_control.py  (v4 - Robust Signal Processing Edition)
==============================================================
UPDATES IN THIS VERSION:
1. Anti-Aliasing: Implements a 2Hz Butterworth low-pass filter BEFORE downsampling 
   to eliminate high-frequency MRI-room noise from aliasing into the 1Hz signal.
2. Log-Transformed Fences: Zero-bounded, right-skewed metrics (smoothness, flatline) 
   are now log-transformed before modified z-score calculation, allowing the math 
   to correctly flag true non-responders (like sub-S05).
3. Debounce Logic: Step-change detection now includes a non-maximum suppression 
   (debounce) window so a single artifact isn't smeared into 50+ contiguous flags.
4. Calibrated Thresholds: Within-subject spike threshold raised from 5.0 to 20.0 
   to avoid flagging healthy, high-velocity physiological Skin Conductance Responses.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# =============================================================================
# CONFIGURATION
# =============================================================================
MOVIE_NAME = "Chatter"

DATA_DIR = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\data\output_sorted_by_movie'
OUTPUT_DIR = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\results\eda_quality_control'

EDA_COLUMN_INDEX = 2
KNOWN_TRIGGER_COLUMN = 7  

COHORT_MODZ_THRESHOLD = 3.5  
MODIFIED_ZSCORE_THRESHOLD = 20.0  # Raised to 20 to isolate true electrical artifacts, not physiological SCRs
STEP_CHANGE_WINDOW = 10           

ABS_FLOOR_US = 0.5
ABS_CEILING_US = 50.0
ABS_EXTREME_FRACTION_LIMIT = 0.10
TRUNCATION_FRACTION_LIMIT = 0.70

# =============================================================================
# SIGNAL PROCESSING UTILITIES
# =============================================================================
def _longest_valid_window(col_values):
    starts = np.where((col_values[1:] == 5) & (col_values[:-1] == 0))[0] + 1
    ends = np.where((col_values[1:] == 0) & (col_values[:-1] == 5))[0] + 1
    if len(starts) == 0 or len(ends) == 0:
        return None, None, 0
    insert_positions = np.searchsorted(ends, starts, side='right')
    valid_mask = insert_positions < len(ends)
    if not np.any(valid_mask):
        return None, None, 0
    matched_starts = starts[valid_mask]
    matched_ends = ends[insert_positions[valid_mask]]
    durations = matched_ends - matched_starts
    best_idx = np.argmax(durations)
    return int(matched_starts[best_idx]), int(matched_ends[best_idx]), int(durations[best_idx])

def detect_trigger_column(df):
    best_col_idx, best_duration = None, 0
    for col_idx in range(len(df.columns)):
        col_values = df.iloc[:, col_idx].values
        unique_vals = set(np.unique(col_values))
        if 0 not in unique_vals or 5 not in unique_vals:
            continue
        _, _, duration = _longest_valid_window(col_values)
        if duration > best_duration:
            best_duration = duration
            best_col_idx = col_idx
    return best_col_idx

def find_movie_window(df, trigger_col_idx):
    col_values = df[trigger_col_idx].values
    start_idx, end_idx, _ = _longest_valid_window(col_values)
    return start_idx, end_idx

def modified_zscore(x):
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - median) / mad

def apply_anti_aliasing_filter(data, fs=1000, cutoff=2.0):
    """Applies a 4th-order Butterworth low-pass filter to remove MRI-room electrical hum."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    # Use filtfilt for zero-phase distortion
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# =============================================================================
# RAW METRIC EXTRACTION
# =============================================================================
def extract_subject_metrics(eda_1hz):
    n = len(eda_1hz)
    deltas = np.diff(eda_1hz)

    frac_moving = np.mean(np.abs(deltas) > 0.02)

    total_range = np.ptp(eda_1hz)
    smoothness_ratio = (np.std(deltas) / total_range) if total_range > 0 else 0.0

    # Debounced Spike Detection
    spike_locs = []
    if len(deltas) > 1:
        mz = modified_zscore(deltas)
        raw_spikes = np.where(np.abs(mz) > MODIFIED_ZSCORE_THRESHOLD)[0]
        last_spike = -999
        for loc in raw_spikes:
            if loc - last_spike > 2:  # 2-second debounce for spikes
                spike_locs.append(int(loc))
                last_spike = loc

    # Debounced Step-Change Detection
    step_locs = []
    if n > 2 * STEP_CHANGE_WINDOW:
        rolling_mean = pd.Series(eda_1hz).rolling(window=STEP_CHANGE_WINDOW, center=True, min_periods=1).mean().values
        step_diffs = np.diff(rolling_mean)
        mz_step = modified_zscore(step_diffs)
        raw_steps = np.where(np.abs(mz_step) > MODIFIED_ZSCORE_THRESHOLD)[0]
        
        last_step = -999
        for loc in raw_steps:
            if loc - last_step > STEP_CHANGE_WINDOW: # Full window debounce
                step_locs.append(int(loc))
                last_step = loc

    frac_extreme = np.mean((eda_1hz < ABS_FLOOR_US) | (eda_1hz > ABS_CEILING_US))

    return {
        "length": n,
        "frac_moving": frac_moving,
        "smoothness_ratio": smoothness_ratio,
        "n_spikes": len(spike_locs),
        "spike_locs": spike_locs,
        "n_steps": len(step_locs),
        "step_locs": step_locs,
        "frac_extreme": frac_extreme,
    }

# =============================================================================
# COHORT-RELATIVE FLAGGING
# =============================================================================
def cohort_modz_lower_flags(values, threshold=COHORT_MODZ_THRESHOLD, use_log=False):
    v = np.asarray(values)
    if use_log:
        # Log-transform solves the zero-boundary trap for strictly positive metrics
        v = np.log(v + 1e-8)
    mz = modified_zscore(v)
    return mz < -threshold

def cohort_modz_upper_flags(values, threshold=COHORT_MODZ_THRESHOLD):
    mz = modified_zscore(np.asarray(values))
    return mz > threshold

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    movie_folder = os.path.join(DATA_DIR, MOVIE_NAME)
    participant_files = sorted(glob.glob(os.path.join(movie_folder, '*physio*.tsv')))

    if not participant_files:
        print(f"ERROR: No physio files found in '{movie_folder}'")
        return

    print(f"Running QC on {len(participant_files)} subjects for '{MOVIE_NAME}'\n")

    subject_signals = {}
    subject_metrics = {}
    subject_status = {}

    for p_file in participant_files:
        filename = os.path.basename(p_file)
        match = re.match(r'(sub-[A-Za-z0-9]+)', filename)
        subject_id = match.group(1) if match else filename.split('_')[0]

        try:
            if KNOWN_TRIGGER_COLUMN is not None:
                needed_cols = sorted(set([EDA_COLUMN_INDEX, KNOWN_TRIGGER_COLUMN]))
                df = pd.read_csv(p_file, sep='\t', header=None, usecols=needed_cols)
                df.columns = needed_cols
            else:
                df = pd.read_csv(p_file, sep='\t', header=None)
        except Exception as e:
            subject_status[subject_id] = ("READ_ERROR", [f"File read error: {e}"])
            continue

        trigger_col = KNOWN_TRIGGER_COLUMN if KNOWN_TRIGGER_COLUMN is not None else detect_trigger_column(df)
        if trigger_col is None:
            subject_status[subject_id] = ("NO_TRIGGER", ["No valid trigger window found"])
            continue

        start_idx, end_idx = find_movie_window(df, trigger_col)
        if start_idx is None:
            subject_status[subject_id] = ("NO_WINDOW", ["Trigger column found but no valid window"])
            continue

        eda_raw = df[EDA_COLUMN_INDEX].values
        sliced = eda_raw[start_idx:end_idx]
        
        # --- NEW: Anti-aliasing filter before downsampling ---
        eda_clean = apply_anti_aliasing_filter(sliced, fs=1000, cutoff=2.0)
        
        # Downsample to 1Hz
        eda_1hz = np.array([np.mean(eda_clean[i:i + 1000]) for i in range(0, len(eda_clean), 1000)])

        subject_signals[subject_id] = eda_1hz
        subject_metrics[subject_id] = extract_subject_metrics(eda_1hz)

    if not subject_metrics:
        print("ERROR: No subjects successfully processed.")
        return

    lengths = np.array([m["length"] for m in subject_metrics.values()])
    frac_moving_vals = np.array([m["frac_moving"] for m in subject_metrics.values()])
    smoothness_vals = np.array([m["smoothness_ratio"] for m in subject_metrics.values()])
    n_spikes_vals = np.array([m["n_spikes"] for m in subject_metrics.values()])
    n_steps_vals = np.array([m["n_steps"] for m in subject_metrics.values()])

    median_length = np.median(lengths)

    # --- Use Log-Transform for zero-bounded metric flagging ---
    flags_low_smoothness = cohort_modz_lower_flags(smoothness_vals, use_log=True)
    flags_low_frac_moving = cohort_modz_lower_flags(frac_moving_vals, use_log=True)
    
    flags_high_spikes = cohort_modz_upper_flags(n_spikes_vals)
    flags_high_steps = cohort_modz_upper_flags(n_steps_vals)

    subject_ids_ordered = list(subject_metrics.keys())
    smoothness_flag_lookup = dict(zip(subject_ids_ordered, flags_low_smoothness))
    frac_moving_flag_lookup = dict(zip(subject_ids_ordered, flags_low_frac_moving))
    spikes_flag_lookup = dict(zip(subject_ids_ordered, flags_high_spikes))
    steps_flag_lookup = dict(zip(subject_ids_ordered, flags_high_steps))

    print(f"Cohort median usable length: {median_length:.0f}s")
    print(f"Cohort smoothness ratio: median={np.median(smoothness_vals)*100:.2f}%, "
          f"{np.sum(flags_low_smoothness)} flagged as cohort-relative low outliers")
    print(f"Cohort flatline (frac moving): median={np.median(frac_moving_vals)*100:.2f}%, "
          f"{np.sum(flags_low_frac_moving)} flagged as cohort-relative low outliers")
    print(f"Cohort spike count: median={np.median(n_spikes_vals):.0f}, "
          f"{np.sum(flags_high_spikes)} flagged as cohort-relative high outliers")
    print(f"Cohort step count: median={np.median(n_steps_vals):.0f}, "
          f"{np.sum(flags_high_steps)} flagged as cohort-relative high outliers\n")

    results = []
    for p_file in participant_files:
        filename = os.path.basename(p_file)
        match = re.match(r'(sub-[A-Za-z0-9]+)', filename)
        subject_id = match.group(1) if match else filename.split('_')[0]

        if subject_id in subject_status:
            status, reasons = subject_status[subject_id]
            print(f"  {subject_id}: {status}")
            results.append({
                "Subject": subject_id, "Status": status,
                "Truncated": "", "Flatline": "", "TooSmooth": "",
                "SpikeArtifact": "", "StepChange": "", "ExtremeRange": "",
                "AnyFlag": True, "Reasons": "; ".join(reasons)
            })
            continue

        m = subject_metrics[subject_id]
        reasons = []

        is_truncated = m["length"] < (TRUNCATION_FRACTION_LIMIT * median_length)
        if is_truncated:
            pct = m["length"] / median_length * 100
            reasons.append(f"Truncated: {m['length']}s ({pct:.0f}% of median)")

        is_flatline = frac_moving_flag_lookup[subject_id]
        if is_flatline:
            reasons.append(f"Flatline (Log-modZ Outlier): {m['frac_moving']*100:.1f}% moving")

        is_too_smooth = smoothness_flag_lookup[subject_id]
        if is_too_smooth:
            reasons.append(f"Abnormally smooth (Log-modZ Outlier): {m['smoothness_ratio']*100:.2f}%")

        has_spike_artifact = spikes_flag_lookup[subject_id]
        if has_spike_artifact:
            reasons.append(f"Spike count outlier: {m['n_spikes']} spikes at secs {m['spike_locs'][:5]}")

        has_step_change = steps_flag_lookup[subject_id]
        if has_step_change:
            reasons.append(f"Step-change count outlier: {m['n_steps']} steps at secs {m['step_locs'][:5]}")

        is_extreme = m["frac_extreme"] > ABS_EXTREME_FRACTION_LIMIT
        if is_extreme:
            reasons.append(f"Extreme values: {m['frac_extreme']*100:.1f}% outside limits")

        any_flag = any([is_truncated, is_flatline, is_too_smooth, has_spike_artifact,
                        has_step_change, is_extreme])
        status_str = "FLAGGED" if any_flag else "OK"
        print(f"  {subject_id}: {status_str}" + (f" - {'; '.join(reasons)}" if reasons else ""))

        results.append({
            "Subject": subject_id, "Status": status_str,
            "Truncated": is_truncated, "Flatline": is_flatline,
            "TooSmooth": is_too_smooth, "SpikeArtifact": has_spike_artifact,
            "StepChange": has_step_change, "ExtremeRange": is_extreme,
            "AnyFlag": any_flag, "Reasons": "; ".join(reasons) if reasons else ""
        })

    summary_df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, f"{MOVIE_NAME}_qc_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    n_flagged = summary_df['AnyFlag'].sum()
    print(f"\n{'='*60}\nQC SUMMARY: {n_flagged}/{len(results)} subjects flagged")
    print(f"Saved to: {csv_path}\n{'='*60}")

if __name__ == "__main__":
    main()