"""
eda_quality_control.py  (v3 - cohort-relative recalibration)
==============================================================
WHY THIS VERSION EXISTS:
The previous version used fixed, literature-derived thresholds (e.g. "flag
if smoothness ratio < 3%"). Run on this actual Chatter cohort, that
threshold flagged 26/30 subjects - an implausible result. Inspecting the
printed values showed why: this cohort's smoothness ratios cluster tightly
between ~1% and ~3%, so a fixed 3% cutoff sliced straight through the
middle of one continuous distribution instead of separating a small group
of genuine outliers from the rest. Only sub-S05 (0.19%) showed the kind of
order-of-magnitude separation that should actually count as an outlier.

THE FIX: for checks where "normal" depends on how THIS specific dataset
was collected and preprocessed (smoothness, flatline fraction, artifact
counts) - rather than on a fixed physical constant - flag subjects using
Tukey's IQR fence relative to this cohort's OWN distribution, the same way
the truncation check already (correctly) compared each subject to the
cohort median instead of a hardcoded expected duration.

Tukey's fence: a value is flagged as a "mild" outlier if it falls more
than IQR_MULTIPLIER x IQR below Q1 (for metrics where LOW = bad) or above
Q3 (for metrics where HIGH = bad). IQR_MULTIPLIER = 1.5 is the standard,
widely-used convention (a "mild outlier" fence); 3.0 is the stricter
"extreme outlier" convention. This is a well-established, distribution-
agnostic method - it does not assume normality, which matters here since
physiological delta distributions are typically right-skewed, not
Gaussian.

WHAT REMAINS AN ABSOLUTE (non-cohort-relative) CHECK, AND WHY:
- Extreme floor/ceiling (0.5-50 uS): this tests physical/electrical
  plausibility, which does not depend on what the rest of THIS cohort
  looks like. sub-S01's 55% out-of-range result under the old fixed
  threshold matched what we saw visually (a hard flatline-at-zero
  discontinuity), so this check was already working correctly and is left
  unchanged.
- No-trigger / no-window: these are hard technical failures, not
  distributional judgments.

WITHIN-SUBJECT SPIKE/STEP DETECTION ALSO CHANGED:
Individual timepoint outlier detection now uses a MAD-based (median
absolute deviation) modified z-score instead of a mean/SD-based z-score.
This is a standard, more robust alternative for skewed or heavy-tailed
data (the mean and SD are themselves distorted by the very outliers you
are trying to detect; the median and MAD are not). A subject is then
flagged only if their TOTAL COUNT of such timepoints is itself a
cohort-level outlier - not if they have even one, which is what let
ordinary tail variation trigger the previous version.

HOW TO RUN:
    python eda_quality_control.py
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
MOVIE_NAME = "Chatter"

DATA_DIR = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\data\output_sorted_by_movie'
OUTPUT_DIR = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\results\eda_quality_control'

EDA_COLUMN_INDEX = 2
KNOWN_TRIGGER_COLUMN = 7   # Chatter = 7, per your validated MOVIES_CONFIG

# --- Cohort-relative outlier detection (MAD-based modified z-score, used
#     for ALL cohort-relative checks now - see diagnosis in file docstring
#     history: Tukey's IQR fence degenerates to <=0 on metrics that are
#     bounded at zero and right-skewed, such as smoothness_ratio and
#     frac_moving, which silently disabled those checks entirely) ---
COHORT_MODZ_THRESHOLD = 3.5   # Iglewicz & Hoaglin's commonly-cited default
                               # for the modified z-score method generally

# --- Within-subject timepoint detection (MAD-based, robust to skew) ---
MODIFIED_ZSCORE_THRESHOLD = 5.0   # NOTE: diagnostic printout below will show
                                    # whether this is well-calibrated for
                                    # YOUR actual delta distribution - do not
                                    # trust this number until you've checked
STEP_CHANGE_WINDOW = 10           # seconds; rolling window for step detection

# --- Absolute, non-cohort-relative checks (kept from v2, unchanged) ---
ABS_FLOOR_US = 0.5
ABS_CEILING_US = 50.0
ABS_EXTREME_FRACTION_LIMIT = 0.10

# --- Truncation (already cohort-relative in v2; kept identical) ---
TRUNCATION_FRACTION_LIMIT = 0.70


# =============================================================================
# STEP 1: TRIGGER DETECTION (unchanged from the fixed v2)
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
    """Label-based indexing (df[col]) - see v2 fix notes: positional .iloc
    breaks when KNOWN_TRIGGER_COLUMN's usecols path relabels a reduced
    dataframe with original column numbers."""
    col_values = df[trigger_col_idx].values
    start_idx, end_idx, _ = _longest_valid_window(col_values)
    return start_idx, end_idx


# =============================================================================
# STEP 2: ROBUST (MAD-BASED) WITHIN-SUBJECT OUTLIER DETECTION
# =============================================================================
def modified_zscore(x):
    """
    Iglewicz & Hoaglin's modified z-score: 0.6745 * (x - median) / MAD.
    More robust than a mean/SD z-score when the underlying distribution is
    skewed or heavy-tailed (real physiological deltas usually are), because
    median and MAD are not themselves distorted by the outliers you are
    trying to detect, the way mean and SD are.
    """
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - median) / mad


# =============================================================================
# STEP 3: PER-SUBJECT RAW METRIC EXTRACTION (no flagging yet - flagging
#          happens in a second pass once the cohort distribution is known)
# =============================================================================
def extract_subject_metrics(eda_1hz):
    """Returns a dict of raw, unflagged metrics for one subject."""
    n = len(eda_1hz)
    deltas = np.diff(eda_1hz)

    # Flatline metric: fraction of seconds with a "real" change
    frac_moving = np.mean(np.abs(deltas) > 0.02)

    # Smoothness metric: high-frequency variability relative to total range
    total_range = np.ptp(eda_1hz)
    smoothness_ratio = (np.std(deltas) / total_range) if total_range > 0 else 0.0

    # Spike count: individual seconds whose delta is a robust outlier
    spike_locs = []
    if len(deltas) > 1:
        mz = modified_zscore(deltas)
        spike_locs = np.where(np.abs(mz) > MODIFIED_ZSCORE_THRESHOLD)[0].tolist()

    # Step-change count: sustained rolling-mean level shifts, robust z-score
    step_locs = []
    if n > 2 * STEP_CHANGE_WINDOW:
        rolling_mean = pd.Series(eda_1hz).rolling(
            window=STEP_CHANGE_WINDOW, center=True, min_periods=1
        ).mean().values
        step_diffs = np.diff(rolling_mean)
        mz_step = modified_zscore(step_diffs)
        step_locs = np.where(np.abs(mz_step) > MODIFIED_ZSCORE_THRESHOLD)[0].tolist()

    # Absolute extreme-range fraction (kept as-is; not cohort-relative)
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
# STEP 4: COHORT-RELATIVE FLAGGING (MAD-based modified z-score, not Tukey IQR)
# =============================================================================
def cohort_modz_lower_flags(values, threshold=COHORT_MODZ_THRESHOLD):
    """
    Flags values that are LOW outliers relative to the cohort, using the
    same modified-z-score logic as within-subject detection, applied here
    across subjects instead of across timepoints. Chosen over Tukey's IQR
    fence specifically because it does not degenerate to an impossible
    (<=0) threshold on metrics that are bounded at zero and right-skewed -
    see file docstring for the diagnosed failure of the IQR approach on
    this exact data.
    """
    mz = modified_zscore(np.asarray(values))
    return mz < -threshold


def cohort_modz_upper_flags(values, threshold=COHORT_MODZ_THRESHOLD):
    """Same as above, flagging HIGH outliers (used for spike/step counts)."""
    mz = modified_zscore(np.asarray(values))
    return mz > threshold


# =============================================================================
# STEP 5: MAIN
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    movie_folder = os.path.join(DATA_DIR, MOVIE_NAME)
    participant_files = sorted(glob.glob(os.path.join(movie_folder, '*physio*.tsv')))

    if not participant_files:
        print(f"ERROR: No physio files found in '{movie_folder}'")
        return

    print(f"Running QC on {len(participant_files)} subjects for '{MOVIE_NAME}'\n")

    # --- Pass 1: extract signals + raw metrics for every subject ---
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
        eda_1hz = np.array([np.mean(sliced[i:i + 1000]) for i in range(0, len(sliced), 1000)])

        subject_signals[subject_id] = eda_1hz
        subject_metrics[subject_id] = extract_subject_metrics(eda_1hz)

    if not subject_metrics:
        print("ERROR: No subjects successfully processed.")
        return

    # --- Pass 2: compute cohort-relative fences from the ACTUAL distribution ---
    lengths = np.array([m["length"] for m in subject_metrics.values()])
    frac_moving_vals = np.array([m["frac_moving"] for m in subject_metrics.values()])
    smoothness_vals = np.array([m["smoothness_ratio"] for m in subject_metrics.values()])
    n_spikes_vals = np.array([m["n_spikes"] for m in subject_metrics.values()])
    n_steps_vals = np.array([m["n_steps"] for m in subject_metrics.values()])

    median_length = np.median(lengths)

    # --- DIAGNOSTIC: show the actual distribution of within-subject modified
    #     z-scores BEFORE deciding whether MODIFIED_ZSCORE_THRESHOLD=5.0 is
    #     appropriate. A well-calibrated threshold should flag a small
    #     fraction of timepoints for a typical subject, not ~8% as the
    #     previous run's median spike count of 32/~400 implied. ---
    all_delta_modz = []
    for sid, sig in subject_signals.items():
        d = np.diff(sig)
        if len(d) > 1:
            all_delta_modz.extend(np.abs(modified_zscore(d)).tolist())
    all_delta_modz = np.array(all_delta_modz)
    print("Diagnostic: distribution of |modified z-score| across ALL "
          "subject-seconds (use this to sanity-check MODIFIED_ZSCORE_THRESHOLD):")
    for pct in [50, 90, 95, 99, 99.5]:
        print(f"    {pct}th percentile: {np.percentile(all_delta_modz, pct):.2f}")
    print(f"    (current threshold = {MODIFIED_ZSCORE_THRESHOLD}; if this sits "
          f"below the ~95th percentile, most 'spikes' are just ordinary tail "
          f"variation, not genuine artifacts)\n")

    flags_low_smoothness = cohort_modz_lower_flags(smoothness_vals)
    flags_low_frac_moving = cohort_modz_lower_flags(frac_moving_vals)
    flags_high_spikes = cohort_modz_upper_flags(n_spikes_vals)
    flags_high_steps = cohort_modz_upper_flags(n_steps_vals)

    subject_ids_ordered = list(subject_metrics.keys())

    # --- DIAGNOSTIC: show every subject's actual smoothness value and its
    #     cohort modified z-score, sorted, so we can see EXACTLY where a
    #     specific subject sits relative to the cohort - rather than
    #     inferring "outlier-ness" from eyeballing a printed list, which is
    #     what led to an unverified assumption earlier in this process. ---
    smoothness_modz = modified_zscore(smoothness_vals)
    sort_order = np.argsort(smoothness_vals)
    print("Diagnostic: every subject's smoothness ratio and cohort modified z-score, sorted low to high:")
    for idx in sort_order:
        sid = subject_ids_ordered[idx]
        print(f"    {sid}: {smoothness_vals[idx]*100:.2f}%  (modz = {smoothness_modz[idx]:.2f})")
    print(f"    (current COHORT_MODZ_THRESHOLD = {COHORT_MODZ_THRESHOLD}; a subject is "
          f"flagged only if modz < -{COHORT_MODZ_THRESHOLD})\n")
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

    # --- Assemble final results ---
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
            reasons.append(f"Truncated: {m['length']}s ({pct:.0f}% of median {median_length:.0f}s)")

        is_flatline = frac_moving_flag_lookup[subject_id]
        if is_flatline:
            reasons.append(
                f"Flatline (cohort-relative outlier): {m['frac_moving']*100:.1f}% moving "
                f"(cohort median: {np.median(frac_moving_vals)*100:.1f}%)"
            )

        is_too_smooth = smoothness_flag_lookup[subject_id]
        if is_too_smooth:
            reasons.append(
                f"Abnormally smooth (cohort-relative outlier): {m['smoothness_ratio']*100:.2f}% "
                f"(cohort median: {np.median(smoothness_vals)*100:.2f}%)"
            )

        has_spike_artifact = spikes_flag_lookup[subject_id]
        if has_spike_artifact:
            reasons.append(
                f"Spike count is a cohort outlier: {m['n_spikes']} spikes "
                f"(cohort median: {np.median(n_spikes_vals):.0f}), at second(s) {m['spike_locs'][:5]}"
            )

        has_step_change = steps_flag_lookup[subject_id]
        if has_step_change:
            reasons.append(
                f"Step-change count is a cohort outlier: {m['n_steps']} steps "
                f"(cohort median: {np.median(n_steps_vals):.0f}), at second(s) {m['step_locs'][:5]}"
            )

        is_extreme = m["frac_extreme"] > ABS_EXTREME_FRACTION_LIMIT
        if is_extreme:
            reasons.append(
                f"Extreme values (absolute check): {m['frac_extreme']*100:.1f}% outside "
                f"{ABS_FLOOR_US}-{ABS_CEILING_US} uS"
            )

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

    # --- Save summary CSV ---
    summary_df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, f"{MOVIE_NAME}_qc_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    n_flagged = summary_df['AnyFlag'].sum()
    print(f"\n{'='*60}\nQC SUMMARY: {n_flagged}/{len(results)} subjects flagged")
    print(f"Saved to: {csv_path}\n{'='*60}")

    # --- Re-plot grid ---
    if subject_signals:
        subjects_sorted = sorted(subject_signals.keys())
        n_cols = 5
        n_rows = int(np.ceil(len(subjects_sorted) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows))
        axes = axes.flatten()
        flag_lookup = {r['Subject']: r['AnyFlag'] for r in results}

        for i, subj in enumerate(subjects_sorted):
            ax = axes[i]
            is_flagged = flag_lookup.get(subj, False)
            color = 'crimson' if is_flagged else 'teal'
            ax.plot(subject_signals[subj], color=color, linewidth=1)
            suffix = " [FLAGGED]" if is_flagged else ""
            ax.set_title(f"{subj}{suffix}", fontsize=9,
                        color='crimson' if is_flagged else 'black',
                        fontweight='bold' if is_flagged else 'normal')
            ax.tick_params(labelsize=7)

        for j in range(len(subjects_sorted), len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"{MOVIE_NAME}: EDA by Subject (cohort-relative QC, flagged in red)", fontsize=13)
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"{MOVIE_NAME}_qc_flagged_grid.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Flagged grid plot saved to: {plot_path}")


if __name__ == "__main__":
    main()