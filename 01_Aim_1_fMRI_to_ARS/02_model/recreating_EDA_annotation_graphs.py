import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD BEHAVIORAL ANNOTATIONS (1Hz)
# ==========================================
print("--- Loading IntenseEmotion TSV Data ---")
emotion_path = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\data\annotation_files\Annot_Chatter_stim.tsv'

json_columns = [
    "Standards", "PleasantSelf", "SocialNorms", "PleasantOther", "GoalsOther",
    "Controlled", "Predictable", "Suddenly", "Agent", "Urgency", "Lips",
    "Tears", "Eyebrows", "Smile", "Frown", "Stop", "Undo", "Repeat",
    "Oppose", "Attention", "Tackle", "Command", "Support", "Move", "Care",
    "Bad", "Good", "Calm", "Strong", "IntenseEmotion", "Alert", "AtEase",
    "Muscle", "Heartrate", "Throat", "Stomach", "Warm", "Anger", "Guilt",
    "WarmHeartedness", "Disgust", "Happiness", "Fear", "Regard", "Anxiety",
    "Satisfaction", "Pride", "Surprise", "Love", "Sad"
]

emotion_df = pd.read_csv(emotion_path, sep='\t', header=None, names=json_columns)
intense_emotion_1hz = emotion_df['IntenseEmotion'].values
movie_duration_sec = len(intense_emotion_1hz)

# ==========================================
# 2. DEFINE THE 1.3s TR TIMELINE
# ==========================================
tr_duration = 1.3
samples_per_tr = int(tr_duration * 1000) 
num_trs = int(movie_duration_sec // tr_duration) 
required_eda_samples = num_trs * samples_per_tr

time_1hz = np.arange(movie_duration_sec)
tr_timestamps_sec = np.arange(num_trs) * tr_duration

annot_interpolator = interp1d(time_1hz, intense_emotion_1hz, kind='linear', bounds_error=False, fill_value='extrapolate')
emotion_tr = annot_interpolator(tr_timestamps_sec)

# ==========================================
# 3. VERIFY CONFIG, EXCLUDE BAD DATA, & DOWNSAMPLE
# ==========================================
physio_dir = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\data\output_sorted_by_movie\Chatter'
tsv_files = glob.glob(os.path.join(physio_dir, '*_physio.tsv'))

EDA_INDEX = 2      
TRIGGER_INDEX = 5  

# Explicit exclusion list based on previous QC
EXCLUDE_SUBJECTS = ['sub-S01', 'sub-S05', 'sub-S10', 'sub-S14', 'sub-S07', 'sub-S16', 'sub-S22']

valid_subject_arrays = []
print(f"\n--- Processing {len(tsv_files)} Physio TSV Files ---")

for file_path in tsv_files:
    file_name = os.path.basename(file_path)
    
    if any(bad_id in file_name for bad_id in EXCLUDE_SUBJECTS):
        print(f"QC Exclusion: Skipping {file_name}")
        continue

    try:
        df_physio = pd.read_csv(file_path, sep='\t', header=None)
    except Exception:
        continue
        
    eda_1000hz = df_physio.iloc[:, EDA_INDEX].values
    trigger_1000hz = df_physio.iloc[:, TRIGGER_INDEX].values
    
    start_sample = np.argmax(trigger_1000hz > 2.5)
    if start_sample == 0 and trigger_1000hz[0] <= 2.5: continue
    if len(eda_1000hz) < (start_sample + required_eda_samples): continue
        
    movie_eda_1000hz = eda_1000hz[start_sample : start_sample + required_eda_samples]
    eda_tr = movie_eda_1000hz.reshape(num_trs, samples_per_tr).mean(axis=1)
    
    valid_subject_arrays.append(eda_tr)

master_eda_tr = np.vstack(valid_subject_arrays)

# ==========================================
# 4. RAW SIGNAL PIPELINE (FULL DURATION)
# ==========================================
# Calculate the cohort mean across the entire movie timeline
mean_eda_tr = np.mean(master_eda_tr, axis=0)

# Z-Score directly on the raw, full-length data
norm_emotion = (emotion_tr - np.mean(emotion_tr)) / (np.std(emotion_tr) + 1e-8)
norm_mean_eda = (mean_eda_tr - np.mean(mean_eda_tr)) / (np.std(mean_eda_tr) + 1e-8)

r_group, p_group = pearsonr(norm_emotion, norm_mean_eda)
print(f"\nRaw Full-Length Signals Pearson r: {r_group:.4f} (p-value: {p_group:.4e})")

# ==========================================
# 5. VISUAL PLOTTING
# ==========================================
plt.figure(figsize=(12, 5))
tr_indices = np.arange(num_trs)

# A. Background Individual Lines (Z-Scored Only)
for subj in range(len(valid_subject_arrays)):
    subj_eda = master_eda_tr[subj, :]
    norm_subj = (subj_eda - np.mean(subj_eda)) / (np.std(subj_eda) + 1e-8)
    label_str = 'Individual Raw EDA' if subj == 0 else ""
    plt.plot(tr_indices, norm_subj, color='green', alpha=0.15, linewidth=1, label=label_str)

# B. Foreground Lines
plt.plot(tr_indices, norm_emotion, label='IntenseEmotion (Raw)', color='blue', linewidth=3, zorder=5)
plt.plot(tr_indices, norm_mean_eda, label='EDA / GSR (Raw Clean Cohort Mean)', color='green', linewidth=3, zorder=6)

# C. Aesthetics
plt.title(f'Raw Standardized Signals - Full Duration (r = {r_group:.4f})')
plt.xlabel('fMRI Time (TR Index, 1.3s resolution)')
plt.ylabel('Normalized Amplitude (Z-Score)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()