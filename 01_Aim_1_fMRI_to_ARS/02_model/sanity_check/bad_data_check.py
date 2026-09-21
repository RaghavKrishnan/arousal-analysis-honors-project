import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
physio_dir = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\data\output_sorted_by_movie\Chatter'
# Sorting ensures the grid is in alphabetical order (sub-S01, sub-S02, etc.)
tsv_files = sorted(glob.glob(os.path.join(physio_dir, '*_physio.tsv')))

EDA_INDEX = 2      
TRIGGER_INDEX = 5  

# We know the Chatter movie is ~405 seconds
expected_duration_sec = 405 
required_samples = expected_duration_sec * 1000

# ==========================================
# 2. SETUP THE MATPLOTLIB GRID
# ==========================================
num_files = len(tsv_files)
cols = 5
rows = math.ceil(num_files / cols)

# Create a massive canvas (e.g., 20 inches wide)
fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows), sharex=True)
axes = axes.flatten() # Flatten the 2D array of axes for easy iterating

print(f"Found {num_files} files. Generating QC Grid...")

# ==========================================
# 3. ITERATE, EXTRACT, AND PLOT
# ==========================================
for idx, file_path in enumerate(tsv_files):
    ax = axes[idx]
    file_name = os.path.basename(file_path)
    # Extract just the subject ID (e.g., "sub-S01") for the title
    subj_id = file_name.split('_')[0] 
    
    ax.set_title(subj_id, fontweight='bold')
    
    try:
        df_physio = pd.read_csv(file_path, sep='\t', header=None)
    except Exception as e:
        ax.text(0.5, 0.5, "File Read Error", color='red', ha='center', va='center')
        continue
        
    eda_1000hz = df_physio.iloc[:, EDA_INDEX].values
    trigger_1000hz = df_physio.iloc[:, TRIGGER_INDEX].values
    
    # Trigger Logic
    start_sample = np.argmax(trigger_1000hz > 2.5)
    if start_sample == 0 and trigger_1000hz[0] <= 2.5:
        ax.text(0.5, 0.5, "NO TRIGGER", color='red', ha='center', va='center', fontweight='bold')
        continue
        
    # Length Logic
    if len(eda_1000hz) < (start_sample + required_samples):
        ax.text(0.5, 0.5, "RECORDING TOO SHORT", color='red', ha='center', va='center', fontweight='bold')
        continue
        
    # Extract the true movie block
    movie_eda_1000hz = eda_1000hz[start_sample : start_sample + required_samples]
    
    # Compress 1000ms chunks into 1Hz for rapid rendering
    movie_eda_1hz = movie_eda_1000hz.reshape(expected_duration_sec, 1000).mean(axis=1)
    time_axis = np.arange(expected_duration_sec)
    
    # Plot the raw data
    ax.plot(time_axis, movie_eda_1hz, color='teal', linewidth=1.5)
    ax.grid(True, linestyle='--', alpha=0.3)

# ==========================================
# 4. CLEANUP AND DISPLAY
# ==========================================
# Hide any completely empty subplots if the file count isn't a perfect multiple of columns
for idx in range(num_files, len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle("Raw EDA Data Quality Control (QC) Dashboard - Chatter", fontsize=18, y=1.02)
plt.tight_layout()
plt.show()