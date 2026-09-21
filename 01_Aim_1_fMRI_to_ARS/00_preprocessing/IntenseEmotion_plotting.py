import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import math

def main():
    # ==========================================
    # 1. DIRECTORY SETUP
    # ==========================================
    data_dir = r"C:\Users\ragha\Desktop\Honors Work Arousal Analysis\data\annotation_files"
    
    search_pattern = os.path.join(data_dir, "Annot_*_stim.tsv")
    tsv_files = glob.glob(search_pattern)
    tsv_files.sort()
    
    if not tsv_files:
        print("No .tsv annotation files found. Check your directory path.")
        return
        
    print(f"Found {len(tsv_files)} annotation files. Generating diagnostic plots...")

    # ==========================================
    # 2. YOUR BEHAVIORAL ANNOTATION LOGIC
    # ==========================================
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

    # ==========================================
    # 3. SUBPLOT GRID CONFIGURATION
    # ==========================================
    cols = 4
    rows = math.ceil(len(tsv_files) / cols)
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 12))
    axes = axes.flatten() 
    
    # ==========================================
    # 4. BATCH EXTRACTION AND PLOTTING
    # ==========================================
    for i, file_path in enumerate(tsv_files):
        filename = os.path.basename(file_path)
        movie_name = filename.replace('Annot_', '').replace('_stim.tsv', '')
        ax = axes[i]
        
        try:
            emotion_df = pd.read_csv(file_path, sep='\t', header=None, names=json_columns)
            intense_emotion_1hz = emotion_df['IntenseEmotion'].values
            
        except Exception as e:
            print(f"[!] Error processing {filename}: {e}")
            continue
            
        # Calculate true min and max to diagnose the scale
        true_min = intense_emotion_1hz.min()
        true_max = intense_emotion_1hz.max()
            
        # --- THE PLOTTING BLOCK ---
        ax.plot(intense_emotion_1hz, color='#1f77b4', linewidth=1.5)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Intense Emotion', fontsize=9)
        
        # Display the exact scale mathematically in the title
        ax.set_title(f"{movie_name}\n(Min: {true_min:.3f} | Max: {true_max:.3f})", fontweight='bold', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # WE REMOVED ALL Y-LIM CONSTRAINTS. It will auto-scale perfectly.
        
    # ==========================================
    # 5. CLEANUP AND DISPLAY
    # ==========================================
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()