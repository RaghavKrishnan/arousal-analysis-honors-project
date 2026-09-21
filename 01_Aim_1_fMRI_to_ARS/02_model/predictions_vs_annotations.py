import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def main():
    # ==========================================
    # 1. CONFIGURATION & FILE PATHS
    # ==========================================
    MOVIE_NAME = 'Chatter'
    
    # The five specific parameters you requested
    target_params = ['IntenseEmotion', 'Predictable', 'Alert', 'Attention', 'Heartrate']
    # Distinct colors for each parameter to keep the graph readable
    param_colors = ['blue', 'green', 'purple', 'darkorange', 'cyan']
    
    base_dir = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS'
    
    # UPDATED: Added the MOVIE_NAME subfolder into the path
    pred_path = os.path.join(base_dir, '05_extracted_features', MOVIE_NAME, f'Predicted_Arousal_{MOVIE_NAME}.npy')
    
    data_dir = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\data\annotation_files'
    emotion_path = os.path.join(data_dir, f'Annot_{MOVIE_NAME}_stim.tsv')
    
    output_dir = os.path.join(base_dir, '05_extracted_features', MOVIE_NAME, 'Behavioral_Comparisons')
    os.makedirs(output_dir, exist_ok=True)
    output_plot_path = os.path.join(output_dir, f'{MOVIE_NAME}_Combined_Spaghetti_Plot.png')

    # ==========================================
    # 2. LOAD PREDICTIONS
    # ==========================================
    print(f"--- Loading SVR Predictions for {MOVIE_NAME} ---")
    try:
        predicted_arousal = np.load(pred_path)
    except FileNotFoundError:
        print(f"[!] Error: Could not find predictions at {pred_path}")
        return
        
    num_subjects = predicted_arousal.shape[0]
    mean_predicted_arousal = np.mean(predicted_arousal, axis=0)

    # ==========================================
    # 3. LOAD BEHAVIORAL ANNOTATIONS
    # ==========================================
    print(f"--- Loading Behavioral TSV Data ---")
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

    try:
        emotion_df = pd.read_csv(emotion_path, sep='\t', header=None, names=json_columns)
    except FileNotFoundError:
        print(f"[!] Error: Could not find TSV at {emotion_path}")
        return

    # ==========================================
    # 4. TEMPORAL ALIGNMENT TIMELINES
    # ==========================================
    # Chatter constants
    tr_duration = 1.3
    total_scan_trs = 464 
    window_size = 45

    all_tr_times = np.arange(total_scan_trs) * tr_duration
    pred_window_centers_tr = np.arange(predicted_arousal.shape[1]) + (window_size / 2.0)
    pred_times_sec = pred_window_centers_tr * tr_duration
    
    annot_start_sec = 90.0
    # We will dynamically calculate the end based on the intense emotion array length
    movie_duration_sec = len(emotion_df['IntenseEmotion'].values)
    annot_times_sec = np.arange(movie_duration_sec) + annot_start_sec

    movie_tr_mask = (all_tr_times >= annot_start_sec) & (all_tr_times <= (annot_start_sec + movie_duration_sec))
    movie_tr_times = all_tr_times[movie_tr_mask]
    movie_tr_indices = movie_tr_times / tr_duration

    # ==========================================
    # 5. CONSTRUCT THE VISUALIZATION
    # ==========================================
    print("--- Generating Plot ---")
    plt.figure(figsize=(15, 8)) 

    # Helper function to Z-score the fMRI arrays for visual overlay
    def normalize_array(arr):
        return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

    # --- A. PLOT INDIVIDUAL PREDICTIONS (SPAGHETTI) ---
    for subj in range(num_subjects):
        subj_pred_interpolator = interp1d(pred_times_sec, predicted_arousal[subj, :], kind='linear', bounds_error=False, fill_value='extrapolate')
        subj_pred_at_trs = subj_pred_interpolator(movie_tr_times)
        norm_subj_pred = normalize_array(subj_pred_at_trs)
        
        # Only add the label for the very first subject so the legend doesn't get 30 duplicate entries
        label_str = 'Individual SVR Predictions' if subj == 0 else ""
        plt.plot(movie_tr_indices, norm_subj_pred, color='red', alpha=0.15, linewidth=1, label=label_str)

    # --- B. PLOT THE COHORT MEAN PREDICTION ---
    pred_interpolator = interp1d(pred_times_sec, mean_predicted_arousal, kind='linear', bounds_error=False, fill_value='extrapolate')
    mean_pred_at_trs = pred_interpolator(movie_tr_times)
    norm_mean_pred = normalize_array(mean_pred_at_trs)
    
    plt.plot(movie_tr_indices, norm_mean_pred, color='red', linewidth=4, zorder=10, label='Mean SVR Predicted Arousal')

    # --- C. PLOT THE BEHAVIORAL ANNOTATIONS (NO EXTRA NORMALIZATION) ---
    for param, color in zip(target_params, param_colors):
        if param in emotion_df.columns:
            param_1hz = emotion_df[param].values
            
            # Interpolate behavioral data to the TR timeline
            annot_interpolator = interp1d(annot_times_sec, param_1hz, kind='linear', bounds_error=False, fill_value='extrapolate')
            param_at_trs = annot_interpolator(movie_tr_times)
            
            # Plot the annotation directly (already Z-scored)
            plt.plot(movie_tr_indices, param_at_trs, color=color, linewidth=2.5, zorder=5, label=f'{param} (Annotation)')
        else:
            print(f"[!] Warning: {param} not found in the TSV data. Skipping.")

    # --- D. FORMATTING & DISPLAY ---
    plt.title(f'Figure 1: Arousal Predictions and Emotional annotations for the Chatter movie', fontsize=16, fontweight='bold', y=-0.30)
    plt.xlabel('Time(s)', fontsize=12)
    plt.ylabel('Amplitude (Z-Score)', fontsize=12)
    
    # Place the legend at the bottom center, split into 3 columns
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Upped the DPI from 300 to 600 for a much higher resolution output
    plt.savefig(output_plot_path, dpi=600, bbox_inches='tight')
    plt.show()
    
    print(f"-> Successfully saved combined plot to: {output_plot_path}")

if __name__ == "__main__":
    main()