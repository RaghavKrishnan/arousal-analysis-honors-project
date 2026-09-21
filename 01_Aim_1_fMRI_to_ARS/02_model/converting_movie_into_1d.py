

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def main():

  #setup
    MOVIE_NAME = 'Superhero'
    target_params = ['IntenseEmotion', 'Predictable', 'Alert', 'Attention', 'Heartrate']
    
    base_dir = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS'
    pred_path = os.path.join(base_dir, '05_extracted_features', MOVIE_NAME, f'Predicted_Arousal_{MOVIE_NAME}.npy')
    
    data_dir = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\data\annotation_files'
    emotion_path = os.path.join(data_dir, f'Annot_{MOVIE_NAME}_stim.tsv')
    
    output_dir = os.path.join(base_dir, '05_extracted_features', MOVIE_NAME, 'Behavioral_Comparisons')
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f'{MOVIE_NAME}_LMM_Long_Format.csv')

    #loading data
    print(f"--- Loading Data for {MOVIE_NAME} ---")
    predicted_arousal = np.load(pred_path)
    num_subjects = predicted_arousal.shape[0]

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

    #alignig teh timeline
    tr_duration = 1.3
    total_scan_trs = 464  
    window_size = 45

    all_tr_times = np.arange(total_scan_trs) * tr_duration
    pred_window_centers_tr = np.arange(predicted_arousal.shape[1]) + (window_size / 2.0)
    pred_times_sec = pred_window_centers_tr * tr_duration
    
    annot_start_sec = 90.0
    movie_duration_sec = len(emotion_df['IntenseEmotion'].values)
    annot_times_sec = np.arange(movie_duration_sec) + annot_start_sec

    movie_tr_mask = (all_tr_times >= annot_start_sec) & (all_tr_times <= (annot_start_sec + movie_duration_sec))
    movie_tr_times = all_tr_times[movie_tr_mask]

    # Pre-calculate aligned behavioral annotations to avoid redundant loop operations
    aligned_behaviors = {}
    for param in target_params:
        if param in emotion_df.columns:
            annot_interpolator = interp1d(annot_times_sec, emotion_df[param].values, kind='linear', bounds_error=False, fill_value='extrapolate')
            aligned_behaviors[param] = annot_interpolator(movie_tr_times)
        else:
            aligned_behaviors[param] = np.full(len(movie_tr_times), np.nan)


    #Creating the tidy version of the data
    print("--- Restructuring Matrix to Long Format ---")
    long_data = []
    
    for subj in range(num_subjects):
        subj_id = f"Subject_{subj+1:02d}"
        
        # Interpolating the predictions to fit the curve 
        subj_pred_interpolator = interp1d(pred_times_sec, predicted_arousal[subj, :], kind='linear', bounds_error=False, fill_value='extrapolate')
        subj_pred_at_trs = subj_pred_interpolator(movie_tr_times)
        
        # Map every valid TR window to a unique row
        for tr_idx, tr_time in enumerate(movie_tr_times):
            row = {
                'subjectid': subj_id,
                'TR_Index': tr_idx,
                'Time_sec': tr_time,
                'svr': subj_pred_at_trs[tr_idx]
            }
            # Attach the pre-calculated behavioral annotations
            for param in target_params:
                row[param] = aligned_behaviors[param][tr_idx]
                
            long_data.append(row)

    df_long = pd.DataFrame(long_data)
    df_long.to_csv(output_csv_path, index=False)
    print(f"-> Exported {len(df_long)} total rows to: {output_csv_path}")

if __name__ == "__main__":
    main()