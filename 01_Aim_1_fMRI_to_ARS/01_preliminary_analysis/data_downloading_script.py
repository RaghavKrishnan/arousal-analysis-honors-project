import os
import re
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def download_movie(movie_name, dataset_id, base_dir, s3_client):
    """
    Scans the S3 bucket and downloads the preprocessed fMRI files for a specific movie.
    """
    print(f"\n{'='*60}")
    print(f" INITIALIZING DOWNLOAD FOR: {movie_name}")
    print(f"{'='*60}")

    # Local target directory
    TARGET_FOLDER = os.path.join(base_dir, '04_data', 'fmri_subject_data', movie_name)
    os.makedirs(TARGET_FOLDER, exist_ok=True)

    # ==========================================
    # EXACT REGEX PATTERN MATCHING
    # ==========================================
    pattern_string = rf"^sub-S(0[1-9]|[1-2][0-9]|3[0-2])_ses-[1-5]_task-{movie_name}_space-MNI_desc-ppres_bold\.nii\.gz$"
    file_pattern = re.compile(pattern_string)

    bucket_name = 'openneuro.org'
    prefix = f'{dataset_id}/derivatives/preprocessing/'

    print(f"Connecting to S3 path: s3://{bucket_name}/{prefix}")
    print(f"Hunting across all sessions (1-5) for '{movie_name}'...")
    print("Scanning AWS directory tree (This usually takes 1 to 3 minutes)...")

    # ==========================================
    # SCAN, MAP, AND DEDUPLICATE 
    # ==========================================
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    unique_subjects = {}
    page_count = 0

    for page in pages:
        page_count += 1
        if page_count % 5 == 0:
            print(f"  ...scanned {page_count * 1000} files...")

        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)
                
                if file_pattern.match(filename):
                    subj_match = re.search(r'(sub-S\d{2})', filename)
                    ses_match = re.search(r'(ses-[1-5])', filename)
                    
                    subj = subj_match.group(1) if subj_match else "Unknown Subj"
                    ses = ses_match.group(1) if ses_match else "Unknown Ses"
                    
                    if subj not in unique_subjects:
                        unique_subjects[subj] = {
                            'key': key,
                            'filename': filename,
                            'size_bytes': obj['Size'],
                            'subj': subj,
                            'ses': ses
                        }
                    else:
                        kept_ses = unique_subjects[subj]['ses']
                        print(f"  [!] Duplicate found for {subj}: {kept_ses} and {ses}. Keeping {kept_ses}.")

    matched_files = list(unique_subjects.values())
    matched_files = sorted(matched_files, key=lambda x: x['subj'])

    if not matched_files:
        print(f"\n[ERROR] No files matched your criteria for '{movie_name}'. Moving to next movie...")
        return

    print(f"\nScan Complete! Found {len(matched_files)} unique subject files.")
    print("Session Mapping Log:")
    for f in matched_files:
        print(f"  - {f['subj']} -> Found in {f['ses']}")
        
    print(f"\nBeginning download to:\n-> {TARGET_FOLDER}\n")

    # ==========================================
    # EXECUTE DOWNLOADS
    # ==========================================
    for idx, f_data in enumerate(matched_files):
        filename = f_data['filename']
        key = f_data['key']
        size_bytes = f_data['size_bytes']
        
        local_file_path = os.path.join(TARGET_FOLDER, filename)
        size_mb = size_bytes / (1024 * 1024)
        
        if os.path.exists(local_file_path):
            local_size = os.path.getsize(local_file_path)
            if local_size == size_bytes:
                print(f"[{idx+1:02d}/{len(matched_files)}] SKIPPING (Already complete): {filename}")
                continue
                
        print(f"[{idx+1:02d}/{len(matched_files)}] DOWNLOADING: {filename} ({size_mb:.1f} MB)...")
        s3_client.download_file(bucket_name, key, local_file_path)
        
    print(f"\nSUCCESS: All specific preprocessed files for {movie_name} downloaded.")


def main():
    # ==========================================
    # MASTER CONFIGURATION
    # ==========================================
    DATASET_ID = "ds004892"
    BASE_DIR = r'C:\Users\ragha\Desktop\Honors Work Arousal Analysis\01_Aim_1_fMRI_to_ARS'
    
   
    MOVIE_LIST = [
        "Superhero",
        "Chatter",
        "BetweenViewings",
        "BigBuckBunny",
        "FirstBite",
        "LessonLearned",
        "Payload",
        "Sintel",
        "TearsOfSteel",
        "TheSecretNumber",
        "ToClaireFromSonny",
        "YouAgain",
        "AfterTheRain",
        "Spaceman"

        
    ]

    # Initialize S3 client ONCE for the entire run
    print("Initializing AWS S3 Client...")
    s3_client = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED))

    # Loop through the array and download each movie sequentially
    for movie in MOVIE_LIST:
        download_movie(movie, DATASET_ID, BASE_DIR, s3_client)
        
    print("\n==========================================")
    print("ALL MOVIES SUCCESSFULLY DOWNLOADED.")
    print("==========================================")

if __name__ == "__main__":
    main()