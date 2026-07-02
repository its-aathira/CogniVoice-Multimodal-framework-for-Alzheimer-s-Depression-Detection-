import librosa
import numpy as np
import pandas as pd
import os

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        # Fundamental Frequency extraction
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) == 0: return None

        # Jitter & Shimmer Calculation
        jitter = np.mean(np.abs(np.diff(f0_clean))) / np.mean(f0_clean)
        rms = librosa.feature.rms(y=y)[0]
        rms_clean = rms[rms > 0]
        shimmer = np.mean(np.abs(np.diff(rms_clean))) / np.mean(rms_clean)
        
        return [jitter, shimmer]
    except:
        return None

# --- MAIN BATCH PROCESSING ---
base_path = "/Users/aathirashibu/Documents/CogniVoice /data_depression/Audio_Dataset" # Ensure this matches your folder name
data_rows = []

# Define the folder mapping
categories = {
    'normal': 'Healthy',
    'depression/stage1': 'Depression_Stage1',
    'depression/stage2': 'Depression_Stage2'
}

print("🚀 Starting Batch Voice Extraction...")

for folder_rel, label in categories.items():
    folder_path = os.path.join(base_path, folder_rel)
    if not os.path.exists(folder_path):
        print(f"⚠️ Skipping missing folder: {folder_path}")
        continue
        
    print(f"📂 Processing {label}...")
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            features = extract_features(file_path)
            if features:
                data_rows.append([filename, features[0], features[1], label])

# Save to CSV
df = pd.DataFrame(data_rows, columns=['Filename', 'Jitter', 'Shimmer', 'Diagnosis'])
df.to_csv('modma_voice_features.csv', index=False)
print(f"✅ Success! Extracted features for {len(df)} files into 'modma_voice_features.csv'")