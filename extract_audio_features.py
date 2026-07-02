"""
extract_depression_audio_features.py

Extracts 19 clinical acoustic features from:
  - Normal/         → label 0 (Healthy)
  - Depression/Stage1/ + Stage2/ → label 2 (Depression)

Features (must match voice_feature_extractor_V2.py exactly):
  mfcc_1..mfcc_13, speech_rate, pitch_mean, pitch_std, jitter, shimmer, hnr

Run once to produce: depression_audio_features.csv
Then run: train_depress_voice_model_V2.py

Dependencies:
    pip install librosa parselmouth soundfile tqdm
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
import soundfile as sf
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR     = "/Users/aathirashibu/Documents/CogniVoice /data_depression/Audio_Dataset"
NORMAL_DIR   = os.path.join(BASE_DIR, "Normal")
DEP_DIRS     = [
    os.path.join(BASE_DIR, "Depression", "Stage1"),
    os.path.join(BASE_DIR, "Depression", "Stage2"),
]
OUTPUT_CSV   = "depression_audio_features.csv"
N_MFCC       = 13


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_praat_features(wav_path):
    try:
        snd   = parselmouth.Sound(wav_path)
        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        pp    = call(snd, "To PointProcess (periodic, cc)", 75, 600)

        jitter  = call(pp,        "Get jitter (local)",  0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr         = call(harmonicity, "Get mean", 0, 0)

        pitch_vals = pitch.selected_array['frequency']
        pitch_vals = pitch_vals[pitch_vals > 0]
        pitch_mean = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
        pitch_std  = float(np.std(pitch_vals))  if len(pitch_vals) > 0 else 0.0

        return float(jitter), float(shimmer), float(hnr), pitch_mean, pitch_std
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0


def extract_features(wav_path, label):
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)

        if len(y) < sr * 0.3:   # skip clips under 0.3s
            return None

        # MFCCs (1-indexed)
        mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        # Speech rate proxy
        speech_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        # Save temp wav for parselmouth
        tmp = "_tmp_dep.wav"
        sf.write(tmp, y, sr)
        jitter, shimmer, hnr, pitch_mean, pitch_std = extract_praat_features(tmp)
        if os.path.exists(tmp):
            os.remove(tmp)

        row = {f'mfcc_{i+1}': float(mfccs_mean[i]) for i in range(N_MFCC)}
        row.update({
            'speech_rate': speech_rate,
            'pitch_mean':  pitch_mean,
            'pitch_std':   pitch_std,
            'jitter':      jitter,
            'shimmer':     shimmer,
            'hnr':         hnr,
            'label':       label,
        })
        return row

    except Exception as e:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rows = []

    # ── Normal (Healthy) ──────────────────────────────────────────────────────
    normal_files = sorted(glob.glob(os.path.join(NORMAL_DIR, "*.wav")))
    print(f"📂 Normal (Healthy): {len(normal_files)} files")

    for fpath in tqdm(normal_files, desc="Normal"):
        row = extract_features(fpath, label=0)
        if row:
            rows.append(row)

    healthy_count = sum(1 for r in rows if r['label'] == 0)
    print(f"✅ Extracted: {healthy_count} Healthy rows")

    # ── Depression (Stage1 + Stage2) ──────────────────────────────────────────
    dep_files = []
    for d in DEP_DIRS:
        dep_files += sorted(glob.glob(os.path.join(d, "*.wav")))
    print(f"\n📂 Depression (Stage1+Stage2): {len(dep_files)} files")

    for fpath in tqdm(dep_files, desc="Depression"):
        row = extract_features(fpath, label=2)   # label=2 matches EEG model
        if row:
            rows.append(row)

    dep_count = sum(1 for r in rows if r['label'] == 2)
    print(f"✅ Extracted: {dep_count} Depression rows")

    # ── Save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n💾 Saved: {OUTPUT_CSV}")
    print(f"📊 Final distribution: {dict(df['label'].value_counts().sort_index())}")
    print(f"📐 Columns: {list(df.columns)}")
    print(f"\n📈 Feature means per class:")
    print(df.groupby('label')[['mfcc_1','mfcc_2','jitter','shimmer','hnr','pitch_mean']].mean().round(4))