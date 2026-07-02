"""
extract_alz_combined_features.py

Builds the Alzheimer's voice training dataset from two sources:

  Healthy (label=0):     Normal/*.wav from depression dataset
                         → full 19 features extracted live
  Alzheimer's (label=1): addetector_dataset.csv (mfcc_1..13 only)
                         → pitch/jitter/shimmer/hnr set to 0
                         (model still learns from MFCCs which carry signal)

Output: alz_combined_features.csv
Then run: alzh_voice_model_V3.py
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
NORMAL_DIR   = "/Users/aathirashibu/Documents/CogniVoice /data_depression/Audio_Dataset/Normal"
ALZ_CSV      = "/Users/aathirashibu/Documents/CogniVoice /data_alzheimers/addetector_dataset.csv"
OUTPUT_CSV   = "alz_combined_features.csv"
N_MFCC       = 13

FEATURE_COLS = (
    [f'mfcc_{i}' for i in range(1, N_MFCC + 1)] +
    ['speech_rate', 'pitch_mean', 'pitch_std', 'jitter', 'shimmer', 'hnr']
)  # 19 features — must match voice_feature_extractor_V2.py


# ── Praat feature extraction ──────────────────────────────────────────────────
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


def extract_from_wav(wav_path, label):
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        if len(y) < sr * 0.3:
            return None

        mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        speech_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        tmp = "_tmp_alz.wav"
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
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rows = []

    # ── 1. Healthy — extract from Normal .wav files ───────────────────────────
    normal_files = sorted(glob.glob(os.path.join(NORMAL_DIR, "*.wav")))
    print(f"📂 Healthy (Normal .wav): {len(normal_files)} files")

    for fpath in tqdm(normal_files, desc="Healthy"):
        row = extract_from_wav(fpath, label=0)
        if row:
            rows.append(row)

    healthy_count = sum(1 for r in rows if r['label'] == 0)
    print(f"✅ Healthy extracted: {healthy_count} rows")

    # ── 2. Alzheimer's — reuse MFCCs from CSV, zero out other features ────────
    print(f"\n📂 Alzheimer's (from CSV): {ALZ_CSV}")
    df_alz = pd.read_csv(ALZ_CSV)
    df_alz.columns = df_alz.columns.str.strip()

    # Keep only Alzheimer's rows
    df_alz = df_alz[df_alz['label'] == 1].reset_index(drop=True)
    print(f"📊 Alzheimer's rows in CSV: {len(df_alz)}")

    mfcc_cols = [f'mfcc_{i}' for i in range(1, N_MFCC + 1)]
    missing   = [c for c in mfcc_cols if c not in df_alz.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        print(f"   Available: {list(df_alz.columns)[:10]}")
        exit()

    for _, row_src in df_alz.iterrows():
        row = {f'mfcc_{i}': float(row_src[f'mfcc_{i}']) for i in range(1, N_MFCC + 1)}
        row.update({
            'speech_rate': 0.0,   # not available from CSV
            'pitch_mean':  0.0,
            'pitch_std':   0.0,
            'jitter':      0.0,
            'shimmer':     0.0,
            'hnr':         0.0,
            'label':       1,
        })
        rows.append(row)

    alz_count = sum(1 for r in rows if r['label'] == 1)
    print(f"✅ Alzheimer's rows added: {alz_count}")

    # ── 3. Save ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)[FEATURE_COLS + ['label']]
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n💾 Saved: {OUTPUT_CSV}")
    print(f"📊 Final distribution: {dict(df['label'].value_counts().sort_index())}")
    print(f"\n📈 Feature means per class:")
    print(df.groupby('label')[
        ['mfcc_1', 'mfcc_2', 'jitter', 'shimmer', 'hnr', 'pitch_mean']
    ].mean().round(4))
    print("\n⚠️  Note: Alzheimer's class has 0s for jitter/shimmer/hnr/pitch")
    print("   Model will learn to distinguish using MFCC features.")