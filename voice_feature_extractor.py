"""
voice_feature_extractor.py

Single source of truth for live voice feature extraction.
Import this in cognivoice_app instead of inline extraction.

Features (must match training exactly):
  mfcc_1..mfcc_13, speech_rate, pitch_mean, pitch_std, jitter, shimmer, hnr

Dependencies:
    pip install librosa parselmouth soundfile
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import soundfile as sf
import os

N_MFCC = 13

FEATURE_COLS = (
    [f'mfcc_{i}' for i in range(1, N_MFCC + 1)] +
    ['speech_rate', 'pitch_mean', 'pitch_std', 'jitter', 'shimmer', 'hnr']
)


def _praat_features(wav_path: str):
    """Extract jitter, shimmer, HNR, pitch_mean, pitch_std via Praat."""
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

        return jitter, shimmer, float(hnr), pitch_mean, pitch_std

    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0


def extract_voice_features(wav_path: str) -> dict:
    """
    Extract all 19 clinical features from a .wav file.
    Returns a dict with keys matching FEATURE_COLS.
    """
    y_audio, sr = librosa.load(wav_path, sr=16000, mono=True)

    # MFCCs (1-indexed)
    mfccs      = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    feature_dict = {f'mfcc_{i+1}': float(mfccs_mean[i]) for i in range(N_MFCC)}

    # Speech rate proxy
    feature_dict['speech_rate'] = float(
        np.mean(librosa.feature.zero_crossing_rate(y_audio))
    )

    # Praat features
    tmp = "_tmp_feat.wav"
    sf.write(tmp, y_audio, sr)

    jitter, shimmer, hnr, pitch_mean, pitch_std = _praat_features(tmp)

    if os.path.exists(tmp):
        os.remove(tmp)

    feature_dict['pitch_mean'] = pitch_mean
    feature_dict['pitch_std']  = pitch_std
    feature_dict['jitter']     = jitter
    feature_dict['shimmer']    = shimmer
    feature_dict['hnr']        = hnr

    return feature_dict


def build_vector(feature_dict: dict, col_list: list):
    """
    Build ordered numpy array from feature_dict using col_list.
    Returns (array, missing_cols).
    """
    missing = [c for c in col_list if c not in feature_dict]
    if missing:
        return None, missing
    return np.array([feature_dict[c] for c in col_list], dtype=float), []