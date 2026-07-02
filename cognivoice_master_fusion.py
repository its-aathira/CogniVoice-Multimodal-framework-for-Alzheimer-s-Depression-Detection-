"""
cognivoice_master_fusion_FIXED.py
FIXES:
  1. Loads scalers and applies them before every predict()
  2. Returns a result dict instead of only printing
  3. True weighted-average fusion (not sequential gating)
  4. Voice can override EEG for Healthy path too
  5. Feature shape validation before predict
"""
import joblib
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ── Label convention (shared across ALL models) ────────────────────────────────
#   0 = Healthy
#   1 = Alzheimer's
#   2 = Depression
CLASS_MAP = {0: "Healthy", 1: "Alzheimer's", 2: "Depression"}

# EEG is the primary clinical biomarker. Voice acts as an auxiliary non-invasive signal.
# We heavily weight EEG to prevent Alzheimer's voice misdiagnosis from overturning clinical EEG results.
EEG_WEIGHT   = 0.85
VOICE_WEIGHT = 0.15

def _load(path, label):
    if os.path.exists(path):
        return joblib.load(path)
    raise FileNotFoundError(f"❌ Missing required file: {path}  ({label})")

def load_all_models():
    """Returns a dict with models + scalers. Call once at app startup."""
    return {
        # EEG
        "eeg_model":   _load("model_3class_eeg.pkl",       "EEG model"),
        "eeg_scaler":  _load("scaler_eeg.pkl",             "EEG scaler"),
        # Alzheimer's voice
        "alz_model":   _load("alzheimer_voice_model.pkl",  "Alz voice model"),
        "alz_scaler":  _load("scaler_alz_voice.pkl",       "Alz voice scaler"),
        "alz_cols":    _load("alz_voice_feature_cols.pkl", "Alz voice feature list"),
        # Depression voice
        "dep_model":   _load("depress_voice_model.pkl",    "Dep voice model"),
        "dep_scaler":  _load("scaler_depress_voice.pkl",   "Dep voice scaler"),
        "dep_cols":    _load("depress_voice_feature_cols.pkl", "Dep voice feature list"),
    }


def _safe_proba(model, scaler, features_raw, expected_n_features, label):
    """
    Scale → validate shape → predict_proba.
    Returns a probability array over [0,1,2] (padded with zeros for missing classes).
    """
    arr = np.array(features_raw, dtype=float).reshape(1, -1)

    if arr.shape[1] != expected_n_features:
        raise ValueError(
            f"{label}: expected {expected_n_features} features, got {arr.shape[1]}. "
            "Re-run training and extraction with matching parameters."
        )

    arr_scaled = scaler.transform(arr)
    proba_raw  = model.predict_proba(arr_scaled)[0]   # shape = (n_classes_seen,)
    classes    = model.classes_

    # Expand to full 3-class vector [P(0), P(1), P(2)]
    full_proba = np.zeros(3)
    for idx, cls in enumerate(classes):
        full_proba[cls] = proba_raw[idx]

    return full_proba          # [P(Healthy), P(Alzheimer's), P(Depression)]


def perform_fusion_diagnosis(bundle, eeg_features_raw, voice_features_raw=None):
    """
    Args:
        bundle            : dict returned by load_all_models()
        eeg_features_raw  : list/array of 4 raw (un-scaled) EEG features
        voice_features_raw: list/array of voice features (MFCCs + prosody),
                            or None if no audio was uploaded

    Returns:
        dict with keys:
            eeg_proba, eeg_pred, eeg_conf,
            voice_proba, voice_pred, voice_conf,
            final_proba, final_pred, final_diagnosis, final_conf,
            warnings (list of str)
    """
    result   = {}
    warnings = []

    # ── 1. EEG prediction ─────────────────────────────────────────────────────
    eeg_n = len(bundle["eeg_model"].feature_importances_)   # = 4
    eeg_proba = _safe_proba(
        bundle["eeg_model"], bundle["eeg_scaler"],
        eeg_features_raw, eeg_n, "EEG"
    )
    eeg_pred = int(np.argmax(eeg_proba))
    eeg_conf = float(eeg_proba[eeg_pred] * 100)

    result["eeg_proba"] = eeg_proba
    result["eeg_pred"]  = eeg_pred
    result["eeg_conf"]  = eeg_conf

    # ── 2. Voice prediction ───────────────────────────────────────────────────
    voice_proba = np.zeros(3)

    if voice_features_raw is not None:
        v_arr = np.array(voice_features_raw, dtype=float)

        # Try Alzheimer's voice model first
        alz_n = len(bundle["alz_cols"])
        dep_n = len(bundle["dep_cols"])

        if v_arr.shape[0] == alz_n:
            # Looks like MFCC-based (Alzheimer's model)
            try:
                raw_proba = _safe_proba(
                    bundle["alz_model"], bundle["alz_scaler"],
                    v_arr, alz_n, "Alz-Voice"
                )
                # alz model only knows classes 0 and 1 → map correctly
                voice_proba = raw_proba
            except ValueError as e:
                warnings.append(str(e))

        elif v_arr.shape[0] == dep_n:
            # Prosody-based (Depression model)
            try:
                raw_proba = _safe_proba(
                    bundle["dep_model"], bundle["dep_scaler"],
                    v_arr, dep_n, "Dep-Voice"
                )
                voice_proba = raw_proba
            except ValueError as e:
                warnings.append(str(e))

        else:
            warnings.append(
                f"Voice features shape {v_arr.shape[0]} doesn't match "
                f"Alz model ({alz_n}) or Dep model ({dep_n}). "
                "Voice signal skipped."
            )
    else:
        warnings.append("No voice file uploaded — EEG signal only.")

    voice_pred = int(np.argmax(voice_proba)) if voice_proba.sum() > 0 else -1
    voice_conf = float(np.max(voice_proba) * 100) if voice_proba.sum() > 0 else 0.0

    result["voice_proba"] = voice_proba
    result["voice_pred"]  = voice_pred
    result["voice_conf"]  = voice_conf

    # ── 3. Dynamic Confidence-Weighted fusion ─────────────────────────────────
    if voice_proba.sum() > 0:
        # Cube the confidences to heavily penalize uncertain models
        # Power of 4 allows a highly confident voice model to override an uncertain EEG
        eeg_c = (eeg_conf / 100.0) ** 4
        voice_c = (voice_conf / 100.0) ** 4
        
        # Apply baseline weights (EEG is still the primary clinical biomarker)
        eeg_dyn_w = eeg_c * EEG_WEIGHT
        voice_dyn_w = voice_c * VOICE_WEIGHT
        
        # Normalize weights to sum to 1
        total_w = eeg_dyn_w + voice_dyn_w
        if total_w > 0:
            eeg_dyn_w /= total_w
            voice_dyn_w /= total_w
        else:
            eeg_dyn_w, voice_dyn_w = EEG_WEIGHT, VOICE_WEIGHT
            
        final_proba = eeg_dyn_w * eeg_proba + voice_dyn_w * voice_proba
    else:
        # No voice → EEG alone
        final_proba = eeg_proba.copy()
        warnings.append("Fusion used EEG only (no valid voice signal).")

    final_pred = int(np.argmax(final_proba))
    final_conf = float(final_proba[final_pred] * 100)

    # ── 4. Diagnosis string ───────────────────────────────────────────────────
    emoji = {"Healthy": "🟢", "Alzheimer's": "🔴", "Depression": "🟠"}
    label = CLASS_MAP[final_pred]
    diagnosis = f"{emoji[label]} {label.upper()}"

    result["final_proba"]     = final_proba
    result["final_pred"]      = final_pred
    result["final_diagnosis"] = diagnosis
    result["final_conf"]      = final_conf
    result["warnings"]        = warnings

    return result