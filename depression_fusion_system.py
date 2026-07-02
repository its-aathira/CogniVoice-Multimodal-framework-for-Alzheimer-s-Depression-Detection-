import os
import joblib
import numpy as np
import librosa
import pandas as pd
from generate_patient_report import extract_features_and_raw # Your existing EEG loader

# --- 1. LOAD THE TRAINED MODELS ---
print("⚙️ Loading AI Models...")
try:
    eeg_model = joblib.load("/Users/aathirashibu/Documents/CogniVoice /model_3class_eeg.pkl")   # The Brain Model
    voice_model = joblib.load("/Users/aathirashibu/Documents/CogniVoice /depress_voice_model.pkl")      # The Voice Model
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# --- 2. DEFINE VOICE FEATURE EXTRACTOR (Single File) ---
def get_single_voice_features(audio_path):
    """Extracts Jitter and Shimmer for a single patient file."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Fundamental Frequency (F0) for Jitter
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) < 2: return None
        jitter = np.mean(np.abs(np.diff(f0_clean))) / np.mean(f0_clean)

        # Amplitude (RMS) for Shimmer
        rms = librosa.feature.rms(y=y)[0]
        rms_clean = rms[rms > 0]
        if len(rms_clean) < 2: return None
        shimmer = np.mean(np.abs(np.diff(rms_clean))) / np.mean(rms_clean)
        
        # Return as DataFrame to match training format
        return pd.DataFrame([[jitter, shimmer]], columns=['Jitter', 'Shimmer'])
    except:
        return None

# --- 3. THE FUSION ENGINE ---
def run_clinical_fusion(eeg_path, voice_path):
    print("\n" + "="*40)
    print(f"🏥 STARTING MULTIMODAL DIAGNOSIS")
    print("="*40)

    # --- STEP A: EEG ANALYSIS ---
    print(f"🧠 Analyzing Brain Activity: {os.path.basename(eeg_path)}...")
    eeg_feats, _ = extract_features_and_raw(eeg_path)
    
    if eeg_feats is not None:
        eeg_pred = eeg_model.predict(eeg_feats)[0]
        eeg_prob = np.max(eeg_model.predict_proba(eeg_feats)) * 100
    else:
        print("❌ EEG Processing Failed.")
        return

    # --- STEP B: VOICE ANALYSIS ---
    print(f"🎙️  Analyzing Voice Biomarkers: {os.path.basename(voice_path)}...")
    voice_feats = get_single_voice_features(voice_path)
    
    if voice_feats is not None:
        voice_pred = voice_model.predict(voice_feats)[0]
        voice_prob = np.max(voice_model.predict_proba(voice_feats)) * 100
    else:
        print("❌ Voice Processing Failed.")
        return

    # --- STEP C: LATE FUSION LOGIC ---
    # Class Mapping: 0=Healthy, 1=Alzheimer's, 2=Depression
    classes = {0: "Healthy", 1: "Alzheimer's", 2: "Depression"}
    
    eeg_result = classes.get(eeg_pred, "Unknown")
    voice_result = classes.get(voice_pred, "Unknown")

    print("\n--- INDIVIDUAL RESULTS ---")
    print(f"🔸 EEG Diagnosis:   {eeg_result} ({eeg_prob:.1f}% Confidence)")
    print(f"🔸 Voice Diagnosis: {voice_result} ({voice_prob:.1f}% Confidence)")

    print("\n--- 🏁 FINAL FUSION DECISION ---")
    
    # 1. BOTH AGREE -> HIGH CONFIDENCE
    if eeg_pred == voice_pred:
        final_decision = eeg_result
        if final_decision == "Depression":
            status = "🔴 CLINICAL DEPRESSION CONFIRMED (Major Depressive Disorder)"
            rec = "Recommendation: Immediate Psychiatric Evaluation & SSRI Therapy."
        elif final_decision == "Healthy":
            status = "🟢 HEALTHY (No Pathology Detected)"
            rec = "Recommendation: Routine checkup in 6 months."
        else:
            status = f"🟠 {final_decision.upper()} DETECTED"
            rec = "Recommendation: Neurology referral."

    # 2. DISAGREEMENT -> WEIGHTED LOGIC
    # Scenario: EEG sees Depression (Internal), Voice sees Healthy (Masking)
    elif eeg_pred == 2 and voice_pred == 0:
        status = "🟡 PHYSIOLOGICAL DEPRESSION (Early Stage / Masked)"
        rec = "Insight: Brain activity shows depression, but vocal motor control is intact. Patient may be 'masking' symptoms."

    # Scenario: Voice sees Depression (Symptom), EEG sees Healthy (Internal)
    elif eeg_pred == 0 and voice_pred == 2:
        status = "🟡 BEHAVIORAL DISTRESS (Psychological Stress)"
        rec = "Insight: Voice shows stress/fatigue markers, but neurological patterns are normal. Likely situational stress or fatigue, not clinical MDD."

    else:
        status = "⚪ INCONCLUSIVE / COMPLEX COMORBIDITY"
        rec = "Recommendation: Further multimodal testing required."

    print(f"Diagnosis: {status}")
    print(f"{rec}")
    print("="*40)

# --- 4. RUNNER ---
if __name__ == "__main__":
    # REPLACE THESE WITH REAL FILE PATHS FROM YOUR FOLDERS FOR THE DEMO
    test_eeg = "/Users/aathirashibu/Documents/CogniVoice /data_depression/EEG Data/EEG Data/EEG(7).mat"  # A Stage 2 EEG
    test_voice = "/Users/aathirashibu/Documents/CogniVoice /data_depression/Audio_Dataset/Depression/Stage2/YAF_bought_sad.wav" # A Stage 2 Audio
    
    # Check if files exist before running
    if os.path.exists(test_eeg) and os.path.exists(test_voice):
        run_clinical_fusion(test_eeg, test_voice)
    else:
        print("⚠️  Please update the 'test_eeg' and 'test_voice' paths in the script to valid files on your Mac!")