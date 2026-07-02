import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generate_patient_report import extract_features_and_raw, plot_clinical_heatmap

# --- 1. LOAD THE MODELS ---
# Path to your three-class EEG model and your specialized Alzheimer's Voice model
EEG_MODEL_PATH = "/Users/aathirashibu/Documents/CogniVoice /model_3class_eeg.pkl"
ALZ_VOICE_MODEL_PATH = "/Users/aathirashibu/Documents/CogniVoice /alzheimer_voice_model.pkl"
KAGGLE_CSV_PATH = "/Users/aathirashibu/Documents/CogniVoice /data_alzheimers/addetector_dataset.csv"

print("⚙️ Initializing Alzheimer's Fusion System...")
eeg_model = joblib.load(EEG_MODEL_PATH)
alz_voice_model = joblib.load(ALZ_VOICE_MODEL_PATH)

def run_alzheimers_fusion(eeg_file_path, patient_index):
    print("\n" + "="*50)
    print("🔬 COGNIVOICE: ALZHEIMER'S MULTIMODAL ANALYSIS")
    print("="*50)

    # --- STEP A: REAL-TIME EEG STREAM ---
    print(f"🧠 Processing EEG Scan: {os.path.basename(eeg_file_path)}...")
    eeg_feats, raw = extract_features_and_raw(eeg_file_path)
    
    if eeg_feats is not None:
        eeg_pred = eeg_model.predict(eeg_feats)[0] # 1 = Alzheimer's
        eeg_prob = np.max(eeg_model.predict_proba(eeg_feats)) * 100
    else:
        print("❌ Error: EEG Processing Failed.")
        return

    # --- STEP B: LINGUISTIC & ACOUSTIC STREAM (Kaggle Lookup) ---
    print(f"📂 Accessing Patient Feature Profile (Row: {patient_index})...")
    if os.path.exists(KAGGLE_CSV_PATH):
        df_alz = pd.read_csv(KAGGLE_CSV_PATH)
        df_alz.columns = df_alz.columns.str.strip() # Clean column names
        
        try:
            # Select patient data using .iloc (Row Index)
            patient_row = df_alz.iloc[[patient_index]]
            
            # Extract features used in training (MFCC 1-13 + Linguistic 1-50)
            feature_cols = [c for c in df_alz.columns if 'mfcc' in c.lower() or 'linguistic' in c.lower()]
            X_voice = patient_row[feature_cols]
            
            v_pred = alz_voice_model.predict(X_voice)[0] # 1 = Alzheimer's
            v_prob = np.max(alz_voice_model.predict_proba(X_voice)) * 100
            
            # Get filename for report
            fname = patient_row['Filename'].values[0] if 'Filename' in df_alz.columns else f"Patient_{patient_index}"
        except Exception as e:
            print(f"❌ Error indexing Kaggle dataset: {e}")
            return
    else:
        print("❌ Error: Kaggle dataset CSV not found.")
        return

    # --- STEP C: LATE FUSION LOGIC ---
    print("\n" + "-"*20 + " RESULTS " + "-"*20)
    print(f"Neural (EEG) Finding:   {'Alzheimer’s' if eeg_pred == 1 else 'Healthy'} ({eeg_prob:.1f}%)")
    print(f"Vocal/Ling. Finding:  {'Alzheimer’s' if v_pred == 1 else 'Healthy'} ({v_prob:.1f}%)")

    print("\n--- 🏁 FINAL CLINICAL DECISION ---")
    
    if eeg_pred == 1 and v_pred == 1:
        status = "🔴 CONFIRMED ALZHEIMER'S DISEASE"
        insight = "High convergence between Neural Spectral Slowing (TAR) and Linguistic complexity decline."
    elif eeg_pred == 1 or v_pred == 1:
        status = "🟡 SUSPECTED EARLY-STAGE ALZHEIMER'S"
        insight = "Modality Mismatch. Neuro-markers detected but behavioral symptoms are inconsistent."
    else:
        status = "🟢 HEALTHY CONTROL"
        insight = "All biomarkers within normal clinical limits."

    print(f"DIAGNOSIS: {status}")
    print(f"INSIGHT: {insight}")
    print("="*50)

    # --- STEP D: VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_clinical_heatmap(raw, ax, f"CogniVoice Report: {fname}")
    plt.show()

# --- 4. TEST EXECUTION ---
if __name__ == "__main__":
    # Update these for your live demo
    SAMPLE_EEG = "/Users/aathirashibu/Documents/CogniVoice /data_alzheimers/alzheimers_egg/derivatives/sub-012/eeg/sub-012_task-eyesclosed_eeg.set"
    CHOSEN_INDEX = 15 # Change this to any row number (0 to ~100) from your Kaggle CSV
    
    if os.path.exists(SAMPLE_EEG):
        run_alzheimers_fusion(SAMPLE_EEG, CHOSEN_INDEX)
    else:
        print(f"⚠️ Test file not found: {SAMPLE_EEG}")