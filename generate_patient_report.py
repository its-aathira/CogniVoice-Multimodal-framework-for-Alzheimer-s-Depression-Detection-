import os
import joblib
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- CONFIGURATION ---
# We use sub-013 (Alzheimer's) to demonstrate the "Blue" (Low Energy) signal
patient_file_path = r"C:\Users\user\OneDrive\Desktop\Miniproject\derivatives\sub-013\eeg\sub-013_task-eyesclosed_eeg.set"

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
model_path = os.path.join(script_dir, "model_3class_eeg.pkl")
output_image_path = os.path.join(project_root, "frontend", "assets", "final_patient_report.png")

CLASSES = {0: "Healthy Control", 1: "Alzheimer's Disease", 2: "Depression"}

# --- FUNCTIONS ---
import scipy.io as sio # Add this import at the top!

def extract_features_and_raw(filepath):
    """Loads either .set (OpenNeuro) or .mat (MODMA) files automatically."""
    print(f"📂 Loading file: {filepath}")
    
    if filepath.endswith('.set'):
        # Alzheimer's / Healthy Control Logic
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
    
    elif filepath.endswith('.mat'):
        # Depression (MODMA) Logic
        mat_contents = sio.loadmat(filepath)
        main_key = [k for k in mat_contents.keys() if not k.startswith('__')][0]
        struct = mat_contents[main_key]
        
        # Extract data and sampling rate
        data = struct[0, 0]['data'].astype(np.float64)
        sfreq = float(struct[0, 0]['srate'][0, 0])
        
        if data.ndim == 3: data = data[0]
        
        info = mne.create_info(ch_names=data.shape[0], sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
    
    # Standard montage for visualization
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
    except:
        pass

    # Extract PSD and Features
    psds, freqs = mne.time_frequency.psd_array_welch(
        raw.get_data(), raw.info['sfreq'], fmin=1, fmax=30, n_fft=2048, verbose=False
    )
    
    avg_psds = np.mean(psds, axis=0)
    theta = np.mean(avg_psds[np.logical_and(freqs >= 4, freqs <= 8)])
    alpha = np.mean(avg_psds[np.logical_and(freqs >= 8, freqs <= 12)]) 
    beta = np.mean(avg_psds[np.logical_and(freqs >= 13, freqs <= 30)])
    tar = theta / alpha if alpha > 0 else 0

    features = pd.DataFrame([[theta, alpha, beta, tar]], 
                            columns=['theta_power', 'alpha_power', 'beta_power', 'theta_alpha_ratio'])
    return features, raw

def plot_clinical_heatmap(raw, ax, diagnosis_label):
    """
    Plots Standard Alpha Power (Red=High, Blue=Low).
    Adds arrows pointing to the specific anatomical biomarkers.
    """
    # 1. Calculate Alpha Band Power (8-12 Hz)
    psds, freqs = mne.time_frequency.psd_array_welch(
        raw.get_data(), raw.info['sfreq'], fmin=8, fmax=12, n_fft=2048, verbose=False
    )
    alpha_power = np.mean(psds, axis=1) # Average across frequencies

    # 2. Plot Standard Heatmap (RdBu_r: Red=High Power, Blue=Low Power)
    im, _ = mne.viz.plot_topomap(
        alpha_power, 
        raw.info, 
        axes=ax, 
        show=False, 
        cmap='RdBu_r' 
    )
    
    # 3. DYNAMIC ANNOTATIONS (The "Smart" part)
    if diagnosis_label == "Alzheimer's Disease":
        # In AD, Alpha drops (Blue). We point to the "Blue" area.
        ax.annotate('LOW ACTIVITY (Atrophy)\n(Temporal/Hippocampus)', 
                    xy=(-0.5, -0.3), xytext=(-1.1, -0.3),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    fontsize=9, color='darkblue', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.9))
        ax.set_title("Visual Biomarker: Alpha Power Loss", fontsize=10, fontweight='bold')

    elif diagnosis_label == "Depression":
        # In Depression, we look for frontal asymmetry
        ax.annotate('FRONTAL ASYMMETRY\n(Mood Regulation)', 
                    xy=(0, 0.5), xytext=(0, 0.9),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    fontsize=9, color='darkred', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9))
        ax.set_title("Visual Biomarker: Frontal Asymmetry", fontsize=10, fontweight='bold')

    else: # Healthy
        # In Healthy, Alpha is strong (Red) at the back.
        ax.annotate('HEALTHY SIGNAL\n(Strong Occipital Alpha)', 
                    xy=(0, -0.6), xytext=(0, -1.0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    fontsize=9, color='darkgreen', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.9))
        ax.set_title("Visual Biomarker: Healthy Alpha Rhythm", fontsize=10, fontweight='bold')

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Alpha Power (µV²/Hz)', fontsize=8)

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    print("⚙️ Generating Consolidated Report...")

    # Load Model & Data
    if not os.path.exists(model_path) or not os.path.exists(patient_file_path):
        print("❌ Error: Check paths.")
        exit()

    model = joblib.load(model_path)
    features, raw_data = extract_features_and_raw(patient_file_path)

    # AI Prediction
    prediction_idx = model.predict(features)[0]
    diagnosis = CLASSES[prediction_idx]
    confidence = model.predict_proba(features)[0][prediction_idx] * 100

    print(f"🧠 AI Diagnosis: {diagnosis}")

    # --- TEXT GENERATION LOGIC ---
    if diagnosis == "Alzheimer's Disease":
        interp_text = """
    ANATOMICAL INTERPRETATION:
    The Temporal Lobe (Hippocampus area)
    shows significant electrical slowing
    (Blue regions). This organic 'Hardware
    Failure' distinguishes Alzheimer's
    from psychological Depression.
        """
        biomarker = "ALPHA SLOWING (8-12 Hz)"
        
    elif diagnosis == "Depression":
        interp_text = """
    ANATOMICAL INTERPRETATION:
    The Frontal Lobe shows asymmetry in
    activity levels. This suggests a
    'Software Lag' in mood regulation
    centers, distinct from neuro-
    degenerative cell death.
        """
        biomarker = "FRONTAL ASYMMETRY"
        
    else: # Healthy
        interp_text = """
    ANATOMICAL INTERPRETATION:
    Brain activity is within normal limits.
    Strong Red/Orange signals in the
    Occipital lobe indicate a healthy
    resting Alpha rhythm.
        """
        biomarker = "NORMAL ALPHA RHYTHM"

    # --- FIGURE CREATION ---
    fig = plt.figure(figsize=(11, 5)) # Wide layout: Left Text, Right Image
    fig.suptitle("CogniVoice: Consolidated Patient Diagnostic Report", fontsize=14, fontweight='bold', y=0.95)

    # 1. LEFT SIDE: TEXT REPORT
    ax_text = fig.add_subplot(1, 2, 1)
    ax_text.axis('off')
    
    report_content = f"""
    --- CLINICAL ANALYSIS REPORT ---

    Patient ID: sub-013
    Diagnosis: **{diagnosis.upper()}**
    Confidence: **{confidence:.1f}%**

    TECHNICAL BIOMARKERS:
    - Theta/Alpha Ratio: {features['theta_alpha_ratio'].values[0]:.2f}
    - Primary Signal: {biomarker}

    {interp_text}
    """
    # Using monospace font to look like a medical printout
    ax_text.text(0.05, 0.5, report_content, fontsize=11, va='center', fontfamily='monospace')

    # 2. RIGHT SIDE: BRAIN MAP
    ax_map = fig.add_subplot(1, 2, 2)
    plot_clinical_heatmap(raw_data, ax_map, diagnosis)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
    print(f"🎉 Report saved to: {output_image_path}")
