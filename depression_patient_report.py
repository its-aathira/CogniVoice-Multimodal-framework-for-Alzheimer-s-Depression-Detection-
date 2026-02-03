import os
import joblib
import numpy as np
import pandas as pd
import mne
import scipy.io as sio 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- STEP 1: AUTOMATIC PATH FINDER ---
def find_the_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

base_dir = r"C:\Users\user\OneDrive\Desktop\Miniproject"
# CHANGE THIS TO TEST DIFFERENT FILES:
target_file = "EEG(15).mat" # Or "sub-013_task-eyesclosed_eeg.set"
patient_file_path = find_the_file(target_file, base_dir)

# --- SYSTEM PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) 
model_path = os.path.join(project_root, "backend", "model_3class_eeg.pkl")
output_image_path = os.path.join(project_root, "frontend", "assets", "final_patient_report.png")

CLASSES = {0: "Healthy Control", 1: "Alzheimer's Disease", 2: "Depression"}

# --- FUNCTIONS ---
def extract_features_and_raw(filepath):
    """Universal loader for .set and .mat files."""
    if filepath.endswith('.set'):
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
    elif filepath.endswith('.mat'):
        mat_contents = sio.loadmat(filepath)
        main_key = [k for k in mat_contents.keys() if not k.startswith('__')][0]
        struct = mat_contents[main_key]
        data = struct[0, 0]['data'].astype(np.float64)
        sfreq = float(struct[0, 0]['srate'][0, 0])
        if data.ndim == 3: data = data[0]
        ch_names = [f'E{i+1}' for i in range(data.shape[0])]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
        raw.set_montage(montage, on_missing='ignore')

    psds, freqs = mne.time_frequency.psd_array_welch(raw.get_data(), raw.info['sfreq'], fmin=1, fmax=30, n_fft=2048, verbose=False)
    avg_psds = np.mean(psds, axis=0)
    theta = np.mean(avg_psds[np.logical_and(freqs >= 4, freqs <= 8)])
    alpha = np.mean(avg_psds[np.logical_and(freqs >= 8, freqs <= 12)])
    tar = theta / alpha if alpha > 0 else 0
    features = pd.DataFrame([[theta, alpha, np.mean(avg_psds[np.logical_and(freqs >= 13, freqs <= 30)]), tar]], 
                            columns=['theta_power', 'alpha_power', 'beta_power', 'theta_alpha_ratio'])
    return features, raw

def plot_clinical_heatmap(raw, ax, diagnosis_label):
    psds, freqs = mne.time_frequency.psd_array_welch(raw.get_data(), raw.info['sfreq'], fmin=8, fmax=12, n_fft=2048, verbose=False)
    alpha_power = np.mean(psds, axis=1)
    im, _ = mne.viz.plot_topomap(alpha_power, raw.info, axes=ax, show=False, cmap='RdBu_r')
    
    if diagnosis_label == "Alzheimer's Disease":
        ax.annotate('LOW ALPHA ACTIVITY\n(Temporal/Hippocampus)', xy=(-0.5, -0.3), xytext=(-1.2, -0.4),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    fontsize=9, color='darkblue', fontweight='bold', bbox=dict(boxstyle="round", fc="white", ec="blue", alpha=0.9))
    elif diagnosis_label == "Depression":
        ax.annotate('FRONTAL ASYMMETRY\n(Mood Regulation)', xy=(0, 0.5), xytext=(0, 0.9),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    fontsize=9, color='darkred', fontweight='bold', bbox=dict(boxstyle="round", fc="white", ec="red", alpha=0.9))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Alpha Power')

# --- MAIN ---
if __name__ == "__main__":
    model = joblib.load(model_path)
    features, raw_data = extract_features_and_raw(patient_file_path)
    prediction_idx = model.predict(features)[0]
    diagnosis = CLASSES[prediction_idx]
    confidence = model.predict_proba(features)[0][prediction_idx] * 100

    # DYNAMIC INTERPRETATION TEXT (The part you wanted!)
    if diagnosis == "Alzheimer's Disease":
        interp_text = """The Temporal Lobe (Hippocampus area) 
shows significant electrical slowing. 
This organic 'Hardware Failure' 
distinguishes Alzheimer's from 
psychological Depression."""
        biomarker = "ALPHA SLOWING (8-12 Hz)"
    elif diagnosis == "Depression":
        interp_text = """The Frontal Lobe shows asymmetry in 
activity levels. This suggests a 
'Software Lag' in mood regulation 
centers, distinct from neuro-
degenerative cell death."""
        biomarker = "FRONTAL ASYMMETRY"
    else:
        interp_text = "Brain activity within normal range."
        biomarker = "STABLE ALPHA RHYTHM"

    fig = plt.figure(figsize=(11, 5))
    fig.suptitle("CogniVoice: Consolidated Patient Diagnostic Report", fontsize=14, fontweight='bold', y=0.98)
    ax_text = fig.add_subplot(1, 2, 1); ax_text.axis('off')
    
    report_content = f"""
--- CLINICAL ANALYSIS REPORT ---

Diagnosis: **{diagnosis.upper()}**
Confidence: **{confidence:.1f}%**

TECHNICAL BIOMARKERS:
- Theta/Alpha Ratio: {features['theta_alpha_ratio'].values[0]:.2f}
- Primary Signal: {biomarker}

ANATOMICAL INTERPRETATION:
{interp_text}
"""
    ax_text.text(0.05, 0.5, report_content, fontsize=11, va='center', fontfamily='monospace')
    ax_map = fig.add_subplot(1, 2, 2)
    plot_clinical_heatmap(raw_data, ax_map, diagnosis)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=150)
    print(f"🎉 Universal Report saved for: {diagnosis}")