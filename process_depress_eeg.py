import os
import numpy as np
import pandas as pd
import scipy.io as sio
import mne

# --- CONFIGURATION ---
modma_eeg_path = r"/Users/aathirashibu/Documents/CogniVoice /data_depression/EEG Data/EEG Data"

print(f"🚀 Starting MODMA EEG Extraction from: {modma_eeg_path}")

all_features = []

# Verify folder exists
if not os.path.exists(modma_eeg_path):
    print("❌ Error: Path not found. Please check your folder name.")
    exit()

for filename in os.listdir(modma_eeg_path):
    if filename.endswith(".mat"):
        file_path = os.path.join(modma_eeg_path, filename)
        
        try:
            # 1. Load MATLAB file
            mat_contents = sio.loadmat(file_path)
            
            # 2. Extract from EEGLAB structure
            # Based on your error, the data is inside an object with fields
            # We need to find the main key (usually 'EEG')
            main_key = [k for k in mat_contents.keys() if not k.startswith('__')][0]
            struct = mat_contents[main_key]
            
            # The structure fields are accessed by index. 
            # In EEGLAB .mat files, 'data' is usually the 15th field (index 15)
            # or we can look for it by name in the dtype
            names = struct.dtype.names
            data_idx = names.index('data')
            srate_idx = names.index('srate')
            
            eeg_data = struct[0, 0][data_idx]
            sfreq = float(struct[0, 0][srate_idx][0, 0])
            
            # 3. Ensure data is float64 and handle shape
            eeg_data = eeg_data.astype(np.float64)
            if eeg_data.ndim == 3: # If trials x channels x points
                eeg_data = eeg_data[0] # Take first trial
            
            # 4. Convert to MNE (Multiply by 1e-6 to convert microvolts to Volts)
            eeg_data = eeg_data * 1e-6
            info = mne.create_info(ch_names=eeg_data.shape[0], sfreq=sfreq, ch_types='eeg')
            raw = mne.io.RawArray(eeg_data, info, verbose=False)

            # 5. Extract Biomarkers
            theta = raw.compute_psd(fmin=4, fmax=8, verbose=False).get_data().mean()
            alpha = raw.compute_psd(fmin=8, fmax=12, verbose=False).get_data().mean()
            beta = raw.compute_psd(fmin=13, fmax=30, verbose=False).get_data().mean()
            tar = theta / alpha

            all_features.append({
                "subject": filename,
                "theta_power": theta,
                "alpha_power": alpha,
                "beta_power": beta,
                "theta_alpha_ratio": tar,
                "label": 2  # 2 = Depression
            })
            print(f"✅ Successfully processed: {filename}")

        except Exception as e:
            print(f"❌ Error on {filename}: {e}")

# Save to CSV
if all_features:
    df = pd.DataFrame(all_features)
    
    # NEW: Safety check to create the directory if it doesn't exist
    output_dir = "."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created missing directory: {output_dir}")

    output_path = os.path.join(output_dir, "depression_eeg_features.csv")
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*40)
    print(f"🎉 FINAL SUCCESS! Extracted {len(df)} patients.")
    print(f"💾 File: {output_path}")
    print("="*40)