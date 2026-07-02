import librosa
import numpy as np
import os

def extract_voice_biomarkers(audio_path):
    print(f"🎙️ Analyzing Audio: {os.path.basename(audio_path)}")
    
    # 1. Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # 2. Extract Fundamental Frequency (F0)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_clean = f0[~np.isnan(f0)]
    
    if len(f0_clean) == 0:
        return None

    # 3. Calculate Jitter (Frequency Instability)
    # Average absolute difference between consecutive periods
    diff_f0 = np.abs(np.diff(f0_clean))
    jitter = np.mean(diff_f0) / np.mean(f0_clean)
    
    # 4. Calculate Shimmer (Amplitude Instability)
    # We use the Root Mean Square (RMS) energy as a proxy for amplitude
    rms = librosa.feature.rms(y=y)[0]
    rms_clean = rms[rms > 0]
    diff_rms = np.abs(np.diff(rms_clean))
    shimmer = np.mean(diff_rms) / np.mean(rms_clean)
    
    return {
        "Jitter": round(jitter, 6),
        "Shimmer": round(shimmer, 6)
    }

if __name__ == "__main__":
    # Test with a MODMA wav file
    test_audio = "/Users/aathirashibu/Documents/CogniVoice /data_depression/Audio_Dataset/Depression/Stage1/OAF_burn_sad.wav" 
    if os.path.exists(test_audio):
        results = extract_voice_biomarkers(test_audio)
        print(f"✅ Biomarkers Extracted: {results}")
    else:
        print("❌ Audio file not found. Check your path.")