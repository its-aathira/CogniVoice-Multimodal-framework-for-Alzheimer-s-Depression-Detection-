import os

# Update this if your folder name is different
modma_path = "/Users/aathirashibu/Documents/CogniVoice /data_depression" 

print(f"📂 Scanning: {modma_path}")

# Check EEG
eeg_dir = os.path.join(modma_path, "EEG Data")
if os.path.exists(eeg_dir):
    files = [f for f in os.listdir(eeg_dir) if not f.startswith('.')]
    print(f"\n🧠 Found {len(files)} EEG files.")
    if len(files) > 0:
        print(f"   Example: {files[0]}")
else:
    print("\n❌ Could not find 'EEG Data' folder. Check the name!")

# Check Audio
audio_dir = os.path.join(modma_path, "Audio_Dataset")
if os.path.exists(audio_dir):
    files = [f for f in os.listdir(audio_dir) if not f.startswith('.')]
    print(f"\n🎙️ Found {len(files)} Audio files.")
    if len(files) > 0:
        print(f"   Example: {files[0]}")
else:
    print("\n❌ Could not find 'Audio_Dataset' folder. Check the name!")