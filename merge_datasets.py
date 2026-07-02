import pandas as pd
import os

# Define paths
path_existing = "/Users/aathirashibu/Documents/CogniVoice /master_eeg_multiband.csv"
path_new = "/Users/aathirashibu/Documents/CogniVoice /depression_eeg_features.csv"
output_path = "/Users/aathirashibu/Documents/CogniVoice /combined_eeg.csv"

print("🔄 Merging Datasets...")

try:
    # 1. Load the files
    df_old = pd.read_csv(path_existing)
    df_new = pd.read_csv(path_new)

    # 2. Check for matching columns
    # We only care about the feature columns and the label
    required_cols = ['theta_power', 'alpha_power', 'beta_power', 'theta_alpha_ratio', 'label']
    
    # Filter both DFs to keep only relevant columns (avoids mismatched 'subject' formats)
    df_old = df_old[required_cols]
    df_new = df_new[required_cols]

    # 3. Combine
    df_final = pd.concat([df_old, df_new], ignore_index=True)

    # 4. Save
    df_final.to_csv(output_path, index=False)

    print("\n✅ MERGE SUCCESSFUL!")
    print(f"   - Total Samples: {len(df_final)}")
    print(f"   - Saved to: {output_path}")
    print("\nClass Distribution:")
    print(df_final['label'].value_counts().rename({0:'Healthy', 1:'Alzheimers', 2:'Depression'}))

except Exception as e:
    print(f"\n❌ Error merging files: {e}")
    print("Tip: Check if 'master_eeg_multiband_final.csv' exists in the data folder.")