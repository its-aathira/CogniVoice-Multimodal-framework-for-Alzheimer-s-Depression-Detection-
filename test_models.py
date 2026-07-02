#to test if the models are working correctly
'''
import joblib
import pandas as pd

# Load your EEG dataset again
df = pd.read_csv("/Users/aathirashibu/Documents/CogniVoice /combined_eeg.csv")

X = df[['theta_power', 'alpha_power', 'beta_power', 'theta_alpha_ratio']]
y = df['label']

# Load model
model = joblib.load("model_3class_eeg.pkl")

# Test on training data
print("🔍 Testing EEG model on training data...\n")

for i in range(5):
    sample = X.iloc[i]
    actual = y.iloc[i]
    pred = model.predict([sample])[0]

    print(f"Sample {i}")
    print("Actual:", actual)
    print("Predicted:", pred)
    print("-"*30)
    print("This is the next part")
    print(df['label'].value_counts())
    for i in range(20):
        sample = X.iloc[i]
        actual = y.iloc[i]
        pred = model.predict([sample])[0]

        print(f"Actual: {actual}, Predicted: {pred}")


'''


'''
import pandas as pd
df = pd.read_csv('combined_eeg.csv')
print(df.groupby('label')[['theta_power','alpha_power','beta_power','theta_alpha_ratio']].mean().round(6))
print()
print(df.groupby('label')[['theta_power','alpha_power','beta_power','theta_alpha_ratio']].std().round(6))
'''

'''
from generate_patient_report import extract_features_and_raw
features, _ = extract_features_and_raw('temp_eeg.mat')
print('Extracted features:')
print(features)
'''

'''
import pandas as pd
df = pd.read_csv('combined_eeg.csv')
print(df[df['label']==0].head(3))
print()
print(df[df['label']==1].head(3))'''

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib 

model = joblib.load("model_3class_eeg.pkl")
# 1. Get feature importance from your Random Forest
importances = model.feature_importances_
feature_names = ['Theta', 'Alpha', 'Beta', 'TAR', 'MFCC_1', 'MFCC_2', 'Jitter', 'Shimmer'] # Add your full list here

# 2. Create a DataFrame for easier plotting
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# 3. Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')

plt.title('CogniVoice: Feature Importance Ranking', fontsize=14)
plt.xlabel('Gini Importance Score', fontsize=12)
plt.ylabel('Clinical Biomarker', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 4. Save for your report
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()