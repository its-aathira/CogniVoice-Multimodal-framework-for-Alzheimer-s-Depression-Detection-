
import pandas as pd
import numpy as np

df = pd.read_csv('combined_eeg.csv')
print('Before scaling:')
print(df.groupby('label')[['theta_power','alpha_power','beta_power']].mean())

# Scale labels 0 and 1 up by 1e12 to match Depression scale
mask = df['label'].isin([0, 1])
df.loc[mask, 'theta_power'] = df.loc[mask, 'theta_power'] * 1e12
df.loc[mask, 'alpha_power'] = df.loc[mask, 'alpha_power'] * 1e12
df.loc[mask, 'beta_power']  = df.loc[mask, 'beta_power']  * 1e12

# Recompute TAR with corrected values
df.loc[mask, 'theta_alpha_ratio'] = (
    df.loc[mask, 'theta_power'] / (df.loc[mask, 'alpha_power'] + 1e-6)
)

print()
print('After scaling:')
print(df.groupby('label')[['theta_power','alpha_power','beta_power','theta_alpha_ratio']].mean().round(4))

df.to_csv('combined_eeg_fixed.csv', index=False)
print()
print('Saved: combined_eeg_fixed.csv')
