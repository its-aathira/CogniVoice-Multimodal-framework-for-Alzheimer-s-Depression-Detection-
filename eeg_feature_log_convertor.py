
import pandas as pd
import numpy as np

df = pd.read_csv('combined_eeg_fixed.csv')

# Log scale compresses the 1000x difference
df['theta_power'] = np.log1p(df['theta_power'])
df['alpha_power'] = np.log1p(df['alpha_power'])
df['beta_power']  = np.log1p(df['beta_power'])

print('After log scaling:')
print(df.groupby('label')[['theta_power','alpha_power','beta_power','theta_alpha_ratio']].mean().round(4))

df.to_csv('combined_eeg_logscaled.csv', index=False)
print('Saved: combined_eeg_logscaled.csv')
