import pandas as pd

import os
CSV = os.path.join(os.path.dirname(__file__) or '.', 'experiments_bace.csv')

def split_name(x):
    if not isinstance(x, str) or '_' not in x:
        return x, ''
    x_lower = x.lower()
    if x_lower.startswith('graphmae'):
        parts = x.split('_', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], ''
    parts = x.rsplit('_', 1)
    return parts[0], parts[1]


df = pd.read_csv(CSV)
df[['base_model', 'variant']] = df['experiment'].apply(lambda x: pd.Series(split_name(x)))
print(df[['experiment', 'base_model', 'variant']])
print('\nCounts per base_model:')
print(df['base_model'].value_counts())
print('\nVariants for base_model graphmae:')
print(df[df['base_model']=='graphmae'][['experiment','variant','emissions']])
