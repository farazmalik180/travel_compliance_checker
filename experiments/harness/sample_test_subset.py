import os
import json
import hashlib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

test_path = 'experiments/data/dataset_test.csv'
df = pd.read_csv(test_path)

# Stratify by ground_truth_label x is_non_traditional
df['stratum'] = df['ground_truth_label'].astype(str) + '_' + df['is_non_traditional'].astype(str)

seed = 2026
sss = StratifiedShuffleSplit(n_splits=1, train_size=1000, random_state=seed)
train_idx, _ = next(sss.split(df, df['stratum']))
subset_df = df.iloc[train_idx].copy().reset_index(drop=True)

profile_ids = subset_df['profile_id'].tolist()
out_dict = {
    'seed': seed,
    'source_file': 'experiments/data/dataset_test.csv',
    'stratification_keys': ['ground_truth_label', 'is_non_traditional'],
    'n_sampled': len(profile_ids),
    'strata_distribution': subset_df['stratum'].value_counts().to_dict(),
    'threat_count': int((subset_df['ground_truth_label'] == 1).sum()),
    'genuine_count': int((subset_df['ground_truth_label'] == 0).sum()),
    'non_traditional_count': int((subset_df['is_non_traditional'] == 1).sum()),
    'profile_ids': profile_ids
}

out_path = 'experiments/data/test_subset_1000_ids.json'
content = json.dumps(out_dict, indent=2)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(content)

sha256_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

print('Saved 1,000 test profile IDs to:', out_path)
print('Seed:', seed)
print('SHA-256:', sha256_hash)
print(f"Threat count: {out_dict['threat_count']} ({out_dict['threat_count']/len(subset_df)*100:.1f}%)")
print(f"Genuine count: {out_dict['genuine_count']} ({out_dict['genuine_count']/len(subset_df)*100:.1f}%)")
print(f"Non-traditional count: {out_dict['non_traditional_count']}")
print(f"Strata counts: {out_dict['strata_distribution']}")

