"""
Generate nested, seeded, stratified test subsets (200-ID and 100-ID)
drawn strictly from the 1,000-profile core test subset.
Stratification is maintained across ground_truth_label x is_non_traditional.
"""
import os
import json
import hashlib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

DATA_TEST_PATH = "experiments/data/dataset_test.csv"
SUBSET_1000_PATH = "experiments/data/test_subset_1000_ids.json"
SUBSET_200_PATH = "experiments/data/test_subset_200_ids.json"
SUBSET_100_PATH = "experiments/data/test_subset_100_ids.json"

def main():
    # 1. Load 1,000 subset IDs
    with open(SUBSET_1000_PATH, "r", encoding="utf-8") as f:
        data_1000 = json.load(f)
    ids_1000 = set(data_1000["profile_ids"])
    
    # 2. Load dataset_test and filter to 1,000 IDs
    df_test = pd.read_csv(DATA_TEST_PATH)
    df_1000 = df_test[df_test["profile_id"].isin(ids_1000)].copy().reset_index(drop=True)
    df_1000["stratum"] = df_1000["ground_truth_label"].astype(str) + "_" + df_1000["is_non_traditional"].astype(str)
    
    # 3. Draw 200-ID subset from the 1,000 subset
    seed_200 = 2026
    sss_200 = StratifiedShuffleSplit(n_splits=1, train_size=200, random_state=seed_200)
    idx_200, _ = next(sss_200.split(df_1000, df_1000["stratum"]))
    df_200 = df_1000.iloc[idx_200].copy().reset_index(drop=True)
    
    ids_200 = df_200["profile_id"].tolist()
    out_200 = {
        "seed": seed_200,
        "parent_subset": "experiments/data/test_subset_1000_ids.json",
        "stratification_keys": ["ground_truth_label", "is_non_traditional"],
        "n_sampled": len(ids_200),
        "threat_count": int((df_200["ground_truth_label"] == 1).sum()),
        "genuine_count": int((df_200["ground_truth_label"] == 0).sum()),
        "non_traditional_count": int((df_200["is_non_traditional"] == 1).sum()),
        "strata_distribution": df_200["stratum"].value_counts().to_dict(),
        "profile_ids": ids_200
    }
    content_200 = json.dumps(out_200, indent=2)
    with open(SUBSET_200_PATH, "w", encoding="utf-8") as f:
        f.write(content_200)
    hash_200 = hashlib.sha256(content_200.encode("utf-8")).hexdigest()
    
    # 4. Draw 100-ID subset from the 200 subset
    seed_100 = 2026
    sss_100 = StratifiedShuffleSplit(n_splits=1, train_size=100, random_state=seed_100)
    idx_100, _ = next(sss_100.split(df_200, df_200["stratum"]))
    df_100 = df_200.iloc[idx_100].copy().reset_index(drop=True)
    
    ids_100 = df_100["profile_id"].tolist()
    out_100 = {
        "seed": seed_100,
        "parent_subset": "experiments/data/test_subset_200_ids.json",
        "stratification_keys": ["ground_truth_label", "is_non_traditional"],
        "n_sampled": len(ids_100),
        "threat_count": int((df_100["ground_truth_label"] == 1).sum()),
        "genuine_count": int((df_100["ground_truth_label"] == 0).sum()),
        "non_traditional_count": int((df_100["is_non_traditional"] == 1).sum()),
        "strata_distribution": df_100["stratum"].value_counts().to_dict(),
        "profile_ids": ids_100
    }
    content_100 = json.dumps(out_100, indent=2)
    with open(SUBSET_100_PATH, "w", encoding="utf-8") as f:
        f.write(content_100)
    hash_100 = hashlib.sha256(content_100.encode("utf-8")).hexdigest()
    
    print("=== NESTED SUBSETS CREATED ===")
    print(f"200-ID Subset: Path={SUBSET_200_PATH}")
    print(f"  Seed: {seed_200}")
    print(f"  SHA-256: {hash_200}")
    print(f"  Threats: {out_200['threat_count']} ({out_200['threat_count']/200*100:.1f}%), Genuine: {out_200['genuine_count']} ({out_200['genuine_count']/200*100:.1f}%)")
    print(f"  Non-Traditional: {out_200['non_traditional_count']}")
    print(f"  Strata: {out_200['strata_distribution']}")
    print()
    print(f"100-ID Subset: Path={SUBSET_100_PATH}")
    print(f"  Seed: {seed_100}")
    print(f"  SHA-256: {hash_100}")
    print(f"  Threats: {out_100['threat_count']} ({out_100['threat_count']/100*100:.1f}%), Genuine: {out_100['genuine_count']} ({out_100['genuine_count']/100*100:.1f}%)")
    print(f"  Non-Traditional: {out_100['non_traditional_count']}")
    print(f"  Strata: {out_100['strata_distribution']}")

if __name__ == "__main__":
    main()

