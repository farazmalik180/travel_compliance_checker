"""
Ceiling Check: Evaluates ML baselines (Logistic Regression & HistGradientBoosting)
trained on dev split and evaluated on held-out test split.
Tests whether the dataset has realistic epistemic uncertainty or is trivially separable (AUC > 0.95).
Also computes the Oracle's demographic fairness gaps and FPR on non-traditional subgroup.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def run_ceiling_check():
    dev_path = "experiments/data/dataset_dev.csv"
    test_path = "experiments/data/dataset_test.csv"
    
    if not os.path.exists(dev_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Dev/Test CSVs not found. Run generate_dataset.py first.")
        
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)
    
    # Feature columns (Agent-observable features ONLY. Exclude latent, ground truth, and sensitive attributes)
    categorical_features = [
        "employment_type", "profession", "income_stability",
        "bank_funds_category", "passport_history", "destination",
        "visa_category", "purpose", "watchlist_flag"
    ]
    boolean_features = [
        "has_return_ticket", "has_hotel_booking",
        "has_financial_proof", "has_protector_stamp"
    ]
    numeric_features = [
        "bank_funds_pkr", "local_ties_score", "document_authenticity_score"
    ]
    
    all_features = categorical_features + boolean_features + numeric_features
    target = "ground_truth_label"
    
    X_train = dev_df[all_features].copy()
    y_train = dev_df[target].values
    
    X_test = test_df[all_features].copy()
    y_test = test_df[target].values
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("pass", "passthrough", boolean_features + numeric_features)
        ]
    )
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    print("=" * 60)
    print("CEILING CHECK: ML ORACLE ON AGENT-OBSERVABLE FEATURES")
    print("=" * 60)
    
    for name, clf in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Predict on Test
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.50).astype(int)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_overall = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Non-traditional subgroup FPR
        nt_mask = (test_df["is_non_traditional"] == 1) & (test_df["ground_truth_label"] == 0)
        nt_pred = y_pred[nt_mask]
        fpr_non_trad = nt_pred.mean() if len(nt_pred) > 0 else 0.0
        
        # Traditional subgroup FPR
        trad_mask = (test_df["is_non_traditional"] == 0) & (test_df["ground_truth_label"] == 0)
        trad_pred = y_pred[trad_mask]
        fpr_trad = trad_pred.mean() if len(trad_pred) > 0 else 0.0
        
        # Fairness gaps (Demographic Parity gap = max gap in offload rate)
        # Gender
        gender_rates = {g: y_pred[test_df["gender"] == g].mean() for g in test_df["gender"].unique()}
        dp_gender = max(gender_rates.values()) - min(gender_rates.values())
        
        # Age
        age_rates = {a: y_pred[test_df["age_band"] == a].mean() for a in test_df["age_band"].unique()}
        dp_age = max(age_rates.values()) - min(age_rates.values())
        
        # Region
        region_rates = {r: y_pred[test_df["region"] == r].mean() for r in test_df["region"].unique()}
        dp_region = max(region_rates.values()) - min(region_rates.values())
        
        results[name] = {
            "AUC": auc,
            "Accuracy": acc,
            "Recall": recall,
            "FPR_overall": fpr_overall,
            "FPR_non_traditional": fpr_non_trad,
            "FPR_traditional": fpr_trad,
            "DP_gender_gap": dp_gender,
            "DP_age_gap": dp_age,
            "DP_region_gap": dp_region
        }
        
        print(f"\n--- {name} ---")
        print(f"Test ROC-AUC:            {auc:.4f} (Ceiling check < 0.95: {auc < 0.95})")
        print(f"Test Accuracy:           {acc:.4f}")
        print(f"Test Recall (Threats):   {recall:.4f}")
        print(f"Test FPR (Overall):      {fpr_overall:.4f}")
        print(f"Test FPR (Non-Trad):     {fpr_non_trad:.4f}")
        print(f"Test FPR (Traditional):  {fpr_trad:.4f}")
        print(f"Demographic Parity Gaps:")
        print(f"  Gender gap:            {dp_gender:.4f}")
        print(f"  Age band gap:          {dp_age:.4f}")
        print(f"  Region gap:            {dp_region:.4f}")

    print("\n" + "=" * 60)
    return results

if __name__ == "__main__":
    run_ceiling_check()
