"""
Ceiling Check & Evaluation Harness:
1. True Latent Bayes Oracle (evaluated using latent intent score Z and true beta priors).
2. Supervised ML Baselines (Logistic Regression & HistGradientBoosting on agent-observable features).
3. Matched Recall thresholds (Recall = 0.50 and Recall = 0.70) selected strictly on DEV and evaluated on TEST.
4. Reference baselines: Always-Clear and Always-Offload.
5. Permutation noise floor for delta FPR disparity on DEV.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def compute_metrics_at_threshold(y_true, scores, threshold, is_non_trad=None):
    y_pred = (scores >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    fpr_nt = None
    fpr_trad = None
    delta_fpr = None
    if is_non_trad is not None:
        nt_genuine = (is_non_trad == 1) & (y_true == 0)
        trad_genuine = (is_non_trad == 0) & (y_true == 0)
        fpr_nt = y_pred[nt_genuine].mean() if nt_genuine.sum() > 0 else 0.0
        fpr_trad = y_pred[trad_genuine].mean() if trad_genuine.sum() > 0 else 0.0
        delta_fpr = fpr_nt - fpr_trad
        
    return {
        "threshold": threshold,
        "accuracy": acc,
        "recall": recall,
        "fpr": fpr,
        "fpr_non_trad": fpr_nt,
        "fpr_trad": fpr_trad,
        "delta_fpr_nt": delta_fpr
    }

def find_threshold_for_target_recall(y_dev, scores_dev, target_recall):
    """
    Finds threshold on dev set that achieves target recall (e.g. 0.50 or 0.70) monotonically.
    """
    # Sort unique scores in descending order
    unique_scores = np.sort(np.unique(scores_dev))[::-1]
    best_thresh = 0.5
    min_diff = float("inf")
    
    for th in unique_scores:
        y_pred = (scores_dev >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_dev, y_pred, labels=[0, 1]).ravel()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        diff = abs(rec - target_recall)
        if diff < min_diff:
            min_diff = diff
            best_thresh = th
        # If we passed target recall, break
        if rec >= target_recall:
            break
            
    return float(best_thresh)

def compute_permutation_noise_floor(y_dev, is_non_trad_dev, scores_dev, threshold, n_perms=1000, seed=42):
    """
    Permutes is_non_traditional among genuine travelers on dev to establish
    the 95th percentile empirical noise floor for delta FPR.
    """
    np.random.seed(seed)
    genuine_mask = (y_dev == 0)
    nt_genuine = is_non_trad_dev[genuine_mask]
    y_pred_genuine = (scores_dev[genuine_mask] >= threshold).astype(int)
    
    delta_list = []
    for _ in range(n_perms):
        shuffled_nt = np.random.permutation(nt_genuine)
        fpr_nt = y_pred_genuine[shuffled_nt == 1].mean()
        fpr_trad = y_pred_genuine[shuffled_nt == 0].mean()
        delta_list.append(abs(fpr_nt - fpr_trad))
        
    return np.percentile(delta_list, 95)

def run_ceiling_check():
    dev_df = pd.read_csv("experiments/data/dataset_dev.csv")
    test_df = pd.read_csv("experiments/data/dataset_test.csv")
    
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
    nt_train = dev_df["is_non_traditional"].values
    z_train = dev_df["latent_intent_score"].values
    
    X_test = test_df[all_features].copy()
    y_test = test_df[target].values
    nt_test = test_df["is_non_traditional"].values
    z_test = test_df["latent_intent_score"].values
    
    # 1. TRIVIAL REFERENCE BASELINES
    acc_always_clear = (y_test == 0).mean()
    acc_always_offload = (y_test == 1).mean()
    
    print("=" * 70)
    print("1. TRIVIAL REFERENCE BASELINES (EVALUATED ON TEST SET, N=4,000)")
    print("=" * 70)
    print(f"Always-Clear Baseline:   Accuracy = {acc_always_clear*100:.2f}%, Recall = 0.00%, FPR = 0.00%")
    print(f"Always-Offload Baseline: Accuracy = {acc_always_offload*100:.2f}%, Recall = 100.00%, FPR = 100.00%")
    
    # 2. TRUE LATENT BAYES ORACLE
    # P(Y*=1 | Z=z) using true Beta prior densities
    def true_bayes_prob(z_arr):
        f0 = beta_dist.pdf(z_arr, 2.0, 4.5)
        f1 = beta_dist.pdf(z_arr, 4.2, 2.8)
        p1 = 0.20
        p0 = 0.80
        num = p1 * f1
        den = p1 * f1 + p0 * f0
        return num / den

    oracle_scores_dev = true_bayes_prob(z_train)
    oracle_scores_test = true_bayes_prob(z_test)
    oracle_auc = roc_auc_score(y_test, oracle_scores_test)
    
    # Select Oracle thresholds on Dev
    th_oracle_50 = find_threshold_for_target_recall(y_train, oracle_scores_dev, 0.50)
    th_oracle_70 = find_threshold_for_target_recall(y_train, oracle_scores_dev, 0.70)
    
    m_oracle_50 = compute_metrics_at_threshold(y_test, oracle_scores_test, th_oracle_50, nt_test)
    m_oracle_70 = compute_metrics_at_threshold(y_test, oracle_scores_test, th_oracle_70, nt_test)
    
    print("\n" + "=" * 70)
    print("2. TRUE LATENT BAYES ORACLE (UPPER BOUND THEORETICAL LIMIT)")
    print("=" * 70)
    print(f"True Latent Bayes Oracle ROC-AUC: {oracle_auc:.4f}")
    print(f"  At Matched Recall = 0.50 (Dev threshold={th_oracle_50:.4f}):")
    print(f"    Test Recall: {m_oracle_50['recall']*100:.2f}%, Test FPR: {m_oracle_50['fpr']*100:.2f}%, Accuracy: {m_oracle_50['accuracy']*100:.2f}%")
    print(f"    FPR (Non-Trad): {m_oracle_50['fpr_non_trad']*100:.2f}%, FPR (Trad): {m_oracle_50['fpr_trad']*100:.2f}%, Delta FPR: {m_oracle_50['delta_fpr_nt']*100:.2f}%")
    print(f"  At Matched Recall = 0.70 (Dev threshold={th_oracle_70:.4f}):")
    print(f"    Test Recall: {m_oracle_70['recall']*100:.2f}%, Test FPR: {m_oracle_70['fpr']*100:.2f}%, Accuracy: {m_oracle_70['accuracy']*100:.2f}%")
    print(f"    FPR (Non-Trad): {m_oracle_70['fpr_non_trad']*100:.2f}%, FPR (Trad): {m_oracle_70['fpr_trad']*100:.2f}%, Delta FPR: {m_oracle_70['delta_fpr_nt']*100:.2f}%")

    # 3. SUPERVISED ML BASELINES ON OBSERVABLES ONLY
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
    
    print("\n" + "=" * 70)
    print("3. SUPERVISED ML BASELINES (AGENT-OBSERVABLE FEATURES ONLY)")
    print("=" * 70)
    
    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        
        scores_dev = pipe.predict_proba(X_train)[:, 1]
        scores_test = pipe.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, scores_test)
        
        # Dev thresholds
        th_50 = find_threshold_for_target_recall(y_train, scores_dev, 0.50)
        th_70 = find_threshold_for_target_recall(y_train, scores_dev, 0.70)
        
        m_50 = compute_metrics_at_threshold(y_test, scores_test, th_50, nt_test)
        m_70 = compute_metrics_at_threshold(y_test, scores_test, th_70, nt_test)
        
        # Noise floor on dev at th_50
        noise_floor_50 = compute_permutation_noise_floor(y_train, nt_train, scores_dev, th_50)
        
        print(f"\n--- {name} ---")
        print(f"Test ROC-AUC: {auc:.4f} (Ceiling target < 0.95: {auc < 0.95})")
        print(f"Permutation Noise Floor for Delta FPR (Dev 95% CI): +/- {noise_floor_50*100:.2f}%")
        print(f"  At Matched Recall = 0.50 (Dev threshold={th_50:.4f}):")
        print(f"    Test Recall: {m_50['recall']*100:.2f}%, Test FPR: {m_50['fpr']*100:.2f}%, Accuracy: {m_50['accuracy']*100:.2f}%")
        print(f"    FPR (Non-Trad): {m_50['fpr_non_trad']*100:.2f}%, FPR (Trad): {m_50['fpr_trad']*100:.2f}%, Delta FPR: {m_50['delta_fpr_nt']*100:.2f}%")
        print(f"  At Matched Recall = 0.70 (Dev threshold={th_70:.4f}):")
        print(f"    Test Recall: {m_70['recall']*100:.2f}%, Test FPR: {m_70['fpr']*100:.2f}%, Accuracy: {m_70['accuracy']*100:.2f}%")
        print(f"    FPR (Non-Trad): {m_70['fpr_non_trad']*100:.2f}%, FPR (Trad): {m_70['fpr_trad']*100:.2f}%, Delta FPR: {m_70['delta_fpr_nt']*100:.2f}%")
        
    print("=" * 70)

if __name__ == "__main__":
    run_ceiling_check()
