"""
Compute Supervised ML Baselines directly from Test ROC curve (with bootstrap CIs)
and from Out-Of-Fold (5-fold CV) Dev-calibrated thresholds.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score

def compute_ml_metrics():
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

    X_dev = dev_df[all_features].copy()
    y_dev = dev_df[target].values
    nt_dev = dev_df["is_non_traditional"].values

    X_test = test_df[all_features].copy()
    y_test = test_df[target].values
    nt_test = test_df["is_non_traditional"].values

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

    def get_roc_operating_point(y_true, scores, target_recall, nt_flags):
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        target_tp = int(np.ceil(target_recall * n_pos))
        
        desc_indices = np.argsort(scores)[::-1]
        y_sorted = y_true[desc_indices]
        
        cum_tp = np.cumsum(y_sorted == 1)
        idx = np.searchsorted(cum_tp, target_tp)
        threshold_at_recall = scores[desc_indices[min(idx, len(scores)-1)]]
        
        preds = (scores >= threshold_at_recall).astype(int)
        
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tn = np.sum((preds == 0) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        
        recall = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        acc = (tp + tn) / len(y_true)
        
        genuine_mask = (y_true == 0)
        n_nt_gen = np.sum(genuine_mask & (nt_flags == 1))
        n_trad_gen = np.sum(genuine_mask & (nt_flags == 0))
        
        fpr_nt = np.sum((preds == 1) & genuine_mask & (nt_flags == 1)) / n_nt_gen if n_nt_gen > 0 else 0
        fpr_trad = np.sum((preds == 1) & genuine_mask & (nt_flags == 0)) / n_trad_gen if n_trad_gen > 0 else 0
        delta_fpr = fpr_nt - fpr_trad
        
        return {
            "threshold": threshold_at_recall,
            "recall": recall,
            "fpr": fpr,
            "accuracy": acc,
            "fpr_nt": fpr_nt,
            "fpr_trad": fpr_trad,
            "delta_fpr": delta_fpr
        }

    def bootstrap_operating_point(y_true, scores, target_recall, nt_flags, n_boot=1000, rng_seed=42):
        rng = np.random.default_rng(rng_seed)
        n = len(y_true)
        records = []
        for _ in range(n_boot):
            boot_idx = rng.integers(0, n, size=n)
            rec = get_roc_operating_point(y_true[boot_idx], scores[boot_idx], target_recall, nt_flags[boot_idx])
            records.append(rec)
        df_boot = pd.DataFrame(records)
        return {
            "fpr_ci": (np.percentile(df_boot["fpr"], 2.5), np.percentile(df_boot["fpr"], 97.5)),
            "fpr_nt_ci": (np.percentile(df_boot["fpr_nt"], 2.5), np.percentile(df_boot["fpr_nt"], 97.5)),
            "fpr_trad_ci": (np.percentile(df_boot["fpr_trad"], 2.5), np.percentile(df_boot["fpr_trad"], 97.5)),
            "delta_fpr_ci": (np.percentile(df_boot["delta_fpr"], 2.5), np.percentile(df_boot["delta_fpr"], 97.5))
        }

    for name, clf in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_dev, y_dev)
        scores_test = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, scores_test)

        # 5-fold CV out-of-fold predictions on dev
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_scores_dev = cross_val_predict(pipe, X_dev, y_dev, cv=cv, method="predict_proba")[:, 1]
        
        # OOF dev threshold calibrated for 0.50 and 0.70 recall on dev positives
        pos_oof = np.sort(oof_scores_dev[y_dev == 1])
        # To get recall R, cutoff is at quantile (1-R) of positive scores
        th_oof_50 = pos_oof[int(len(pos_oof) * (1 - 0.50))]
        th_oof_70 = pos_oof[int(len(pos_oof) * (1 - 0.70))]

        # Test results using OOF-calibrated dev thresholds
        preds_oof_50 = (scores_test >= th_oof_50).astype(int)
        recall_oof_50 = np.sum((preds_oof_50 == 1) & (y_test == 1)) / np.sum(y_test == 1)
        fpr_oof_50 = np.sum((preds_oof_50 == 1) & (y_test == 0)) / np.sum(y_test == 0)

        preds_oof_70 = (scores_test >= th_oof_70).astype(int)
        recall_oof_70 = np.sum((preds_oof_70 == 1) & (y_test == 1)) / np.sum(y_test == 1)
        fpr_oof_70 = np.sum((preds_oof_70 == 1) & (y_test == 0)) / np.sum(y_test == 0)

        # Primary Threshold-free: Exact Test ROC curve operating points with 1,000 bootstrap CIs
        pt_50 = get_roc_operating_point(y_test, scores_test, 0.50, nt_test)
        boot_50 = bootstrap_operating_point(y_test, scores_test, 0.50, nt_test)

        pt_70 = get_roc_operating_point(y_test, scores_test, 0.70, nt_test)
        boot_70 = bootstrap_operating_point(y_test, scores_test, 0.70, nt_test)

        print(f"\n==================== {name} ====================")
        print(f"Test ROC-AUC: {auc:.4f}")
        print("\n--- [PRIMARY] Direct Test ROC Operating Points (Matched Recall with 1,000 Bootstrap 95% CIs) ---")
        print(f"Matched Recall = 0.50:")
        print(f"  Test Cutoff Score: {pt_50['threshold']:.4f}")
        print(f"  Test Recall: {pt_50['recall']*100:.2f}%")
        print(f"  Overall Test FPR: {pt_50['fpr']*100:.2f}% (95% CI: [{boot_50['fpr_ci'][0]*100:.2f}%, {boot_50['fpr_ci'][1]*100:.2f}%])")
        print(f"  FPR Non-Trad: {pt_50['fpr_nt']*100:.2f}% (95% CI: [{boot_50['fpr_nt_ci'][0]*100:.2f}%, {boot_50['fpr_nt_ci'][1]*100:.2f}%])")
        print(f"  FPR Traditional: {pt_50['fpr_trad']*100:.2f}% (95% CI: [{boot_50['fpr_trad_ci'][0]*100:.2f}%, {boot_50['fpr_trad_ci'][1]*100:.2f}%])")
        print(f"  Delta FPR (Non-Trad - Trad): {pt_50['delta_fpr']*100:.2f}% (95% CI: [{boot_50['delta_fpr_ci'][0]*100:.2f}%, {boot_50['delta_fpr_ci'][1]*100:.2f}%])")
        
        print(f"\nMatched Recall = 0.70:")
        print(f"  Test Cutoff Score: {pt_70['threshold']:.4f}")
        print(f"  Test Recall: {pt_70['recall']*100:.2f}%")
        print(f"  Overall Test FPR: {pt_70['fpr']*100:.2f}% (95% CI: [{boot_70['fpr_ci'][0]*100:.2f}%, {boot_70['fpr_ci'][1]*100:.2f}%])")
        print(f"  FPR Non-Trad: {pt_70['fpr_nt']*100:.2f}% (95% CI: [{boot_70['fpr_nt_ci'][0]*100:.2f}%, {boot_70['fpr_nt_ci'][1]*100:.2f}%])")
        print(f"  FPR Traditional: {pt_70['fpr_trad']*100:.2f}% (95% CI: [{boot_70['fpr_trad_ci'][0]*100:.2f}%, {boot_70['fpr_trad_ci'][1]*100:.2f}%])")
        print(f"  Delta FPR (Non-Trad - Trad): {pt_70['delta_fpr']*100:.2f}% (95% CI: [{boot_70['delta_fpr_ci'][0]*100:.2f}%, {boot_70['delta_fpr_ci'][1]*100:.2f}%])")

        print("\n--- [SECONDARY] Dev Out-Of-Fold (5-Fold CV) Calibrated Thresholds Applied to Test ---")
        print(f"OOF Dev Threshold for R=0.50 (tau={th_oof_50:.4f}) -> Test Recall: {recall_oof_50*100:.2f}%, Test FPR: {fpr_oof_50*100:.2f}%")
        print(f"OOF Dev Threshold for R=0.70 (tau={th_oof_70:.4f}) -> Test Recall: {recall_oof_70*100:.2f}%, Test FPR: {fpr_oof_70*100:.2f}%")

if __name__ == "__main__":
    compute_ml_metrics()
