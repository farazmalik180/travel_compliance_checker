"""
Run Deterministic Rule Engine (S0) and Supervised ML Baselines
on all 4,000 Test Profiles (and nested subsets: 1,000, 200, 100).
Requires zero LLM API calls ($0.00 compute).
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

DATA_DEV_PATH = "experiments/data/dataset_dev.csv"
DATA_TEST_PATH = "experiments/data/dataset_test.csv"
RESULTS_DIR = "experiments/results"
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "test_baselines_4000_predictions.csv")

def check_statutory_violations(row):
    violations = []
    if row.get("watchlist_flag") == "Verified Watchlist Alert":
        violations.append("Verified Watchlist Alert")
    vcat = row.get("visa_category")
    if vcat == "WORK":
        if not row.get("has_protector_stamp", False):
            violations.append("Missing Protector Stamp (WORK)")
    elif vcat == "VISIT":
        if not row.get("has_return_ticket", False):
            violations.append("Missing Return Ticket (VISIT)")
        if not row.get("has_hotel_booking", False):
            violations.append("Missing Hotel Booking (VISIT)")
        if not row.get("has_financial_proof", False):
            violations.append("Missing Financial Proof (VISIT)")
    elif vcat == "BUSINESS":
        if not row.get("has_return_ticket", False):
            violations.append("Missing Return Ticket (BUSINESS)")
        if not row.get("has_financial_proof", False):
            violations.append("Missing Financial Proof (BUSINESS)")
    return violations

def evaluate_s0(df):
    y_true = df["ground_truth_label"].values
    nt_flags = df["is_non_traditional"].values
    
    preds = []
    has_viols = []
    for _, row in df.iterrows():
        v = check_statutory_violations(row.to_dict())
        has_v = len(v) > 0
        has_viols.append(has_v)
        preds.append(1 if has_v else 0)
        
    preds = np.array(preds)
    has_viols = np.array(has_viols)
    
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    tn = np.sum((preds == 0) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Discretionary FPR
    gen_mask = (y_true == 0)
    gen_no_viol = gen_mask & (~has_viols)
    disc_fpr = np.sum((preds == 1) & gen_no_viol) / np.sum(gen_no_viol) if np.sum(gen_no_viol) > 0 else 0.0
    
    # Hard Threat Leakage Rate
    thr_viol = (y_true == 1) & has_viols
    leak_rate = np.sum((preds == 0) & thr_viol) / np.sum(thr_viol) if np.sum(thr_viol) > 0 else 0.0
    
    # Demographic FPR disparity
    n_nt_gen = np.sum(gen_mask & (nt_flags == 1))
    n_tr_gen = np.sum(gen_mask & (nt_flags == 0))
    fpr_nt = np.sum((preds == 1) & gen_mask & (nt_flags == 1)) / n_nt_gen if n_nt_gen > 0 else 0
    fpr_tr = np.sum((preds == 1) & gen_mask & (nt_flags == 0)) / n_tr_gen if n_tr_gen > 0 else 0
    delta_fpr = fpr_nt - fpr_tr
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "FPR": fpr,
        "Discretionary_FPR": disc_fpr,
        "Hard_Threat_Leakage": leak_rate,
        "FPR_NonTrad": fpr_nt,
        "FPR_Trad": fpr_tr,
        "Delta_FPR_NT": delta_fpr,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "predictions": preds
    }

def get_roc_point(y_true, scores, target_recall, nt_flags):
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    target_tp = int(np.ceil(target_recall * n_pos))
    
    desc_idx = np.argsort(scores)[::-1]
    y_sorted = y_true[desc_idx]
    cum_tp = np.cumsum(y_sorted == 1)
    idx = np.searchsorted(cum_tp, target_tp)
    threshold = scores[desc_idx[min(idx, len(scores)-1)]]
    
    preds = (scores >= threshold).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    tn = np.sum((preds == 0) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    
    gen_mask = (y_true == 0)
    n_nt_gen = np.sum(gen_mask & (nt_flags == 1))
    n_tr_gen = np.sum(gen_mask & (nt_flags == 0))
    
    fpr_nt = np.sum((preds == 1) & gen_mask & (nt_flags == 1)) / n_nt_gen if n_nt_gen > 0 else 0
    fpr_tr = np.sum((preds == 1) & gen_mask & (nt_flags == 0)) / n_tr_gen if n_tr_gen > 0 else 0
    
    return {
        "threshold": threshold,
        "recall": tp / n_pos if n_pos > 0 else 0,
        "fpr": fp / n_neg if n_neg > 0 else 0,
        "accuracy": (tp + tn) / len(y_true),
        "fpr_nt": fpr_nt,
        "fpr_trad": fpr_tr,
        "delta_fpr": fpr_nt - fpr_tr,
        "preds": preds
    }

def run_baselines():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_dev = pd.read_csv(DATA_DEV_PATH)
    df_test = pd.read_csv(DATA_TEST_PATH)
    
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
    
    X_dev = df_dev[all_features].copy()
    y_dev = df_dev[target].values
    
    X_test = df_test[all_features].copy()
    y_test = df_test[target].values
    nt_test = df_test["is_non_traditional"].values
    
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("pass", "passthrough", boolean_features + numeric_features)
    ])
    
    print("="*85)
    print("1. EVALUATION ON FULL TEST SET (N = 4,000 Profiles)")
    print("="*85)
    
    # 1. Evaluate S0 on Full Test
    s0_res = evaluate_s0(df_test)
    print("\n--- S0: Deterministic Statutory Rule Engine (Full N=4,000) ---")
    print(f"Accuracy: {s0_res['Accuracy']*100:.2f}% | Precision: {s0_res['Precision']*100:.2f}% | Recall: {s0_res['Recall']*100:.2f}%")
    print(f"Overall FPR: {s0_res['FPR']*100:.2f}% (TP={s0_res['TP']}, FP={s0_res['FP']}, TN={s0_res['TN']}, FN={s0_res['FN']})")
    print(f"Discretionary FPR: {s0_res['Discretionary_FPR']*100:.2f}% (Procedural rule engine never offloads on discretion)")
    print(f"Hard-Rule Threat Leakage: {s0_res['Hard_Threat_Leakage']*100:.2f}% (Statutory threats cleared)")
    print(f"FPR Non-Traditional: {s0_res['FPR_NonTrad']*100:.2f}% | FPR Traditional: {s0_res['FPR_Trad']*100:.2f}% | Delta FPR: {s0_res['Delta_FPR_NT']*100:.2f}%")
    
    # 2. Evaluate ML Baselines on Full Test
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
    }
    
    pred_dict = {
        "profile_id": df_test["profile_id"].values,
        "ground_truth_label": y_test,
        "is_non_traditional": nt_test,
        "s0_pred": s0_res["predictions"]
    }
    
    for name, clf in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_dev, y_dev)
        scores = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, scores)
        
        pt50 = get_roc_point(y_test, scores, 0.50, nt_test)
        pt70 = get_roc_point(y_test, scores, 0.70, nt_test)
        
        safe_name = name.lower().replace(" ", "_")
        pred_dict[f"{safe_name}_score"] = scores
        pred_dict[f"{safe_name}_pred_r70"] = pt70["preds"]
        
        print(f"\n--- {name} (Full N=4,000) ---")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"At Recall = 0.50: Overall FPR = {pt50['fpr']*100:.2f}% | Non-Trad FPR = {pt50['fpr_nt']*100:.2f}% | Trad FPR = {pt50['fpr_trad']*100:.2f}% | Delta = {pt50['delta_fpr']*100:.2f}%")
        print(f"At Recall = 0.70: Overall FPR = {pt70['fpr']*100:.2f}% | Non-Trad FPR = {pt70['fpr_nt']*100:.2f}% | Trad FPR = {pt70['fpr_trad']*100:.2f}% | Delta = {pt70['delta_fpr']*100:.2f}%")

    # Save predictions
    df_preds = pd.DataFrame(pred_dict)
    df_preds.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nSaved all 4,000 baseline predictions to: {PREDICTIONS_PATH}")

    # Subsets performance
    subsets = {
        "1,000 Core Test Subset": "experiments/data/test_subset_1000_ids.json",
        "200 Core Test Subset": "experiments/data/test_subset_200_ids.json",
        "100 Core Test Subset": "experiments/data/test_subset_100_ids.json"
    }
    
    for s_name, s_path in subsets.items():
        if os.path.exists(s_path):
            with open(s_path, "r", encoding="utf-8") as f:
                s_ids = set(json.load(f)["profile_ids"])
            sub_df = df_test[df_test["profile_id"].isin(s_ids)].copy().reset_index(drop=True)
            s0_sub = evaluate_s0(sub_df)
            print(f"\n==================== {s_name.upper()} (N = {len(sub_df)}) ====================")
            print(f"S0 Rule Engine -> Accuracy: {s0_sub['Accuracy']*100:.2f}%, Recall: {s0_sub['Recall']*100:.2f}%, FPR: {s0_sub['FPR']*100:.2f}%, Delta FPR NT: {s0_sub['Delta_FPR_NT']*100:.2f}%")
            
            sub_preds = df_preds[df_preds["profile_id"].isin(s_ids)]
            y_sub = sub_preds["ground_truth_label"].values
            nt_sub = sub_preds["is_non_traditional"].values
            
            for m_safe, m_title in [("logistic_regression", "Logistic Regression"), ("histgradientboosting", "HistGradientBoosting")]:
                auc_sub = roc_auc_score(y_sub, sub_preds[f"{m_safe}_score"])
                pt70_sub = get_roc_point(y_sub, sub_preds[f"{m_safe}_score"].values, 0.70, nt_sub)
                print(f"{m_title} -> ROC-AUC: {auc_sub:.4f}, FPR at R=0.70: {pt70_sub['fpr']*100:.2f}%, Delta FPR NT: {pt70_sub['delta_fpr']*100:.2f}%")

if __name__ == "__main__":
    run_baselines()

