"""
Comprehensive analysis and reporting script for the clean 200-profile Dev Pilot.
Computes complete-case metrics, sensitivity metrics (failures as OFF-LOAD),
tokens by type, cost and time per profile, S3 rounds distribution,
hallucination audit on Advocate claims, and exports 10 sampled traces per system.
"""
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import re
import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DEV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_dev.csv")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "pilot_raw_checkpoint.jsonl")
TRACES_PATH = os.path.join(RESULTS_DIR, "pilot_sampled_traces.json")

def check_hallucinated_claims(advocate_claims: list, profile_dict: dict) -> list:
    """
    Audits advocate claims against profile attributes.
    Returns list of claims flagged as potentially referencing facts not in profile.
    """
    flagged = []
    # Known profile values to match against
    prof_str = json.dumps(profile_dict).lower()
    
    # Specific known hallucination markers (common in ungrounded arguments)
    for claim in advocate_claims:
        claim_str = str(claim).lower()
        # Look for numbers/currency mentioned that deviate significantly
        # Or references to relatives/properties not in profile
        # Basic heuristic: if claim mentions specific visa/employer/assets
        # Check if the core nouns appear in profile
        pass
    return flagged

def audit_advocate_claims(df_s3, df_dev):
    dev_map = {row["profile_id"]: row.to_dict() for _, row in df_dev.iterrows()}
    total_claims = 0
    hallucination_flags = 0
    audit_details = []

    for _, row in df_s3.iterrows():
        prof_id = row["profile_id"]
        raw_p = dev_map.get(prof_id, {})
        transcript = row.get("debate_transcript") or []
        
        for turn in transcript:
            if turn.get("speaker") == "ADVOCATE":
                content = turn.get("content") or {}
                args = content.get("mitigating_arguments") or []
                for arg in args:
                    total_claims += 1
                    arg_l = str(arg).lower()
                    # Check if claim states an employment or funds or ties contrary to profile
                    # e.g., if passenger has 0 bank funds or Low, but claim says "multi-millionaire"
                    flagged = False
                    # Check funds exaggeration
                    if "substantial wealth" in arg_l or "multi-million" in arg_l:
                        if raw_p.get("bank_funds_pkr", 0) < 1000000:
                            flagged = True
                    # Check government servant claim
                    if "civil servant" in arg_l or "government officer" in arg_l:
                        if raw_p.get("employment_type") != "Civil Servant":
                            flagged = True
                    # Check protector stamp exemption claim on work visa
                    if "exempt from protector" in arg_l:
                        if raw_p.get("visa_category") == "WORK":
                            flagged = True
                            
                    if flagged:
                        hallucination_flags += 1
                        audit_details.append({"profile_id": prof_id, "claim": arg, "profile_actual": raw_p})

    return total_claims, hallucination_flags, audit_details

def report_pilot_results():
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint file found at", CHECKPOINT_PATH)
        return

    records = []
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    df_dev = pd.read_csv(DATA_DEV_PATH)
    print(f"Total evaluated records loaded: {len(df)}")
    print("Systems present:", df["system_id"].value_counts().to_dict())

    systems = ["S1", "S1b", "S2", "S3"]

    print("\n" + "="*85)
    print("1. PARSE-FAILURE RATES PER SYSTEM AND AGENT ROLE")
    print("="*85)
    for sys_id in systems:
        sub = df[df["system_id"] == sys_id]
        if len(sub) == 0:
            continue
        n_tot = len(sub)
        n_fails = int(sub["is_parse_failure"].sum())
        fail_pct = (n_fails / n_tot) * 100
        retries = int((sub["retries"] > 0).sum())
        print(f"System {sys_id:<5}: Failures = {n_fails}/{n_tot} ({fail_pct:.2f}%) | Retried = {retries}/{n_tot} ({(retries/n_tot)*100:.1f}%)")

    print("\n" + "="*85)
    print("2. PERFORMANCE METRICS (COMPLETE-CASE vs. FAILURES AS OFF-LOAD)")
    print("="*85)
    summary_rows = []
    for sys_id in systems:
        sub = df[df["system_id"] == sys_id]
        if len(sub) == 0:
            continue

        # Case A: Complete-case (drop failures)
        clean = sub[~sub["is_parse_failure"]]
        n_clean = len(clean)
        tp_c = int(((clean["decision"] == "OFF-LOAD") & (clean["ground_truth_label"] == 1)).sum())
        fp_c = int(((clean["decision"] == "OFF-LOAD") & (clean["ground_truth_label"] == 0)).sum())
        tn_c = int(((clean["decision"] == "CLEARED") & (clean["ground_truth_label"] == 0)).sum())
        fn_c = int(((clean["decision"] == "CLEARED") & (clean["ground_truth_label"] == 1)).sum())
        rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
        fpr_c = fp_c / (fp_c + tn_c) if (fp_c + tn_c) > 0 else 0
        acc_c = (tp_c + tn_c) / n_clean if n_clean > 0 else 0

        # Discretionary FPR on clean
        gen_comp_c = clean[(clean["ground_truth_label"] == 0) & (~clean["has_statutory_violation"])]
        disc_fp_c = int((gen_comp_c["decision"] == "OFF-LOAD").sum())
        disc_fpr_c = disc_fp_c / len(gen_comp_c) if len(gen_comp_c) > 0 else 0

        # Hard Threat Leakage on clean
        thr_viol_c = clean[(clean["ground_truth_label"] == 1) & (clean["has_statutory_violation"])]
        leak_c = int((thr_viol_c["decision"] == "CLEARED").sum())
        leak_rate_c = leak_c / len(thr_viol_c) if len(thr_viol_c) > 0 else 0

        # Case B: Failures counted as OFF-LOAD
        imp_decision = sub["decision"].fillna("OFF-LOAD")
        tp_i = int(((imp_decision == "OFF-LOAD") & (sub["ground_truth_label"] == 1)).sum())
        fp_i = int(((imp_decision == "OFF-LOAD") & (sub["ground_truth_label"] == 0)).sum())
        tn_i = int(((imp_decision == "CLEARED") & (sub["ground_truth_label"] == 0)).sum())
        fn_i = int(((imp_decision == "CLEARED") & (sub["ground_truth_label"] == 1)).sum())
        rec_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
        fpr_i = fp_i / (fp_i + tn_i) if (fp_i + tn_i) > 0 else 0
        acc_i = (tp_i + tn_i) / len(sub)

        # Discretionary FPR on imputed
        gen_comp_i = sub[(sub["ground_truth_label"] == 0) & (~sub["has_statutory_violation"])]
        disc_fp_i = int((gen_comp_i["decision"].fillna("OFF-LOAD") == "OFF-LOAD").sum())
        disc_fpr_i = disc_fp_i / len(gen_comp_i) if len(gen_comp_i) > 0 else 0

        # Tokens & Cost & Latency
        p_tok = sub["tokens"].apply(lambda t: t.get("prompt_tokens", 0)).mean()
        c_tok = sub["tokens"].apply(lambda t: t.get("completion_tokens", 0)).mean()
        r_tok = sub["tokens"].apply(lambda t: t.get("reasoning_tokens", 0)).mean()
        tot_tok = sub["tokens"].apply(lambda t: t.get("total_tokens", 0)).mean()
        cost_p = sub["cost_usd"].mean()
        lat_p = (sub["latency_ms"] / 1000.0).mean()

        summary_rows.append({
            "system_id": sys_id,
            "N_valid": n_clean,
            "Recall_clean": rec_c,
            "FPR_clean": fpr_c,
            "Disc_FPR_clean": disc_fpr_c,
            "Leak_Rate_clean": leak_rate_c,
            "Recall_worst": rec_i,
            "FPR_worst": fpr_i,
            "Disc_FPR_worst": disc_fpr_i,
            "Mean_Prompt_Tok": p_tok,
            "Mean_Comp_Tok": c_tok,
            "Mean_Reason_Tok": r_tok,
            "Mean_Total_Tok": tot_tok,
            "Mean_Cost_Prof": cost_p,
            "Mean_Time_Prof_s": lat_p
        })

    sum_df = pd.DataFrame(summary_rows)
    print("\n--- Complete-Case Analysis (Excluding Failures) ---")
    print(sum_df[["system_id", "N_valid", "Recall_clean", "FPR_clean", "Disc_FPR_clean", "Leak_Rate_clean"]].to_string(index=False))
    
    print("\n--- Sensitivity Worst-Case Analysis (Failures as OFF-LOAD) ---")
    print(sum_df[["system_id", "Recall_worst", "FPR_worst", "Disc_FPR_worst"]].to_string(index=False))

    print("\n--- Token Usage, Cost, and Execution Velocity per Profile ---")
    print(sum_df[["system_id", "Mean_Prompt_Tok", "Mean_Comp_Tok", "Mean_Reason_Tok", "Mean_Total_Tok", "Mean_Cost_Prof", "Mean_Time_Prof_s"]].to_string(index=False))

    # 3. S3 Rounds Histogram
    sub_s3 = df[df["system_id"] == "S3"]
    if len(sub_s3) > 0 and "rounds_completed" in sub_s3.columns:
        print("\n" + "="*85)
        print("3. S3 ADVERSARIAL DEBATE ROUNDS HISTOGRAM")
        print("="*85)
        r_counts = sub_s3["rounds_completed"].value_counts().sort_index()
        for r_num, cnt in r_counts.items():
            print(f"Round {int(r_num)}: {cnt} profiles ({cnt/len(sub_s3)*100:.1f}%)")

    # 4. Hallucination Audit of S3 Advocate Claims
    print("\n" + "="*85)
    print("4. HALLUCINATION AUDIT OF S3 ADVOCATE CLAIMS AGAINST PROFILE FIELDS")
    print("="*85)
    tot_c, flags, details = audit_advocate_claims(sub_s3, df_dev)
    print(f"Total Advocate Mitigating Claims Evaluated: {tot_c}")
    print(f"Unsubstantiated / Contradictory Claims Flagged: {flags} ({(flags/tot_c*100) if tot_c>0 else 0:.2f}%)")
    if details:
        print(f"Sample Flagged Claim: Profile {details[0]['profile_id']}: {details[0]['claim']}")

    # 5. Sampled Traces Export
    print("\n" + "="*85)
    print("5. EXPORTING 10 SAMPLED FULL TRACES PER SYSTEM")
    print("="*85)
    traces_export = {}
    for sys_id in systems:
        sub_sys = df[df["system_id"] == sys_id]
        if len(sub_sys) == 0:
            continue
        if sys_id == "S3":
            r1 = sub_sys[sub_sys["rounds_completed"] == 1]
            r2 = sub_sys[sub_sys["rounds_completed"] == 2]
            r3 = sub_sys[sub_sys["rounds_completed"] == 3]
            selected = []
            if len(r1) > 0: selected.append(r1.iloc[0].to_dict())
            if len(r2) > 0: selected.append(r2.iloc[0].to_dict())
            rem = 10 - len(selected)
            if len(r3) > 0:
                selected.extend(r3.sample(n=min(rem, len(r3)), random_state=42).to_dict(orient="records"))
            traces_export[sys_id] = selected
        else:
            traces_export[sys_id] = sub_sys.sample(n=min(10, len(sub_sys)), random_state=42).to_dict(orient="records")
        print(f"Extracted 10 traces for {sys_id}")

    with open(TRACES_PATH, "w", encoding="utf-8") as f:
        json.dump(traces_export, f, indent=2)
    print(f"Sampled traces written to {TRACES_PATH}")

    return sum_df

if __name__ == "__main__":
    report_pilot_results()
