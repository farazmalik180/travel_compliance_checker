"""
Comprehensive analysis of dev pilot progress:
- Parse failures per system
- S3 rounds histogram and calls per profile (mean, SD)
- Decision divergence from Prosecutor's R1 proposal
- Advocate concession rate
- Token means with n, SD, and SE per system
- Calibrated k for S1c with SE
- Pairwise system concordance and error correlation
"""
import json
import pandas as pd
import numpy as np

CHECKPOINT_PATH = "experiments/results/pilot_raw_checkpoint.jsonl"
DATA_DEV_PATH = "experiments/data/dataset_dev.csv"

def analyze():
    records = []
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    df = pd.DataFrame(records)
    print(f"Total evaluated checkpoint records: {len(df)}")
    
    systems = ["S1", "S1b", "S2", "S3"]
    
    print("\n" + "="*80)
    print("1. PARSE FAILURES PER SYSTEM")
    print("="*80)
    for s in systems:
        sub = df[df["system_id"] == s]
        n_tot = len(sub)
        n_fails = int(sub["is_parse_failure"].sum())
        pct = (n_fails / n_tot * 100.0) if n_tot > 0 else 0.0
        print(f"  {s:<5}: {n_fails}/{n_tot} parse failures ({pct:.2f}%)")
        
    print("\n" + "="*80)
    print("2. S3 ADVERSARIAL DEBATE METRICS")
    print("="*80)
    s3_df = df[df["system_id"] == "S3"]
    n_s3 = len(s3_df)
    
    # Rounds histogram
    rounds_counts = s3_df["rounds_completed"].value_counts().sort_index()
    print("Debate Rounds Distribution:")
    for r, cnt in rounds_counts.items():
        print(f"  Round {r}: {cnt}/{n_s3} profiles ({cnt/n_s3*100:.1f}%)")
        
    calls_list = s3_df["calls_count"].values
    mean_calls = float(np.mean(calls_list))
    sd_calls = float(np.std(calls_list, ddof=1)) if n_s3 > 1 else 0.0
    print(f"\nCalls per Profile: Mean = {mean_calls:.2f}, SD = {sd_calls:.2f} (Min = {np.min(calls_list)}, Max = {np.max(calls_list)})")
    
    # Prosecutor R1 divergence
    diff_pros_r1 = 0
    concession_count = 0
    for _, row in s3_df.iterrows():
        t = row.get("debate_transcript") or []
        if t:
            pros_r1 = t[0].get("content", {}).get("proposed_action")
            final_dec = row.get("decision")
            if pros_r1 != final_dec:
                diff_pros_r1 += 1
            # Check advocate concession
            pleas = [turn.get("content", {}).get("defense_plea") for turn in t if turn.get("speaker") == "ADVOCATE"]
            if any(p == "CONCEDE_OFFLOAD" for p in pleas):
                concession_count += 1
                
    print(f"\nProsecutor R1 Proposal Divergence:")
    print(f"  Final Decision Differs from Prosecutor R1: {diff_pros_r1}/{n_s3} ({diff_pros_r1/n_s3*100:.1f}%)")
    print(f"\nAdvocate Concession Rate:")
    print(f"  Advocate Conceded CONCEDE_OFFLOAD: {concession_count}/{n_s3} ({concession_count/n_s3*100:.1f}%)")
    
    print("\n" + "="*80)
    print("3. TOKEN MEANS, STANDARD DEVIATIONS & STANDARD ERRORS PER SYSTEM")
    print("="*80)
    token_stats = []
    for s in systems:
        sub = df[df["system_id"] == s]
        n_s = len(sub)
        p_tok = [t["prompt_tokens"] for t in sub["tokens"]]
        c_tok = [t["completion_tokens"] for t in sub["tokens"]]
        r_tok = [t["reasoning_tokens"] for t in sub["tokens"]]
        tot_tok = [t["total_tokens"] for t in sub["tokens"]]
        cost = sub["cost_usd"].values
        
        m_tot = float(np.mean(tot_tok))
        sd_tot = float(np.std(tot_tok, ddof=1)) if n_s > 1 else 0.0
        se_tot = sd_tot / np.sqrt(n_s) if n_s > 1 else 0.0
        
        token_stats.append({
            "system_id": s,
            "n": n_s,
            "mean_prompt": float(np.mean(p_tok)),
            "mean_comp": float(np.mean(c_tok)),
            "mean_reason": float(np.mean(r_tok)),
            "mean_total": m_tot,
            "sd_total": sd_tot,
            "se_total": se_tot,
            "mean_cost_usd": float(np.mean(cost))
        })
        
    df_tok = pd.DataFrame(token_stats)
    print(df_tok[["system_id", "n", "mean_prompt", "mean_comp", "mean_reason", "mean_total", "sd_total", "se_total", "mean_cost_usd"]].to_string(index=False))
    
    # 4. Calibrated k for S1c
    s1_mean = df_tok[df_tok["system_id"] == "S1"]["mean_total"].values[0]
    s1_sd = df_tok[df_tok["system_id"] == "S1"]["sd_total"].values[0]
    s1_n = df_tok[df_tok["system_id"] == "S1"]["n"].values[0]
    
    s3_mean = df_tok[df_tok["system_id"] == "S3"]["mean_total"].values[0]
    s3_sd = df_tok[df_tok["system_id"] == "S3"]["sd_total"].values[0]
    s3_n = df_tok[df_tok["system_id"] == "S3"]["n"].values[0]
    
    ratio = s3_mean / s1_mean
    # Standard error of ratio via Taylor expansion delta method:
    # SE(R) = R * sqrt( (SE_s3 / Mean_s3)^2 + (SE_s1 / Mean_s1)^2 )
    se_s3 = s3_sd / np.sqrt(s3_n)
    se_s1 = s1_sd / np.sqrt(s1_n)
    se_ratio = ratio * np.sqrt((se_s3 / s3_mean)**2 + (se_s1 / s1_mean)**2)
    k_int = int(np.round(ratio))
    
    print("\n" + "="*80)
    print("4. CALIBRATION OF S1c COMPUTE CONTROL (k parameter)")
    print("="*80)
    print(f"Mean Tokens S3: {s3_mean:.1f} (SD = {s3_sd:.1f}, n = {s3_n})")
    print(f"Mean Tokens S1: {s1_mean:.1f} (SD = {s1_sd:.1f}, n = {s1_n})")
    print(f"Exact Ratio (S3 / S1): {ratio:.3f} (SE = {se_ratio:.3f})")
    print(f"Calibrated Integer k: {k_int} calls (with 95% CI: [{ratio - 1.96*se_ratio:.2f}, {ratio + 1.96*se_ratio:.2f}])")
    
    # 5. Pairwise Concordance
    print("\n" + "="*80)
    print("5. PAIRWISE SYSTEM DECISION CONCORDANCE & ERROR CORRELATION")
    print("="*80)
    piv = df.pivot(index="profile_id", columns="system_id", values="decision").dropna()
    print(f"Profiles with complete evaluations across all 4 systems: n = {len(piv)}")
    matrix = []
    for s_a in systems:
        row = {"System": s_a}
        for s_b in systems:
            match = (piv[s_a] == piv[s_b]).mean()
            row[s_b] = f"{match*100:.1f}%"
        matrix.append(row)
    print(pd.DataFrame(matrix).to_string(index=False))

if __name__ == "__main__":
    analyze()

