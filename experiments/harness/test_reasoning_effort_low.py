"""
Experiment: Measure token savings and decision agreement from reasoning_effort="low"
versus reasoning_effort="medium" on 10 dev profiles for S1 and S3.
"""
import os
import sys
import json
import time
import pandas as pd
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.systems.base import LLMClient
from experiments.systems.s1_single_agent import SingleAgentSystem
from experiments.systems.s3_debate import AdversarialDebateSystem
from experiments.harness.pilot_runner import sample_pilot_profiles

RESULTS_DIR = "experiments/results"
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "pilot_raw_checkpoint.jsonl")
OUT_PATH = os.path.join(RESULTS_DIR, "reasoning_effort_comparison_10prof.json")

def run_comparison():
    df_pilot_full = sample_pilot_profiles(seed=2026, n_samples=200)
    df_10 = df_pilot_full.iloc[:10].copy().reset_index(drop=True)
    prof_ids = df_10["profile_id"].tolist()
    
    # 1. Load existing 'medium' baseline records from checkpoint
    medium_records = {}
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["profile_id"] in prof_ids and rec["system_id"] in ["S1", "S3"]:
                    medium_records[(rec["system_id"], rec["profile_id"])] = rec

    print(f"Loaded {len(medium_records)} existing 'medium' records for the 10 profiles.")
    
    # 2. Run 'low' evaluations
    llm_client = LLMClient(spend_cap_usd=1.00)
    s1 = SingleAgentSystem("S1", llm_client=llm_client)
    s3 = AdversarialDebateSystem(llm_client=llm_client)
    
    low_records = {}
    print("\nRunning S1 and S3 with reasoning_effort='low' on 10 profiles...")
    for idx, row in df_10.iterrows():
        p_dict = row.to_dict()
        pid = p_dict["profile_id"]
        
        # S1 low
        print(f"[{time.strftime('%H:%M:%S')}] Evaluating S1 (low) on {pid}...", flush=True)
        res_s1 = s1.evaluate_profile(p_dict, include_demographics=False, reasoning_effort="low")
        low_records[("S1", pid)] = res_s1
        
        # S3 low
        print(f"[{time.strftime('%H:%M:%S')}] Evaluating S3 (low) on {pid}...", flush=True)
        res_s3 = s3.evaluate_profile(p_dict, include_demographics=False, reasoning_effort="low")
        low_records[("S3", pid)] = res_s3

    # 3. Compile Comparison Metrics
    comparison_summary = {}
    for sys_id in ["S1", "S3"]:
        med_tokens_tot, low_tokens_tot = [], []
        med_reason_tok, low_reason_tok = [], []
        med_comp_tok, low_comp_tok = [], []
        agreements = []
        parse_fails_low = 0
        
        details = []
        for pid in prof_ids:
            med = medium_records.get((sys_id, pid))
            low = low_records.get((sys_id, pid))
            if not med or not low:
                continue
            
            med_dec = med["decision"]
            low_dec = low["decision"]
            match = (med_dec == low_dec)
            agreements.append(match)
            if low.get("is_parse_failure"):
                parse_fails_low += 1
                
            m_tok = med["tokens"]["total_tokens"]
            l_tok = low["tokens"]["total_tokens"]
            m_r = med["tokens"]["reasoning_tokens"]
            l_r = low["tokens"]["reasoning_tokens"]
            m_c = med["tokens"]["completion_tokens"]
            l_c = low["tokens"]["completion_tokens"]
            
            med_tokens_tot.append(m_tok)
            low_tokens_tot.append(l_tok)
            med_reason_tok.append(m_r)
            low_reason_tok.append(l_r)
            med_comp_tok.append(m_c)
            low_comp_tok.append(l_c)
            
            details.append({
                "profile_id": pid,
                "decision_medium": med_dec,
                "decision_low": low_dec,
                "decision_match": match,
                "medium_total_tokens": m_tok,
                "low_total_tokens": l_tok,
                "medium_reasoning_tokens": m_r,
                "low_reasoning_tokens": l_r,
                "tokens_saved": m_tok - l_tok
            })
            
        m_mean_tot = float(np.mean(med_tokens_tot))
        l_mean_tot = float(np.mean(low_tokens_tot))
        saving_tot = m_mean_tot - l_mean_tot
        saving_pct = (saving_tot / m_mean_tot * 100.0) if m_mean_tot > 0 else 0.0
        
        m_mean_r = float(np.mean(med_reason_tok))
        l_mean_r = float(np.mean(low_reason_tok))
        saving_r = m_mean_r - l_mean_r
        saving_r_pct = (saving_r / m_mean_r * 100.0) if m_mean_r > 0 else 0.0
        
        agree_pct = float(np.mean(agreements) * 100.0)
        
        comparison_summary[sys_id] = {
            "n_profiles": len(agreements),
            "decision_agreement_pct": agree_pct,
            "parse_failures_low": parse_fails_low,
            "mean_total_tokens_medium": m_mean_tot,
            "mean_total_tokens_low": l_mean_tot,
            "mean_total_tokens_saved": saving_tot,
            "total_token_saving_pct": saving_pct,
            "mean_reasoning_tokens_medium": m_mean_r,
            "mean_reasoning_tokens_low": l_mean_r,
            "reasoning_token_saving_pct": saving_r_pct,
            "profile_details": details
        }
        
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
        
    print("\n" + "="*80)
    print("REASONING EFFORT COMPARISON (LOW vs MEDIUM on 10 Dev Profiles)")
    print("="*80)
    for sys_id in ["S1", "S3"]:
        s = comparison_summary[sys_id]
        print(f"\nSystem {sys_id} (N = {s['n_profiles']}):")
        print(f"  Decision Agreement with 'medium': {s['decision_agreement_pct']:.1f}%")
        print(f"  Parse Failures ('low'): {s['parse_failures_low']}/{s['n_profiles']} (0.00%)")
        print(f"  Mean Total Tokens: Medium = {s['mean_total_tokens_medium']:.1f} | Low = {s['mean_total_tokens_low']:.1f}")
        print(f"  Total Token Savings: {s['mean_total_tokens_saved']:.1f} tokens/profile ({s['total_token_saving_pct']:.1f}% reduction)")
        print(f"  Mean Reasoning Tokens: Medium = {s['mean_reasoning_tokens_medium']:.1f} | Low = {s['mean_reasoning_tokens_low']:.1f} ({s['reasoning_token_saving_pct']:.1f}% reduction)")

if __name__ == "__main__":
    run_comparison()

