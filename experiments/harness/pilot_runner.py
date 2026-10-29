"""
Execution harness for the 200-profile Dev Pilot across S1, S1b, S2, and S3.
Features thread-safe concurrency (respecting Groq paid tier limits),
missing-value failure recording, live checkpointing, and a 2% parse-failure gate.
"""
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from experiments.systems.base import LLMClient, SpendCapExceededError
from experiments.systems.s1_single_agent import SingleAgentSystem
from experiments.systems.s2_hierarchical import HierarchicalSystem
from experiments.systems.s3_debate import AdversarialDebateSystem

DATA_DEV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_dev.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "pilot_raw_checkpoint.jsonl")
MANIFEST_PATH = os.path.join(RESULTS_DIR, "run_manifest.json")

checkpoint_lock = threading.Lock()

def check_statutory_violations(row: Dict[str, Any]) -> List[str]:
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

def sample_pilot_profiles(seed: int = 2026, n_samples: int = 200) -> pd.DataFrame:
    df_dev = pd.read_csv(DATA_DEV_PATH)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
    train_idx, _ = next(sss.split(df_dev, df_dev["ground_truth_label"]))
    df_pilot = df_dev.iloc[train_idx].copy().reset_index(drop=True)
    return df_pilot

def run_pilot(spend_cap_usd: float = 2.00, seed: int = 2026, max_workers: int = 5):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_pilot = sample_pilot_profiles(seed=seed, n_samples=200)
    print(f"Sampled {len(df_pilot)} dev profiles with seed={seed}.", flush=True)
    print(f"Threats: {(df_pilot['ground_truth_label'] == 1).sum()}, Genuine: {(df_pilot['ground_truth_label'] == 0).sum()}", flush=True)

    completed_keys = set()
    records = []
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    records.append(rec)
                    completed_keys.add((rec["system_id"], rec["profile_id"]))
        print(f"Loaded {len(records)} existing checkpoint records.", flush=True)

    llm_client = LLMClient(spend_cap_usd=spend_cap_usd)
    
    systems = {
        "S1": SingleAgentSystem("S1", llm_client=llm_client),
        "S1b": SingleAgentSystem("S1b", llm_client=llm_client),
        "S2": HierarchicalSystem(llm_client=llm_client),
        "S3": AdversarialDebateSystem(llm_client=llm_client)
    }

    # Build queue of pending tasks
    tasks = []
    for idx, row in df_pilot.iterrows():
        p_dict = row.to_dict()
        prof_id = p_dict["profile_id"]
        for sys_id in ["S1", "S1b", "S2", "S3"]:
            if (sys_id, prof_id) not in completed_keys:
                tasks.append((sys_id, p_dict))

    print(f"Total pending evaluations to execute: {len(tasks)} (Concurrency = {max_workers} workers)", flush=True)

    parse_failures_by_system = {"S1": 0, "S1b": 0, "S2": 0, "S3": 0}
    for r in records:
        if r.get("is_parse_failure"):
            parse_failures_by_system[r["system_id"]] = parse_failures_by_system.get(r["system_id"], 0) + 1

    def execute_eval(task_tuple):
        sys_id, profile_dict = task_tuple
        system = systems[sys_id]
        
        prof_id = profile_dict["profile_id"]
        gt = int(profile_dict["ground_truth_label"])
        is_nt = int(profile_dict["is_non_traditional"])
        violations = check_statutory_violations(profile_dict)
        has_violation = len(violations) > 0

        res = system.evaluate_profile(profile_dict, include_demographics=False)

        decision = res["decision"]
        is_parse_failure = res.get("is_parse_failure", False)
        
        is_genuine = (gt == 0)
        is_threat = (gt == 1)

        is_hard_rule_offload = (decision == "OFF-LOAD" and is_genuine and has_violation)
        is_discretionary_offload = (decision == "OFF-LOAD" and is_genuine and not has_violation)
        is_hard_rule_threat_leakage = (decision == "CLEARED" and is_threat and has_violation)

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "system_id": sys_id,
            "profile_id": prof_id,
            "ground_truth_label": gt,
            "is_non_traditional": is_nt,
            "decision": decision,
            "risk_score": res["risk_score"],
            "confidence": res["confidence"],
            "rationale": res["rationale"],
            "is_parse_failure": is_parse_failure,
            "parse_status": res["parse_status"],
            "retries": res["retries"],
            "calls_count": res.get("calls_count", 1),
            "rounds_completed": res.get("rounds_completed", 1 if sys_id == "S3" else None),
            "tokens": res["tokens"],
            "cost_usd": res["cost_usd"],
            "latency_ms": res["latency_ms"],
            "statutory_violations": violations,
            "has_statutory_violation": has_violation,
            "is_hard_rule_offload": is_hard_rule_offload,
            "is_discretionary_offload": is_discretionary_offload,
            "is_hard_rule_threat_leakage": is_hard_rule_threat_leakage,
            "specialist_outputs": res.get("specialist_outputs", None),
            "debate_transcript": res.get("debate_transcript", None),
            "raw_response": res.get("raw_response", "")
        }

        with checkpoint_lock:
            with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            records.append(entry)
            completed_keys.add((sys_id, prof_id))
            if is_parse_failure:
                parse_failures_by_system[sys_id] += 1
            
            cnt = len(records)
            if cnt % 10 == 0 or cnt == len(df_pilot) * 4:
                print(f"[{time.strftime('%H:%M:%S')}] Eval {cnt}/{len(df_pilot)*4} completed. Spend: ${llm_client.total_spend_usd:.4f}", flush=True)

        return entry

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(execute_eval, t): t for t in tasks}
            for future in as_completed(future_to_task):
                try:
                    entry = future.result()
                    # Check 2% gate (4 failures out of 200 per system)
                    sys_id = entry["system_id"]
                    if parse_failures_by_system[sys_id] > 4:
                        print(f"\n[GATE HALT] System {sys_id} exceeded 2% parse failure threshold ({parse_failures_by_system[sys_id]}/200 failures)!", flush=True)
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                except SpendCapExceededError as e:
                    print(f"\n[ALERT] SPEND CAP HALT: {e}", flush=True)
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                except Exception as e:
                    print(f"\n[ERROR] Worker error: {e}", flush=True)

    except Exception as e:
        print(f"\n[ERROR] ThreadPool execution encountered: {e}", flush=True)

    # Save manifest
    manifest = {
        "execution_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "dev_pilot_clean",
        "sample_size": len(df_pilot),
        "seed": seed,
        "spend_cap_usd": spend_cap_usd,
        "actual_spend_usd": llm_client.total_spend_usd,
        "total_calls": llm_client.total_calls,
        "total_prompt_tokens": llm_client.total_prompt_tokens,
        "total_completion_tokens": llm_client.total_completion_tokens,
        "total_reasoning_tokens": llm_client.total_reasoning_tokens,
        "model": "openai/gpt-oss-120b",
        "max_tokens": 4096,
        "reasoning_effort": "medium",
        "parse_failures_by_system": parse_failures_by_system
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return records, manifest

if __name__ == "__main__":
    records, manifest = run_pilot(spend_cap_usd=2.00, seed=2026, max_workers=5)
    print("Clean Pilot runner finished. Manifest written to", MANIFEST_PATH, flush=True)
