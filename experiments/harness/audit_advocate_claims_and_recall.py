"""
Run Advocate Claims Audit, 50-Injected False Claims Recall Benchmark,
and export 50 real Advocate claims to CSV for manual human labeling.
"""
import os
import sys
import json
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.harness.claim_auditor import audit_advocate_claim, run_advocate_claims_audit

DATA_DEV_PATH = "experiments/data/dataset_dev.csv"
CHECKPOINT_PATH = "experiments/results/pilot_raw_checkpoint.jsonl"
OUT_CSV_PATH = "experiments/results/advocate_claims_sample_50.csv"
OUT_RECALL_JSON = "experiments/results/claim_auditor_recall_test.json"

def run_injected_recall_test():
    """Generates 50 known false/contradictory claims and evaluates auditor recall."""
    test_cases = []
    
    # Category 1: False return ticket (10 cases)
    for i in range(10):
        test_cases.append({
            "type": "false_return_ticket",
            "claim": f"Passenger presents a valid return ticket confirming return on flight PK-{700+i}.",
            "profile": {"has_return_ticket": False, "visa_category": "VISIT", "bank_funds_pkr": 500000}
        })
        
    # Category 2: False hotel booking (10 cases)
    for i in range(10):
        test_cases.append({
            "type": "false_hotel_booking",
            "claim": f"Passenger has a confirmed hotel booking at Grand Palace Hotel for {i+3} days.",
            "profile": {"has_hotel_booking": False, "visa_category": "VISIT", "bank_funds_pkr": 500000}
        })
        
    # Category 3: False financial proof (10 cases)
    for i in range(10):
        test_cases.append({
            "type": "false_financial_proof",
            "claim": "Passenger has presented adequate financial proof and verified bank statements.",
            "profile": {"has_financial_proof": False, "bank_funds_pkr": 50000}
        })
        
    # Category 4: False protector stamp (10 cases)
    for i in range(10):
        test_cases.append({
            "type": "false_protector_stamp",
            "claim": "Passenger holds a valid protector stamp properly endorsed on passport.",
            "profile": {"has_protector_stamp": False, "visa_category": "WORK"}
        })
        
    # Category 5: Wealth exaggeration (5 cases)
    for i in range(5):
        test_cases.append({
            "type": "wealth_exaggeration",
            "claim": "Passenger possesses substantial wealth and multi-million personal assets in Pakistan.",
            "profile": {"bank_funds_pkr": 120000 + i*10000}
        })
        
    # Category 6: False government official (5 cases)
    for i in range(5):
        test_cases.append({
            "type": "false_employment_official",
            "claim": "Passenger is a permanent civil servant traveling for government training.",
            "profile": {"employment_type": "Informal/Gig"}
        })

    detected_count = 0
    results_detail = []
    for tc in test_cases:
        flags = audit_advocate_claim(tc["claim"], tc["profile"])
        is_detected = (len(flags) > 0)
        if is_detected:
            detected_count += 1
        results_detail.append({
            "type": tc["type"],
            "claim": tc["claim"],
            "detected": is_detected,
            "flags": flags
        })
        
    recall = detected_count / len(test_cases)
    summary = {
        "total_injected_false_claims": len(test_cases),
        "detected_false_claims": detected_count,
        "auditor_recall": recall,
        "recall_percentage": recall * 100.0,
        "details": results_detail
    }
    with open(OUT_RECALL_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    return summary

def export_50_real_claims_to_csv():
    df_dev = pd.read_csv(DATA_DEV_PATH).set_index("profile_id").to_dict(orient="index")
    records = []
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    real_claims = []
    for r in records:
        if r.get("system_id") != "S3":
            continue
        pid = r.get("profile_id")
        prof = df_dev.get(pid, {})
        transcript = r.get("debate_transcript") or []
        for turn in transcript:
            if turn.get("speaker") == "ADVOCATE":
                args = turn.get("content", {}).get("mitigating_arguments", [])
                for arg in args:
                    flags = audit_advocate_claim(str(arg), prof)
                    real_claims.append({
                        "profile_id": pid,
                        "claim_text": str(arg),
                        "visa_category": prof.get("visa_category"),
                        "bank_funds_pkr": prof.get("bank_funds_pkr"),
                        "employment_type": prof.get("employment_type"),
                        "has_return_ticket": prof.get("has_return_ticket"),
                        "has_hotel_booking": prof.get("has_hotel_booking"),
                        "has_financial_proof": prof.get("has_financial_proof"),
                        "has_protector_stamp": prof.get("has_protector_stamp"),
                        "passport_history": prof.get("passport_history"),
                        "auditor_verdict": "FLAGGED: " + "; ".join(flags) if flags else "GROUNDED",
                        "human_label": "" # To be filled by user
                    })

    # Take first 50 unique claims
    sample_50 = real_claims[:50]
    df_sample = pd.DataFrame(sample_50)
    df_sample.insert(0, "claim_id", range(1, len(df_sample) + 1))
    df_sample.to_csv(OUT_CSV_PATH, index=False)
    print(f"Exported {len(df_sample)} real Advocate claims to {OUT_CSV_PATH}")
    return df_sample

if __name__ == "__main__":
    print("=== RUNNING INJECTED FALSE CLAIMS RECALL TEST (N = 50) ===")
    rec_res = run_injected_recall_test()
    print(f"Detected: {rec_res['detected_false_claims']}/{rec_res['total_injected_false_claims']} false claims.")
    print(f"Auditor Recall: {rec_res['recall_percentage']:.2f}%\n")
    
    print("=== EXPORTING 50 REAL ADVOCATE CLAIMS FOR MANUAL LABELING ===")
    export_50_real_claims_to_csv()

