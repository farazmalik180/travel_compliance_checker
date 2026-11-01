"""
Deterministic Factual Claim Auditor for Traveler Advocate Claims.
Audits arguments directly against passenger profile ground-truth attributes
WITHOUT using any LLM calls (100% deterministic rule and regex logic).
"""
import re
from typing import Dict, Any, List, Tuple

def audit_advocate_claim(claim_text: str, profile: Dict[str, Any]) -> List[str]:
    """
    Checks if an Advocate claim contradicts or invents facts outside profile fields.
    Returns list of violation descriptions (empty if claim is grounded).
    """
    c = claim_text.lower()
    flags = []

    # 1. Statutory Document Contradictions
    is_exempt = any(ex in c for ex in ["not required", "exempt", "not applicable", "n/a", "no return ticket required", "no protector required", "no hotel required"])

    # Financial proof
    if any(k in c for k in ["verified financial proof", "presented financial proof", "adequate financial proof", "proof of funds"]):
        if not profile.get("has_financial_proof", False) and not is_exempt:
            flags.append("CONTRADICTION: Claims verified financial proof, but has_financial_proof is False")

    # Return ticket
    if any(k in c for k in ["valid return ticket", "presents return ticket", "confirmed return ticket", "round-trip ticket"]):
        if not profile.get("has_return_ticket", False) and not is_exempt:
            flags.append("CONTRADICTION: Claims valid return ticket, but has_return_ticket is False")

    # Hotel booking
    if any(k in c for k in ["confirmed hotel", "hotel booking", "confirmed accommodation", "verified lodging"]):
        if not profile.get("has_hotel_booking", False) and not is_exempt:
            flags.append("CONTRADICTION: Claims confirmed hotel booking, but has_hotel_booking is False")

    # Protector stamp
    if any(k in c for k in ["holds protector", "has protector stamp", "protector stamp verified", "valid protector stamp"]):
        if not profile.get("has_protector_stamp", False) and not is_exempt:
            flags.append("CONTRADICTION: Claims Protector stamp, but has_protector_stamp is False")

    # 2. Financial Amount & Wealth Exaggeration
    funds = profile.get("bank_funds_pkr", 0)
    if any(k in c for k in ["substantial wealth", "multi-million", "wealthy", "high net worth"]):
        if funds < 1_000_000:
            flags.append(f"EXAGGERATION: Claims substantial wealth, but bank_funds_pkr is only {funds:,}")

    # Check for specific numbers mentioned in claim that exceed actual funds by > 50%
    num_matches = re.findall(r"(?:pkr|rs\.?)\s*([\d,]+)", c)
    for nm in num_matches:
        try:
            val = int(nm.replace(",", ""))
            if val > funds * 1.5 and val > 100_000:
                flags.append(f"UNGROUNDED_NUMBER: Claims {val:,} PKR, exceeding actual bank funds {funds:,} PKR")
        except ValueError:
            pass

    # 3. Employment & Profession Contradictions
    emp = str(profile.get("employment_type", ""))
    if any(k in c for k in ["civil servant", "government official", "government officer", "public sector officer"]):
        if emp != "Civil Servant":
            flags.append(f"CONTRADICTION: Claims Civil Servant, but employment_type is '{emp}'")

    if any(k in c for k in ["tenured", "formal corporate employment", "permanent corporate position"]):
        if emp in ["Unemployed", "Informal/Gig"]:
            flags.append(f"CONTRADICTION: Claims formal corporate employment, but employment_type is '{emp}'")

    # 4. Travel History Contradictions
    pass_hist = str(profile.get("passport_history", ""))
    if any(k in c for k in ["extensive international travel", "frequent international traveler", "prior overseas travel record"]):
        if pass_hist == "Fresh":
            flags.append("CONTRADICTION: Claims extensive prior overseas travel, but passport_history is 'Fresh'")

    # 5. Local Ties Exaggeration
    ties = profile.get("local_ties_score", 50.0)
    if any(k in c for k in ["deep rooted ties", "exceptionally strong community ties", "unshakeable local ties"]):
        if ties < 30.0:
            flags.append(f"EXAGGERATION: Claims exceptionally strong ties, but local_ties_score is {ties:.1f}")

    return flags

def run_advocate_claims_audit(records: List[Dict[str, Any]], dev_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Audits all advocate claims across records in a deterministic, LLM-free manner."""
    total_claims = 0
    flagged_records = []
    
    for r in records:
        if r.get("system_id") != "S3":
            continue
        prof_id = r.get("profile_id")
        prof = dev_map.get(prof_id, {})
        transcript = r.get("debate_transcript") or []
        for turn in transcript:
            if turn.get("speaker") == "ADVOCATE":
                args = turn.get("content", {}).get("mitigating_arguments", [])
                for arg in args:
                    total_claims += 1
                    flags = audit_advocate_claim(str(arg), prof)
                    if flags:
                        flagged_records.append({
                            "profile_id": prof_id,
                            "claim": str(arg),
                            "flags": flags,
                            "profile_summary": {
                                "bank_funds_pkr": prof.get("bank_funds_pkr"),
                                "employment_type": prof.get("employment_type"),
                                "has_financial_proof": prof.get("has_financial_proof"),
                                "has_return_ticket": prof.get("has_return_ticket"),
                                "has_hotel_booking": prof.get("has_hotel_booking"),
                                "has_protector_stamp": prof.get("has_protector_stamp"),
                                "passport_history": prof.get("passport_history")
                            }
                        })
                        
    return {
        "total_claims_evaluated": total_claims,
        "flagged_claims_count": len(flagged_records),
        "unsubstantiated_rate": (len(flagged_records) / total_claims) if total_claims > 0 else 0.0,
        "flagged_claims": flagged_records
    }
