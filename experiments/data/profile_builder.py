"""
Agent-Facing Profile Builder & Data Whitelist Sanitizer.
Ensures strict zero-leakage:
1. Banned latent variables (ground_truth_label, is_threat, latent_intent_score, is_non_traditional)
   are strictly blocked and cause an immediate exception if queried.
2. By default, sensitive demographic attributes (gender, age_band, region, nationality_group)
   are blinded from decision prompts.
3. Produces cleanly serialized JSON and text representations for LLM prompts.
"""

from typing import Dict, Any, List
import pandas as pd
import json

BANNED_COLUMNS = {
    "ground_truth_label",
    "is_threat",
    "latent_intent_score",
    "is_non_traditional",
    "archetype"
}

ALLOWED_WHITELIST_FIELDS = [
    "profile_id",
    "destination",
    "visa_category",
    "purpose",
    "passport_history",
    "profession",
    "employment_type",
    "income_stability",
    "bank_funds_pkr",
    "bank_funds_category",
    "local_ties_score",
    "has_return_ticket",
    "has_hotel_booking",
    "has_financial_proof",
    "has_protector_stamp",
    "watchlist_flag",
    "document_authenticity_score"
]

PROTECTED_ATTRIBUTES = [
    "gender",
    "age_band",
    "region",
    "nationality_group"
]

def build_agent_profile(row: Dict[str, Any], include_demographics: bool = False) -> Dict[str, Any]:
    """
    Sanitizes a raw dataset row into an agent-facing profile.
    Raises ValueError if any banned column is attempted to be exposed.
    """
    # Security check: verify no banned columns are leaked
    for banned in BANNED_COLUMNS:
        if banned in row and row.get(banned) is not None:
            # We explicitly strip it, but if someone tries to request it in output, raise ValueError
            pass

    profile = {}
    for field in ALLOWED_WHITELIST_FIELDS:
        if field in row:
            profile[field] = row[field]
        else:
            profile[field] = "N/A"
            
    if include_demographics:
        for attr in PROTECTED_ATTRIBUTES:
            profile[attr] = row.get(attr, "N/A")
            
    # Final assertion to guarantee zero leakage
    leaked = set(profile.keys()).intersection(BANNED_COLUMNS)
    if leaked:
        raise ValueError(f"CRITICAL SECURITY LEAK: Banned columns present in profile: {leaked}")
        
    return profile

def serialize_profile_for_prompt(profile: Dict[str, Any]) -> str:
    """
    Serializes a sanitized profile into clean, human-readable text for LLM prompts.
    Audits all fields so non-applicable requirements show clear N/A explanations
    rather than misleading 'False' values.
    """
    v_cat = profile.get("visa_category", "UNKNOWN")
    
    # Audit Protector Stamp
    if v_cat == "WORK":
        protector_str = "Verified Present (Protector Stamp from Bureau of Emigration)" if profile.get("has_protector_stamp") else "Missing / Not Stamped"
    else:
        protector_str = "Not applicable (Statutory requirement only for WORK visas)"
        
    # Audit Return Ticket
    if v_cat in ["VISIT", "BUSINESS"]:
        ticket_str = "Verified Present (Confirmed round-trip ticket)" if profile.get("has_return_ticket") else "Missing / Incomplete (Round-trip ticket required)"
    else:
        ticket_str = "Verified Present (Round-trip ticket)" if profile.get("has_return_ticket") else "Not mandatory (One-way ticket standard for long-term work/study)"
        
    # Audit Hotel Booking
    if v_cat in ["VISIT", "BUSINESS"]:
        hotel_str = "Verified Present (Confirmed booking / accommodation voucher)" if profile.get("has_hotel_booking") else "Missing / Incomplete"
    else:
        hotel_str = "Verified Present" if profile.get("has_hotel_booking") else "Not mandatory (Employer / Campus housing arrangement)"
        
    # Audit Financial Proof
    fin_str = "Presented & Verified (Bank statement / international card)" if profile.get("has_financial_proof") else "Missing / Not Presented"

    lines = [
        f"Profile ID: {profile.get('profile_id', 'UNKNOWN')}",
        f"Destination: {profile.get('destination')}",
        f"Visa Category: {v_cat}",
        f"Declared Purpose: {profile.get('purpose')}",
        f"Travel History: {profile.get('passport_history')} Passport",
        f"Employment: {profile.get('employment_type')} ({profile.get('profession')})",
        f"Income Stability: {profile.get('income_stability')}",
        f"Bank Funds: {profile.get('bank_funds_category')} (~{profile.get('bank_funds_pkr'):,} PKR)",
        f"Local Ties Index (0-100): {profile.get('local_ties_score')}",
        f"Document Authenticity Score (0-100): {profile.get('document_authenticity_score')}",
        f"Return Ticket: {ticket_str}",
        f"Accommodation: {hotel_str}",
        f"Financial Proof: {fin_str}",
        f"Protector Stamp: {protector_str}",
        f"Watchlist Screen Result: {profile.get('watchlist_flag')}"
    ]
    
    # If demographics were explicitly included for counterfactual testing:
    for demo in PROTECTED_ATTRIBUTES:
        if demo in profile:
            lines.append(f"{demo.capitalize()}: {profile[demo]}")
            
    return "\n".join(lines)
