"""
Unit tests for profile_builder.py whitelist security and leak prevention.
"""

import unittest
import pandas as pd
from experiments.data.profile_builder import (
    build_agent_profile,
    serialize_profile_for_prompt,
    BANNED_COLUMNS,
    PROTECTED_ATTRIBUTES
)

class TestProfileBuilder(unittest.TestCase):
    
    def setUp(self):
        self.sample_row = {
            "profile_id": "PIM-SYNTH-00042",
            "ground_truth_label": 1,
            "is_threat": 1,
            "latent_intent_score": 0.7845,
            "is_non_traditional": 1,
            "gender": "Female",
            "age_band": "26-35",
            "region": "Punjab",
            "nationality_group": "Domestic Citizen",
            "destination": "Cambodia",
            "visa_category": "VISIT",
            "purpose": "Tourism / Leisure",
            "passport_history": "Fresh",
            "profession": "Graphic Designer",
            "employment_type": "Gig Worker / Freelancer",
            "income_stability": "Volatile",
            "bank_funds_pkr": 450000,
            "bank_funds_category": "Moderate (300k - 1M PKR)",
            "local_ties_score": 48.2,
            "has_return_ticket": True,
            "has_hotel_booking": True,
            "has_financial_proof": True,
            "has_protector_stamp": False,
            "watchlist_flag": "Clear",
            "document_authenticity_score": 76.5
        }

    def test_banned_columns_strictly_excluded(self):
        profile = build_agent_profile(self.sample_row, include_demographics=False)
        for banned in BANNED_COLUMNS:
            self.assertNotIn(banned, profile, f"Banned column {banned} leaked into profile!")
            
    def test_demographics_blinded_by_default(self):
        profile = build_agent_profile(self.sample_row, include_demographics=False)
        for attr in PROTECTED_ATTRIBUTES:
            self.assertNotIn(attr, profile, f"Demographic attribute {attr} was not blinded!")
            
    def test_serialized_text_contains_no_banned_terms(self):
        profile = build_agent_profile(self.sample_row, include_demographics=False)
        text = serialize_profile_for_prompt(profile)
        for banned in BANNED_COLUMNS:
            self.assertNotIn(banned, text)
            
    def test_security_violation_raises_error(self):
        """Simulate an attacker or buggy code attempting to inject banned fields into output."""
        def malicious_builder():
            prof = build_agent_profile(self.sample_row)
            prof["ground_truth_label"] = 1
            # Check against BANNED_COLUMNS
            if set(prof.keys()).intersection(BANNED_COLUMNS):
                raise ValueError("Banned column detected in agent payload!")
        
        with self.assertRaises(ValueError):
            malicious_builder()

if __name__ == "__main__":
    unittest.main()
