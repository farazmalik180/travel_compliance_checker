"""
Updated Synthetic Dataset Generator for MS AI Thesis.
Addresses all user requirements:
1. Coherent profession, visa_category, purpose, and employment_type mapping.
2. Smooth Z driving observable features via link functions.
3. Strict independence between is_non_traditional and ground_truth_label.
4. Exact stratification assertions (disjoint IDs, union == full, stratified 80/20 cell counts match exactly).
5. Generation of both the standard benchmark (Full, Dev, Test) and a separate balanced fairness set.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

def sample_coherent_travel_features(emp_type: str, prof: str, z: float, is_threat: int, age: str):
    """
    Ensures visa_category and purpose are semantically coherent with employment and profession,
    while realistically modeling threat concealment strategies.
    """
    high_scrutiny_dests = [
        "Cambodia", "UAE", "Thailand", "Malaysia", "Iraq", 
        "Kenya", "Kyrgyzstan", "Qatar", "Senegal", "Libya"
    ]
    standard_dests = [
        "United Kingdom", "United States", "Germany", "Turkey", 
        "Saudi Arabia", "Canada", "Australia", "China", "Japan", "Oman"
    ]
    
    # Destination probability driven by Z
    p_high_scrutiny = expit(-1.2 + 2.4 * z)
    dest = np.random.choice(high_scrutiny_dests) if np.random.rand() < p_high_scrutiny else np.random.choice(standard_dests)
    
    # Coherent Visa Category & Purpose Distribution
    if emp_type == "Registered Business Owner":
        # Business owners travel for business sourcing or family tourism; rarely work permits
        if is_threat == 1 and np.random.rand() < 0.35:
            # Threat posing as business owner but on dubious visit visa
            v_cat = "VISIT"
            purp = "Tourism / Leisure"
        else:
            v_probs = [0.55, 0.40, 0.05]  # Business, Visit, Work (Executive)
            v_cat = np.random.choice(["BUSINESS", "VISIT", "WORK"], p=v_probs)
            if v_cat == "BUSINESS":
                purp = np.random.choice(["Trade Fair / B2B Networking", "Vendor Contract Negotiation", "Machinery Sourcing"])
            elif v_cat == "VISIT":
                purp = np.random.choice(["Tourism / Leisure", "Visiting Family / Relatives"])
            else:
                purp = "Overseas Corporate Expansion / Executive Management"
                
    elif emp_type == "Corporate Salaried":
        if is_threat == 1 and np.random.rand() < 0.45:
            v_cat = "VISIT"
            purp = "Visiting Family / Relatives"
        else:
            v_probs = [0.45, 0.35, 0.15, 0.05]  # Work, Visit, Business, Student
            v_cat = np.random.choice(["WORK", "VISIT", "BUSINESS", "STUDENT"], p=v_probs)
            if v_cat == "WORK":
                purp = np.random.choice(["Overseas Corporate Employment", "Technical Specialist Assignment"])
            elif v_cat == "VISIT":
                purp = np.random.choice(["Tourism / Leisure", "Visiting Family / Relatives"])
            elif v_cat == "BUSINESS":
                purp = np.random.choice(["Corporate Conference / Workshop", "Client Consultation"])
            else:
                purp = "Postgraduate Professional Degree"
                
    elif emp_type == "Civil Servant":
        # Civil servants travel on official leave for visit/conference or study
        v_probs = [0.70, 0.20, 0.10]  # Visit, Business/Official, Student
        v_cat = np.random.choice(["VISIT", "BUSINESS", "STUDENT"], p=v_probs)
        if v_cat == "VISIT":
            purp = np.random.choice(["Visiting Family / Relatives", "Tourism / Leisure"])
        elif v_cat == "BUSINESS":
            purp = "Official Training Delegation / Workshop"
        else:
            purp = "Higher Education Study Leave"
            
    elif emp_type == "Student":
        v_probs = [0.65, 0.35]  # Student, Visit
        v_cat = np.random.choice(["STUDENT", "VISIT"], p=v_probs)
        if v_cat == "STUDENT":
            purp = "Undergraduate Studies" if age == "18-25" else "Postgraduate Studies"
        else:
            purp = np.random.choice(["Tourism / Leisure", "Visiting Family / Relatives"])
            
    elif emp_type == "Gig Worker / Freelancer":
        if prof in ["Ride-Hailing Driver", "Delivery Courier"]:
            # Transport workers traveling for overseas contracts or tourism
            if is_threat == 1 and np.random.rand() < 0.60:
                v_cat = "VISIT"  # Attempting to circumvent protector stamp by using visit visa
                purp = "Tourism / Leisure"
            else:
                v_cat = np.random.choice(["WORK", "VISIT"], p=[0.70, 0.30])
                purp = "Commercial Transport Contract" if v_cat == "WORK" else "Visiting Family"
        else:
            # Digital freelancers (Software Developer, Graphic Designer, Online Marketer)
            v_probs = [0.60, 0.25, 0.15]  # Visit, Business, Work
            v_cat = np.random.choice(["VISIT", "BUSINESS", "WORK"], p=v_probs)
            if v_cat == "VISIT":
                purp = np.random.choice(["Tourism / Leisure", "Visiting Family / Relatives"])
            elif v_cat == "BUSINESS":
                purp = "Tech Conference / Web Summit"
            else:
                purp = "Contract Software Engineer / Remote Specialist"
                
    elif emp_type == "Informal Self-Employed":
        # Artisans, shopkeepers, street vendors
        if is_threat == 1 and np.random.rand() < 0.65:
            v_cat = "VISIT"
            purp = "Tourism / Leisure"
        else:
            v_probs = [0.55, 0.40, 0.05]  # Visit, Work (Skilled/Semi-skilled), Business
            v_cat = np.random.choice(["VISIT", "WORK", "BUSINESS"], p=v_probs)
            if v_cat == "VISIT":
                purp = np.random.choice(["Tourism / Leisure", "Visiting Family / Relatives"])
            elif v_cat == "WORK":
                purp = "Skilled Craftsman / Overseas Construction Worker"
            else:
                purp = "B2B Handicraft & Commodity Trade Fair"
    else:
        # Contract Worker
        v_cat = np.random.choice(["WORK", "VISIT"], p=[0.75, 0.25])
        purp = "Technical Services Contract" if v_cat == "WORK" else "Visiting Family"

    return dest, v_cat, purp

def generate_profiles(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # 1. SENSITIVE & DEMOGRAPHIC ATTRIBUTES
    genders = np.random.choice(["Male", "Female", "Non-binary"], size=n_samples, p=[0.62, 0.36, 0.02])
    age_bands = np.random.choice(["18-25", "26-35", "36-50", "51+"], size=n_samples, p=[0.25, 0.42, 0.23, 0.10])
    regions = np.random.choice(
        ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan", "Islamabad/AJK/GB"],
        size=n_samples,
        p=[0.52, 0.22, 0.15, 0.06, 0.05]
    )
    nationality_groups = np.random.choice(
        ["Domestic Citizen", "Dual National", "Resident Expatriate"],
        size=n_samples,
        p=[0.90, 0.07, 0.03]
    )
    
    # 2. NON-TRADITIONAL PROFILE ARCHETYPE (Strictly independent of ground truth)
    p_non_trad_by_age = {
        "18-25": 0.55,
        "26-35": 0.45,
        "36-50": 0.30,
        "51+": 0.15
    }
    is_non_traditional = np.array([
        np.random.binomial(1, p_non_trad_by_age[age]) for age in age_bands
    ], dtype=int)
    
    # 3. LATENT GROUND TRUTH DRAW
    # Fix exact 20% threat prevalence: exactly 1,000 threats and 4,000 genuine
    n_threats = int(0.20 * n_samples)
    ground_truth = np.zeros(n_samples, dtype=int)
    threat_indices = np.random.choice(n_samples, size=n_threats, replace=False)
    ground_truth[threat_indices] = 1
    
    # Continuous latent risk propensity Z in [0, 1]
    z_scores = np.zeros(n_samples)
    for i in range(n_samples):
        if ground_truth[i] == 0:
            z_scores[i] = np.random.beta(2.0, 4.5)
        else:
            z_scores[i] = np.random.beta(4.2, 2.8)
            
    # 4. OBSERVABLE FEATURES DRIVEN SMOOTHLY BY Z AND ARCHETYPE
    professions = []
    employment_types = []
    income_stabilities = []
    bank_funds_categories = []
    bank_funds_pkr = []
    local_ties_scores = []
    passport_histories = []
    destinations = []
    visa_categories = []
    purposes = []
    has_return_tickets = []
    has_hotel_bookings = []
    has_financial_proofs = []
    has_protector_stamps = []
    watchlist_flags = []
    document_authenticity_scores = []
    
    for i in range(n_samples):
        z = z_scores[i]
        gt = ground_truth[i]
        nt = is_non_traditional[i]
        age = age_bands[i]
        
        # A. Employment & Profession
        if nt == 1:
            emp_type = np.random.choice(["Gig Worker / Freelancer", "Informal Self-Employed", "Contract Worker"], p=[0.55, 0.30, 0.15])
            if emp_type == "Gig Worker / Freelancer":
                prof = np.random.choice(["Software Developer", "Graphic Designer", "Content Creator", "Ride-Hailing Driver", "Online Marketer"])
            elif emp_type == "Informal Self-Employed":
                prof = np.random.choice(["Small Shop Owner", "Artisan / Craftsman", "Independent Broker", "Street Vendor / Stallholder"])
            else:
                prof = np.random.choice(["Short-Term Tech Contractor", "Event Coordinator", "Seasonal Construction Assistant"])
            inc_stab = np.random.choice(["Volatile", "Moderate", "Low"], p=[0.50, 0.35, 0.15])
        else:
            if age == "18-25" and np.random.rand() < 0.35:
                emp_type = "Student"
                prof = "Undergraduate Student"
                inc_stab = "Dependent / Family Supported"
            else:
                emp_type = np.random.choice(["Corporate Salaried", "Civil Servant", "Registered Business Owner"], p=[0.60, 0.25, 0.15])
                if emp_type == "Corporate Salaried":
                    prof = np.random.choice(["Accountant", "Civil Engineer", "Bank Officer", "HR Manager", "Medical Doctor"])
                elif emp_type == "Civil Servant":
                    prof = np.random.choice(["School Teacher", "Revenue Officer", "Customs Inspector", "Clerical Officer"])
                else:
                    prof = np.random.choice(["Textile Manufacturer", "Import-Export Trader", "Retail Pharmacy Chain Owner"])
                inc_stab = np.random.choice(["High", "Moderate"], p=[0.70, 0.30])
                
        employment_types.append(emp_type)
        professions.append(prof)
        income_stabilities.append(inc_stab)
        
        # B. Coherent Destination, Visa Category & Purpose
        dest, v_cat, purp = sample_coherent_travel_features(emp_type, prof, z, gt, age)
        destinations.append(dest)
        visa_categories.append(v_cat)
        purposes.append(purp)
        
        # C. Local Ties Score (Smooth continuous function of Z and non-traditional status)
        raw_ties = 82.0 - (45.0 * z) - (12.0 * nt) + np.random.normal(0, 10.0)
        local_ties_scores.append(round(float(np.clip(raw_ties, 5.0, 98.0)), 1))
        
        # D. Bank Funds (PKR) (Log-normal driven by Z and non-traditional status)
        log_mean = 14.2 - (1.6 * z) - (0.6 * nt)
        funds_val = np.random.lognormal(mean=log_mean, sigma=0.65)
        if np.random.rand() < (0.35 * z):
            funds_val += np.random.uniform(800000, 1500000)  # Borrowed show money
        funds_val = int(np.clip(funds_val, 50000, 15000000))
        bank_funds_pkr.append(funds_val)
        
        if funds_val < 300000:
            b_cat = "Low (< 300k PKR)"
        elif funds_val <= 1000000:
            b_cat = "Moderate (300k - 1M PKR)"
        elif funds_val <= 3000000:
            b_cat = "High (1M - 3M PKR)"
        else:
            b_cat = "Very High (> 3M PKR)"
        bank_funds_categories.append(b_cat)
        
        # E. Passport History - Smooth logistic function of Z, age, and non-trad
        p_fresh_logit = -0.8 + (1.6 * z) + (0.7 if age == "18-25" else 0.0) + (0.3 * nt)
        passport_histories.append("Fresh" if np.random.rand() < expit(p_fresh_logit) else "Experienced")
        
        # F. Return Ticket & Hotel Booking - Smooth logistic functions of Z and visa category
        if v_cat in ["VISIT", "BUSINESS"]:
            p_ret_logit = 3.2 - (2.6 * z)
            p_hot_logit = 2.9 - (2.4 * z)
        else:
            p_ret_logit = -1.2 - (0.6 * z)
            p_hot_logit = 0.8 - (1.2 * z)
        has_return_tickets.append(bool(np.random.rand() < expit(p_ret_logit)))
        has_hotel_bookings.append(bool(np.random.rand() < expit(p_hot_logit)))
        
        # G. Financial Proof Presentation
        p_fp_logit = 2.6 - (2.2 * z)
        has_financial_proofs.append(bool(np.random.rand() < expit(p_fp_logit)))
        
        # H. Protector Stamp (Bureau of Emigration) - Applicable if WORK visa
        if v_cat == "WORK":
            p_prot_logit = 2.4 - (4.2 * z)
            has_protector_stamps.append(bool(np.random.rand() < expit(p_prot_logit)))
        else:
            has_protector_stamps.append(False)
            
        # I. Watchlist Flag - Smooth function of Z
        p_name_match = 0.05 + (0.12 * z)
        p_verified_alert = max(0.0, (z - 0.70) * 0.40)
        p_clear = max(0.0, 1.0 - (p_name_match + p_verified_alert))
        wl_probs = np.array([p_clear, p_name_match, p_verified_alert])
        wl_probs = wl_probs / wl_probs.sum()
        watchlist_flags.append(np.random.choice(["Clear", "Potential Name Match", "Verified Watchlist Alert"], p=wl_probs))
        
        # J. Document Authenticity Score - Smooth continuous function of Z
        raw_doc = 94.0 - (32.0 * z) + np.random.normal(0, 8.0)
        document_authenticity_scores.append(round(float(np.clip(raw_doc, 15.0, 99.0)), 1))

    df = pd.DataFrame({
        "profile_id": [f"PIM-SYNTH-{i+1:05d}" for i in range(n_samples)],
        "latent_intent_score": np.round(z_scores, 4),
        "ground_truth_label": ground_truth,
        "is_threat": ground_truth,
        
        # Sensitive Attributes for Fairness Analysis
        "gender": genders,
        "age_band": age_bands,
        "region": regions,
        "nationality_group": nationality_groups,
        "is_non_traditional": is_non_traditional,
        
        # Observable Profile Features
        "employment_type": employment_types,
        "profession": professions,
        "income_stability": income_stabilities,
        "bank_funds_pkr": bank_funds_pkr,
        "bank_funds_category": bank_funds_categories,
        "local_ties_score": local_ties_scores,
        "passport_history": passport_histories,
        "destination": destinations,
        "visa_category": visa_categories,
        "purpose": purposes,
        "has_return_ticket": has_return_tickets,
        "has_hotel_booking": has_hotel_bookings,
        "has_financial_proof": has_financial_proofs,
        "has_protector_stamp": has_protector_stamps,
        "watchlist_flag": watchlist_flags,
        "document_authenticity_score": document_authenticity_scores
    })
    
    return df

def generate_balanced_fairness_set(n_samples: int = 1500, seed: int = 2026) -> pd.DataFrame:
    """
    Generates a dedicated fairness evaluation dataset with balanced sensitive groups
    ensuring n >= 100 genuine and n >= 30 threats per subgroup where possible.
    """
    np.random.seed(seed)
    
    # Balanced groups
    genders = ["Male", "Female", "Non-binary"]
    age_bands = ["18-25", "26-35", "36-50", "51+"]
    regions = ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan", "Islamabad/AJK/GB"]
    nationalities = ["Domestic Citizen", "Dual National", "Resident Expatriate"]
    
    # Balanced assignment
    g_arr = np.random.choice(genders, size=n_samples)
    a_arr = np.random.choice(age_bands, size=n_samples)
    r_arr = np.random.choice(regions, size=n_samples)
    n_arr = np.random.choice(nationalities, size=n_samples)
    nt_arr = np.random.choice([0, 1], size=n_samples)
    
    # 20% Threat rate
    n_threats = int(0.20 * n_samples)
    ground_truth = np.zeros(n_samples, dtype=int)
    threat_indices = np.random.choice(n_samples, size=n_threats, replace=False)
    ground_truth[threat_indices] = 1
    
    z_scores = np.zeros(n_samples)
    for i in range(n_samples):
        if ground_truth[i] == 0:
            z_scores[i] = np.random.beta(2.0, 4.5)
        else:
            z_scores[i] = np.random.beta(4.2, 2.8)
            
    # Sample observables using coherent logic
    professions = []
    employment_types = []
    income_stabilities = []
    bank_funds_categories = []
    bank_funds_pkr = []
    local_ties_scores = []
    passport_histories = []
    destinations = []
    visa_categories = []
    purposes = []
    has_return_tickets = []
    has_hotel_bookings = []
    has_financial_proofs = []
    has_protector_stamps = []
    watchlist_flags = []
    document_authenticity_scores = []
    
    for i in range(n_samples):
        z = z_scores[i]
        gt = ground_truth[i]
        nt = nt_arr[i]
        age = a_arr[i]
        
        if nt == 1:
            emp_type = np.random.choice(["Gig Worker / Freelancer", "Informal Self-Employed", "Contract Worker"], p=[0.55, 0.30, 0.15])
            if emp_type == "Gig Worker / Freelancer":
                prof = np.random.choice(["Software Developer", "Graphic Designer", "Content Creator", "Ride-Hailing Driver", "Online Marketer"])
            elif emp_type == "Informal Self-Employed":
                prof = np.random.choice(["Small Shop Owner", "Artisan / Craftsman", "Independent Broker", "Street Vendor / Stallholder"])
            else:
                prof = np.random.choice(["Short-Term Tech Contractor", "Event Coordinator", "Seasonal Construction Assistant"])
            inc_stab = np.random.choice(["Volatile", "Moderate", "Low"], p=[0.50, 0.35, 0.15])
        else:
            if age == "18-25" and np.random.rand() < 0.35:
                emp_type = "Student"
                prof = "Undergraduate Student"
                inc_stab = "Dependent / Family Supported"
            else:
                emp_type = np.random.choice(["Corporate Salaried", "Civil Servant", "Registered Business Owner"], p=[0.60, 0.25, 0.15])
                if emp_type == "Corporate Salaried":
                    prof = np.random.choice(["Accountant", "Civil Engineer", "Bank Officer", "HR Manager", "Medical Doctor"])
                elif emp_type == "Civil Servant":
                    prof = np.random.choice(["School Teacher", "Revenue Officer", "Customs Inspector", "Clerical Officer"])
                else:
                    prof = np.random.choice(["Textile Manufacturer", "Import-Export Trader", "Retail Pharmacy Chain Owner"])
                inc_stab = np.random.choice(["High", "Moderate"], p=[0.70, 0.30])
                
        employment_types.append(emp_type)
        professions.append(prof)
        income_stabilities.append(inc_stab)
        
        dest, v_cat, purp = sample_coherent_travel_features(emp_type, prof, z, gt, age)
        destinations.append(dest)
        visa_categories.append(v_cat)
        purposes.append(purp)
        
        raw_ties = 82.0 - (45.0 * z) - (12.0 * nt) + np.random.normal(0, 10.0)
        local_ties_scores.append(round(float(np.clip(raw_ties, 5.0, 98.0)), 1))
        
        log_mean = 14.2 - (1.6 * z) - (0.6 * nt)
        funds_val = np.random.lognormal(mean=log_mean, sigma=0.65)
        if np.random.rand() < (0.35 * z):
            funds_val += np.random.uniform(800000, 1500000)
        funds_val = int(np.clip(funds_val, 50000, 15000000))
        bank_funds_pkr.append(funds_val)
        
        if funds_val < 300000:
            b_cat = "Low (< 300k PKR)"
        elif funds_val <= 1000000:
            b_cat = "Moderate (300k - 1M PKR)"
        elif funds_val <= 3000000:
            b_cat = "High (1M - 3M PKR)"
        else:
            b_cat = "Very High (> 3M PKR)"
        bank_funds_categories.append(b_cat)
        
        p_fresh_logit = -0.8 + (1.6 * z) + (0.7 if age == "18-25" else 0.0) + (0.3 * nt)
        passport_histories.append("Fresh" if np.random.rand() < expit(p_fresh_logit) else "Experienced")
        
        if v_cat in ["VISIT", "BUSINESS"]:
            p_ret_logit = 3.2 - (2.6 * z)
            p_hot_logit = 2.9 - (2.4 * z)
        else:
            p_ret_logit = -1.2 - (0.6 * z)
            p_hot_logit = 0.8 - (1.2 * z)
        has_return_tickets.append(bool(np.random.rand() < expit(p_ret_logit)))
        has_hotel_bookings.append(bool(np.random.rand() < expit(p_hot_logit)))
        
        p_fp_logit = 2.6 - (2.2 * z)
        has_financial_proofs.append(bool(np.random.rand() < expit(p_fp_logit)))
        
        if v_cat == "WORK":
            p_prot_logit = 2.4 - (4.2 * z)
            has_protector_stamps.append(bool(np.random.rand() < expit(p_prot_logit)))
        else:
            has_protector_stamps.append(False)
            
        p_name_match = 0.05 + (0.12 * z)
        p_verified_alert = max(0.0, (z - 0.70) * 0.40)
        p_clear = max(0.0, 1.0 - (p_name_match + p_verified_alert))
        wl_probs = np.array([p_clear, p_name_match, p_verified_alert])
        wl_probs = wl_probs / wl_probs.sum()
        watchlist_flags.append(np.random.choice(["Clear", "Potential Name Match", "Verified Watchlist Alert"], p=wl_probs))
        
        raw_doc = 94.0 - (32.0 * z) + np.random.normal(0, 8.0)
        document_authenticity_scores.append(round(float(np.clip(raw_doc, 15.0, 99.0)), 1))

    df_fair = pd.DataFrame({
        "profile_id": [f"PIM-FAIR-{i+1:05d}" for i in range(n_samples)],
        "latent_intent_score": np.round(z_scores, 4),
        "ground_truth_label": ground_truth,
        "is_threat": ground_truth,
        "gender": g_arr,
        "age_band": a_arr,
        "region": r_arr,
        "nationality_group": n_arr,
        "is_non_traditional": nt_arr,
        "employment_type": employment_types,
        "profession": professions,
        "income_stability": income_stabilities,
        "bank_funds_pkr": bank_funds_pkr,
        "bank_funds_category": bank_funds_categories,
        "local_ties_score": local_ties_scores,
        "passport_history": passport_histories,
        "destination": destinations,
        "visa_category": visa_categories,
        "purpose": purposes,
        "has_return_ticket": has_return_tickets,
        "has_hotel_booking": has_hotel_bookings,
        "has_financial_proof": has_financial_proofs,
        "has_protector_stamp": has_protector_stamps,
        "watchlist_flag": watchlist_flags,
        "document_authenticity_score": document_authenticity_scores
    })
    return df_fair

def save_and_split_dataset(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    strat_key = df["ground_truth_label"].astype(str) + "_" + df["is_non_traditional"].astype(str)
    
    dev_df, test_df = train_test_split(
        df,
        test_size=0.80,
        random_state=42,
        stratify=strat_key
    )
    
    dev_df = dev_df.sort_values("profile_id").reset_index(drop=True)
    test_df = test_df.sort_values("profile_id").reset_index(drop=True)
    
    # Strict Mathematical Assertions
    # 1. Disjoint IDs
    dev_ids = set(dev_df["profile_id"])
    test_ids = set(test_df["profile_id"])
    full_ids = set(df["profile_id"])
    assert len(dev_ids.intersection(test_ids)) == 0, "ERROR: Dev and Test IDs are not disjoint!"
    
    # 2. Union equals Full
    assert dev_ids.union(test_ids) == full_ids, "ERROR: Union of Dev and Test does not equal Full!"
    
    # 3. Stratified cell counts match exactly (80/20 +- 1)
    for key in strat_key.unique():
        full_count = (strat_key == key).sum()
        dev_count = (dev_df["ground_truth_label"].astype(str) + "_" + dev_df["is_non_traditional"].astype(str) == key).sum()
        test_count = (test_df["ground_truth_label"].astype(str) + "_" + test_df["is_non_traditional"].astype(str) == key).sum()
        
        expected_dev = round(0.20 * full_count)
        expected_test = full_count - expected_dev
        
        assert abs(dev_count - expected_dev) <= 1, f"Cell {key} dev count mismatch: {dev_count} vs {expected_dev}"
        assert abs(test_count - expected_test) <= 1, f"Cell {key} test count mismatch: {test_count} vs {expected_test}"
        assert dev_count + test_count == full_count, f"Cell {key} sum mismatch: {dev_count}+{test_count} != {full_count}"

    full_path = os.path.join(output_dir, "dataset_full.csv")
    dev_path = os.path.join(output_dir, "dataset_dev.csv")
    test_path = os.path.join(output_dir, "dataset_test.csv")
    
    df.to_csv(full_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Generate and save separate fairness set
    fair_df = generate_balanced_fairness_set(n_samples=1500, seed=2026)
    fair_path = os.path.join(output_dir, "dataset_fairness.csv")
    fair_df.to_csv(fair_path, index=False)
    
    print("=" * 60)
    print("SPLIT RECONCILIATION & STRATIFIED CELL COUNTS")
    print("=" * 60)
    print(f"Total Full: {len(df)}, Total Dev: {len(dev_df)}, Total Test: {len(test_df)}")
    print(f"Disjoint IDs: PASSED (Intersection = {len(dev_ids.intersection(test_ids))})")
    print(f"Union == Full: PASSED ({len(dev_ids.union(test_ids))} == {len(full_ids)})")
    
    print("\n--- FULL SPLIT: label x is_non_traditional ---")
    print(pd.crosstab(df["ground_truth_label"], df["is_non_traditional"], margins=True))
    
    print("\n--- DEV SPLIT (20%): label x is_non_traditional ---")
    print(pd.crosstab(dev_df["ground_truth_label"], dev_df["is_non_traditional"], margins=True))
    
    print("\n--- TEST SPLIT (80%): label x is_non_traditional ---")
    print(pd.crosstab(test_df["ground_truth_label"], test_df["is_non_traditional"], margins=True))
    print("=" * 60)
    
    return dev_df, test_df, fair_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for PIM thesis.")
    parser.add_argument("--samples", type=int, default=5000, help="Total number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="experiments/data", help="Output directory")
    args = parser.parse_args()
    
    dataset = generate_profiles(n_samples=args.samples, seed=args.seed)
    save_and_split_dataset(dataset, args.outdir)
