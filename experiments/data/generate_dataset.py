"""
Updated Synthetic Dataset Generator for MS AI Thesis.
Ensures:
1. Z directly and smoothly drives observable features via continuous and link functions.
2. is_non_traditional is strictly independent of ground_truth_label (conditional only on age).
3. Realistic overlap so ceiling classifiers (Logistic Regression, Gradient Boosting) achieve realistic AUC (0.80 - 0.90), avoiding trivial separability (AUC > 0.95).
4. Full reproducibility with fixed seed.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit  # Logistic sigmoid function
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

def generate_profiles(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # 1. SENSITIVE & DEMOGRAPHIC ATTRIBUTES (Prior generation)
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
    # Conditional ONLY on age band (e.g. younger travelers more likely gig/freelance)
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
    # 0 = Genuine Traveler (80%), 1 = Genuine Threat (20%)
    p_threat = 0.20
    ground_truth = np.random.binomial(n=1, p=p_threat, size=n_samples)
    
    # Continuous latent risk propensity Z in [0, 1]
    # Genuine: Beta(2.0, 4.5) -> Mean ~0.308, std ~0.16
    # Threat:  Beta(4.2, 2.8) -> Mean ~0.600, std ~0.17
    # Note: Significant distribution overlap between 0.30 and 0.70
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
    
    high_scrutiny_dests = [
        "Cambodia", "UAE", "Thailand", "Malaysia", "Iraq", 
        "Kenya", "Kyrgyzstan", "Qatar", "Senegal", "Libya"
    ]
    standard_dests = [
        "United Kingdom", "United States", "Germany", "Turkey", 
        "Saudi Arabia", "Canada", "Australia", "China", "Japan", "Oman"
    ]
    
    for i in range(n_samples):
        z = z_scores[i]
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
        
        # B. Local Ties Score (Smooth continuous function of Z and non-traditional status)
        # Baseline ties = 82, reduced by Z*45, non-traditional penalty = 12, Gaussian noise sigma=10
        raw_ties = 82.0 - (45.0 * z) - (12.0 * nt) + np.random.normal(0, 10.0)
        ties = float(np.clip(raw_ties, 5.0, 98.0))
        local_ties_scores.append(round(ties, 1))
        
        # C. Bank Funds (PKR) (Log-normal driven by Z and non-traditional status)
        # Log-mean decreases with Z and non-traditional status
        log_mean = 14.2 - (1.6 * z) - (0.6 * nt)
        # Threat show money effect: probability of temporary show money boost increases with Z
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
        
        # D. Travel History (Passport) - Smooth logistic function of Z, age, and non-trad
        p_fresh_logit = -0.8 + (1.6 * z) + (0.7 if age == "18-25" else 0.0) + (0.3 * nt)
        p_fresh = expit(p_fresh_logit)
        passport_histories.append("Fresh" if np.random.rand() < p_fresh else "Experienced")
        
        # E. Destination - Smooth logistic function of Z
        p_high_scrutiny_logit = -1.2 + (2.4 * z)
        p_high_scrutiny = expit(p_high_scrutiny_logit)
        if np.random.rand() < p_high_scrutiny:
            dest = np.random.choice(high_scrutiny_dests)
        else:
            dest = np.random.choice(standard_dests)
        destinations.append(dest)
        
        # Visa category - Smooth distribution conditioned on Z
        # As Z increases, VISIT visa becomes more prevalent (travelers masking labor intent)
        p_visit = np.clip(0.40 + (0.30 * z), 0.20, 0.80)
        p_work = np.clip(0.35 - (0.15 * z), 0.10, 0.50)
        p_student = 0.15 * (1 - z)
        p_business = max(0.01, 1.0 - (p_visit + p_work + p_student))
        v_probs = np.array([p_visit, p_work, p_student, p_business])
        v_probs = v_probs / v_probs.sum()
        v_cat = np.random.choice(["VISIT", "WORK", "STUDENT", "BUSINESS"], p=v_probs)
        visa_categories.append(v_cat)
        
        # Purpose
        if v_cat == "VISIT":
            purp = np.random.choice(["Tourism / Leisure", "Visiting Family / Relatives", "Attending Social Event / Wedding"])
        elif v_cat == "WORK":
            purp = np.random.choice(["Overseas Employment Contract", "Technical Specialist Assignment", "Skilled Labor"])
        elif v_cat == "STUDENT":
            purp = "Undergraduate / Postgraduate Studies"
        else:
            purp = np.random.choice(["Trade Fair / B2B Networking", "Vendor Contract Negotiation", "Client Meeting"])
        purposes.append(purp)
        
        # F. Return Ticket & Hotel Booking - Smooth logistic functions of Z
        if v_cat in ["VISIT", "BUSINESS"]:
            p_ret_logit = 3.2 - (2.6 * z)
            p_hot_logit = 2.9 - (2.4 * z)
        else:
            p_ret_logit = -1.2 - (0.6 * z)
            p_hot_logit = 0.8 - (1.2 * z)
        has_return_tickets.append(bool(np.random.rand() < expit(p_ret_logit)))
        has_hotel_bookings.append(bool(np.random.rand() < expit(p_hot_logit)))
        
        # G. Financial Proof Presentation - Smooth logistic function of Z
        p_fp_logit = 2.6 - (2.2 * z)
        has_financial_proofs.append(bool(np.random.rand() < expit(p_fp_logit)))
        
        # H. Protector Stamp (Bureau of Emigration) - Applicable if WORK visa
        if v_cat == "WORK":
            p_prot_logit = 2.4 - (4.2 * z)
            has_protector_stamps.append(bool(np.random.rand() < expit(p_prot_logit)))
        else:
            has_protector_stamps.append(False)
            
        # I. Watchlist Flag - Smooth function of Z
        # Exact verified alert occurs only at high Z (and we allow evaluation with/without it)
        p_name_match = 0.05 + (0.12 * z)
        p_verified_alert = max(0.0, (z - 0.70) * 0.40)  # only positive if z > 0.70
        p_clear = max(0.0, 1.0 - (p_name_match + p_verified_alert))
        wl_probs = np.array([p_clear, p_name_match, p_verified_alert])
        wl_probs = wl_probs / wl_probs.sum()
        wl = np.random.choice(["Clear", "Potential Name Match", "Verified Watchlist Alert"], p=wl_probs)
        watchlist_flags.append(wl)
        
        # J. Document Authenticity Score - Smooth continuous function of Z
        raw_doc = 94.0 - (32.0 * z) + np.random.normal(0, 8.0)
        doc_score = float(np.clip(raw_doc, 15.0, 99.0))
        document_authenticity_scores.append(round(doc_score, 1))

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
    
    full_path = os.path.join(output_dir, "dataset_full.csv")
    dev_path = os.path.join(output_dir, "dataset_dev.csv")
    test_path = os.path.join(output_dir, "dataset_test.csv")
    
    df.to_csv(full_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Run chi-square test on independence between is_non_traditional and ground_truth_label
    ct = pd.crosstab(df["is_non_traditional"], df["ground_truth_label"])
    chi2, p_val, dof, ex = chi2_contingency(ct)
    print("\n--- INDEPENDENCE VERIFICATION ---")
    print(f"Chi-square test between is_non_traditional and ground_truth_label:")
    print(f"Chi2 statistic: {chi2:.4f}, p-value: {p_val:.4f} (dof={dof})")
    print(f"Independence confirmed (p > 0.05): {p_val > 0.05}\n")
    
    print(f"Saved full dataset ({len(df)} rows) to: {full_path}")
    print(f"Saved dev dataset ({len(dev_df)} rows) to: {dev_path}")
    print(f"Saved test dataset ({len(test_df)} rows) to: {test_path}")
    
    return dev_df, test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for PIM thesis.")
    parser.add_argument("--samples", type=int, default=5000, help="Total number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="experiments/data", help="Output directory")
    args = parser.parse_args()
    
    print(f"Generating {args.samples} synthetic profiles with seed {args.seed}...")
    dataset = generate_profiles(n_samples=args.samples, seed=args.seed)
    save_and_split_dataset(dataset, args.outdir)
