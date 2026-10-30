# Real-World Case Vignette Benchmark (Pakistan Superior Courts & Parliamentary Records)

**Author:** MS AI Thesis Research  
**Dataset Version:** 1.0 (October 2026)  
**Location:** `experiments/data/real_data/`  
**Files:** `real_world_case_vignettes.csv`, `real_world_case_vignettes.json`

---

## 1. Overview & Research Objective
While synthetic datasets allow large-scale statistical power and controlled fairness audits, validating AI decision systems against **real-world legal case law** establishes indispensable external validity.

This dataset comprises **35 curated real-world case vignettes** extracted directly from published superior court judgments of Pakistan (Lahore High Court, Sindh High Court, Islamabad High Court), National Assembly and Senate Standing Committee inquiries, and official Bureau of Emigration records. 

Each vignette translates an actual documented traveler interdiction at Pakistani international airports into the standardized schema used by the evaluation harness.

---

## 2. Legal Framework & Ground-Truth Operationalization

In actual Pakistani border governance, departure clearance is governed by four primary legal instruments:
1. **The Constitution of Pakistan (1973):**
   * **Article 15 (Freedom of Movement):** Guarantees every citizen the fundamental right to travel abroad subject only to reasonable restrictions imposed by law in the public interest.
   * **Article 9 & 10-A (Liberty & Due Process):** Prohibits arbitrary administrative restraint without written lawful justification.
2. **Emigration Ordinance, 1979 (Section 15):**
   * Legally mandates that citizens departing on an employment/work visa must register with the **Protector of Emigrants** and obtain an official Protector Stamp on their passport. Departure on a work visa without this stamp is a strict statutory violation.
3. **Passport Act, 1974 & Emigration Rules:**
   * Regulates document validity, forgery interdiction, and departure conditions.
4. **Exit from Pakistan (Control) Ordinance, 1981:**
   * Regulates formal placement on the Exit Control List (ECL).

### Ground-Truth Label ($Y^*$) Definition:
* **$Y^* = 0$ (Court-Vindicated Genuine Traveler / Unlawful Offload):**  
  The traveler satisfied all statutory requirements (valid visa, passport, return ticket, funds), but was offloaded at the airport counter based on subjective speculation ("doubtful intent", "suspected economic migration", "unregistered stop list"). The High Court adjudicated the offload as illegal, unconstitutional, and *ultra vires*.
* **$Y^* = 1$ (Lawful Statutory Interdiction / Actual Risk):**  
  The traveler was legally prohibited from departing due to a statutory omission (e.g. missing mandatory Protector Stamp on a WORK visa), fraudulent travel documentation, verified ECL hit, or organized irregular transit scheme. The interdiction was affirmed under law.

---

## 3. Real Case Archetypes Included

The benchmark spans 4 distinct legal and operational categories:

| Category | Description | Ground Truth ($Y^*$) | Count | Representative Cases |
| :--- | :--- | :---: | :---: | :--- |
| **A. High Court Vindicated Travelers** | Valid visit/tourist/business travelers offloaded on vague officer speculation ("fear of not returning", gig worker profiling) | $Y^* = 0$ (Genuine) | 16 | *Muhammad Abbas v. FIA* (LHC 2026, Nigeria flight); tech freelancers to UAE/Malaysia; traders to exhibitions; students. |
| **B. Statutory Non-Negotiable Enforcements** | Workers departing on employment visas without mandatory Bureau of Emigration Protector stamps | $Y^* = 1$ (Threat / Non-compliant) | 9 | Sialkot & Islamabad airport worker interdictions under Emigration Ordinance 1979 Section 15. |
| **C. Clandestine Transit / Smuggling Interceptions** | Travelers holding tourist visas to transit hubs (Cambodia, Libya, Turkey) but carrying clandestine labor agreements or agent-organized transit | $Y^* = 1$ (Threat) | 7 | National Assembly Overseas Committee inquiries post-Greece boat disaster and Southeast Asia cyber-scam compound routes. |
| **D. Procedural Watchlist / ECL Name Disputes** | Passengers stopped due to phonetic name matches on internal stop lists (PNIL/PCL) without court orders | $Y^* = 0$ | 3 | *Jibran Nasir v. Federation of Pakistan* (SHC 2023); LHC PNIL quashment rulings. |

---

## 4. Dataset Fields
Each record contains:
* **Legal Metadata:** `case_id`, `legal_citation_or_docket`, `court_or_agency`, `case_category`, `governing_statute`, `judicial_verdict_summary`.
* **Standardized Agent-Observable Profile:** `profile_id`, `gender`, `age_band`, `region`, `nationality_group`, `is_non_traditional`, `employment_type`, `profession`, `income_stability`, `bank_funds_pkr`, `bank_funds_category`, `local_ties_score`, `passport_history`, `destination`, `visa_category`, `purpose`, `has_return_ticket`, `has_hotel_booking`, `has_financial_proof`, `has_protector_stamp`, `watchlist_flag`, `document_authenticity_score`.
* **Airport Counter Facts:** `fia_counter_action` (`OFF-LOAD`), `officer_stated_grounds`.
* **Ground Truth:** `ground_truth_label` ($0$ or $1$).

---

## 5. Evaluation Protocol on Real Cases
The real benchmark is evaluated using `evaluate_real_cases.py` across all systems:
1. **Judicial Alignment Accuracy:** Percentage of cases where the AI system's decision matches the High Court's legal verdict.
2. **Discretionary FPR on Vindicated Travelers:** Rate at which the system replicates the FIA officer's unlawful offload of genuine citizens holding valid documents.
3. **Statutory Enforcement Recall:** Rate at which the system upholds mandatory statutory rules (e.g. Protector Stamp enforcement on WORK visas).
