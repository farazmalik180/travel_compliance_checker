# Data Dictionary: Synthetic Immigration Compliance Dataset

This data dictionary documents every feature generated in `dataset_full.csv`, `dataset_dev.csv`, and `dataset_test.csv`.

## 1. Identifiers & Ground Truth (Latent Variables)

| Column Name | Data Type | Permissible Values / Range | Description & Generation Logic |
| :--- | :--- | :--- | :--- |
| `profile_id` | String | `PIM-SYNTH-00001` to `PIM-SYNTH-05000` | Unique synthetic passenger profile identifier. |
| `ground_truth_label` | Integer | `0` or `1` | **Primary Latent Ground Truth:** `0` = Genuine Traveler, `1` = Genuine Threat (irregular migration, human trafficking risk, absconding intent, or fraudulent pretenses). Drawn from $\text{Bernoulli}(p=0.20)$ prior to feature generation. **Never exposed to agents.** |
| `is_threat` | Integer | `0` or `1` | Duplicate of `ground_truth_label` for readability in evaluation scripts. |
| `latent_intent_score` | Float | `[0.0, 1.0]` | Continuous latent intent score: Genuine drawn from $\text{Beta}(2.5, 6.5)$ (mean ~0.28); Threat drawn from $\text{Beta}(6.5, 2.5)$ (mean ~0.72). Deliberate overlap exists in range $[0.35, 0.65]$. |

## 2. Sensitive Attributes (For Fairness & Demographic Parity Analysis)

| Column Name | Data Type | Permissible Values / Range | Description & Generation Logic |
| :--- | :--- | :--- | :--- |
| `gender` | Categorical | `Male` (62%), `Female` (36%), `Non-binary` (2%) | Self-reported gender. Generated unconditionally to reflect typical regional departure proportions. |
| `age_band` | Categorical | `18-25` (25%), `26-35` (42%), `36-50` (23%), `51+` (10%) | Age bracket of the passenger. |
| `region` | Categorical | `Punjab` (52%), `Sindh` (22%), `Khyber Pakhtunkhwa` (15%), `Balochistan` (6%), `Islamabad/AJK/GB` (5%) | Province or administrative territory of origin within Pakistan. |
| `nationality_group` | Categorical | `Domestic Citizen` (90%), `Dual National` (7%), `Resident Expatriate` (3%) | Citizenship status category. |
| `is_non_traditional` | Integer (Binary) | `0` (Traditional), `1` (Non-Traditional) | **Fairness Subgroup Flag:** Identifies gig economy workers, freelance digital contractors, informal trade workers, and young first-time jobseekers. Crucial for measuring False Positive Rate (FPR) disparity on non-standard socio-economic backgrounds. |

## 3. Observable Profile Features (Exposed to Agents)

| Column Name | Data Type | Permissible Values / Range | Description & Generation Logic |
| :--- | :--- | :--- | :--- |
| `employment_type` | Categorical | `Corporate Salaried`, `Civil Servant`, `Registered Business Owner`, `Gig Worker / Freelancer`, `Informal Self-Employed`, `Contract Worker`, `Student` | Nature of traveler's livelihood. Conditioned on `is_non_traditional` archetype. |
| `profession` | Categorical | Granular job titles (e.g., `Software Developer`, `Medical Doctor`, `Small Shop Owner`, `Ride-Hailing Driver`, etc.) | Specific declared profession. |
| `income_stability` | Categorical | `High`, `Moderate`, `Low`, `Volatile`, `Dependent / Family Supported` | Assessed regularity of monthly income streams. |
| `bank_funds_pkr` | Integer | `50,000` to `15,000,000` PKR | Continuous bank balance equivalent in PKR. Log-normally distributed conditional on background; threats exhibit bimodal distribution (thin funds vs borrowed "show money"). |
| `bank_funds_category` | Categorical | `Low (< 300k PKR)`, `Moderate (300k - 1M PKR)`, `High (1M - 3M PKR)`, `Very High (> 3M PKR)` | Binned financial tier corresponding to immigration officer screening categories. |
| `local_ties_score` | Float | `[5.0, 98.0]` | Continuous index summarizing domestic anchorage (property ownership, immediate family, duration of residence). Normally distributed with higher variance for non-traditional profiles. |
| `passport_history` | Categorical | `Fresh` (0 prior trips), `Experienced` (1+ prior international trips) | Travel history. Fresh passports are common among young genuine tourists/students as well as irregular migrants. |
| `destination` | Categorical | Monitored list (`Cambodia`, `UAE`, `Thailand`, `Malaysia`, `Iraq`, `Kenya`, `Kyrgyzstan`, etc.) vs Standard list (`UK`, `US`, `Germany`, `Turkey`, `Canada`, etc.) | Destination country. Monitored destinations trigger heightened scrutiny under official FIA directives. |
| `visa_category` | Categorical | `VISIT`, `WORK`, `STUDENT`, `BUSINESS` | Stamped visa class on passport. |
| `purpose` | Categorical | Declared travel purpose (e.g., `Tourism / Leisure`, `Visiting Family`, `Overseas Employment Contract`, `Trade Fair`, etc.) | Intended objective of international travel. |
| `has_return_ticket` | Boolean | `True`, `False` | Physical presence of a confirmed outbound return or onward flight booking. |
| `has_hotel_booking` | Boolean | `True`, `False` | Presence of advance hotel accommodation or registered host voucher. |
| `has_financial_proof` | Boolean | `True`, `False` | Physical possession of endorsed foreign currency, international debit/credit cards, or bank statements. |
| `has_protector_stamp` | Boolean | `True`, `False` | Official Bureau of Emigration stamp. Legally required for Pakistani citizens departing on `WORK` visas; `False` for `VISIT`/`STUDENT`/`BUSINESS`. |
| `watchlist_flag` | Categorical | `Clear`, `Potential Name Match`, `Verified Watchlist Alert` | Integrated Border Management System (IBMS) hit. Genuine travelers may encounter false-positive `Potential Name Match` due to common naming conventions. |
| `document_authenticity_score` | Float | `[15.0, 99.0]` | Simulated forensic OCR score of passport security features, watermark integrity, and visa sticker consistency. |
