# Formal Preregistration: Multi-Agent Travel Compliance & Demographic Fairness Benchmarking

**Author:** MS AI Thesis Research  
**Date of Preregistration:** October 2026  
**Version:** v5.0 (Draft for Review — Reduced Sample Size Protocol & Full-Sample Baselines)  
**Status:** Preregistered specification prior to execution of LLM benchmark calls on test, fairness, or counterfactual splits.

---

## 1. Study Overview & Objective
This study investigates whether multi-agent LLM architectures (Hierarchical Consensus and Peer-to-Peer Adversarial Debate) reduce wrongful offloading (False Positive Rate) and mitigate algorithmic bias against non-traditional travelers (gig workers, freelancers, informal earners) compared to single-agent baselines and token-matched self-consistency ensembles, within the domain of immigration departure screening.

To balance rigorous statistical power with computational feasibility under verified API rate limits and token quotas, Version 5.0 implements a dual-cohort testing architecture:
1. **Full Test Cohort ($N = 4,000$ profiles):** Evaluates deterministic statutory rules (**S0**) and traditional machine learning baselines (Logistic Regression, Random Forest, XGBoost) across the complete held-out test distribution.
2. **Stratified Core Test Subset ($N = 1,000$ profiles):** Evaluates all multi-agent architectures (**S2, S3**), single-agent baselines (**S1, S1b**), compute-matched self-consistency (**S1c**), deterministic rules (**S0**), and ML baselines under identical prompt-frozen inference constraints.
3. **Fairness Evaluation Cohort ($N = 500$ profiles):** Enriched for non-traditional livelihoods and diverse demographic strata.
4. **Counterfactual Demographic Flip Cohort ($N = 300$ base profiles $\times$ 3 attributes):** Evaluates algorithmic sensitivity to counterfactual perturbations with null-control duplication.

---

## 2. Evaluated Systems & Experimental Controls

All LLM systems operate under identical inference constraints using the same model (`openai/gpt-oss-120b`) on Groq with fixed reasoning effort (`"medium"`) and `max_tokens = 4096`. All prompts are version-controlled in `experiments/prompts/` and frozen under git tag `prompts-frozen-v1`. Experimental code will be tagged `code-frozen-v1` immediately prior to benchmark execution.

| System ID | Name | Architecture | Prompt / Interaction Mode | Temperature | Expected Calls / Profile | Role in Study |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S0** | **Deterministic Rule Engine** | Pure Code Engine (No LLM) | Applies statutory rules from `fia_rules_common.txt` | N/A | 0 calls | Procedural Lower-Bound Reference (evaluated on $N=4,000$ and $N=1,000$) |
| **ML-Base** | **Supervised ML Classifiers** | Scikit-Learn / XGBoost | Tabular gradient boosted trees & linear models | N/A | 0 calls | Non-LLM Tabular Baselines (evaluated on $N=4,000$ and $N=1,000$) |
| **S1** | **Single-Agent Baseline** | Monolithic LLM call | Strict departure compliance prompt | $T = 0.0$ | 1 call | Reference Baseline Only |
| **S1b** | **Single-Agent + Context** | Monolithic LLM call | S1 prompt + mitigating-context instruction block | $T = 0.0$ | 1 call | Primary Comparison (Prompt Control) |
| **S1c** | **Self-Consistency Ensemble** | Majority Voting Ensemble | $k$ repeated stochastic calls of S1; majority vote decision; mean risk score. $k$ calibrated from full dev-pilot means ($N=200$) | $T = 0.7$ | $k$ calls ($k = \text{round}(\bar{T}_{S3}/\bar{T}_{S1})$) | Primary Comparison (Compute Control; dropped from flips) |
| **S2** | **Hierarchical Consensus** | Sequential Multi-Agent Pipeline | Routing: Document Integrity Specialist $\to$ Socio-Economic Specialist $\to$ Consensus Risk Adjudicator | $T = 0.0$ | 3 calls | Primary Comparison |
| **S3** | **Adversarial Debate** | Multi-Turn Debate Pipeline | Multi-round debate: Risk Scorer (Prosecutor) $\leftrightarrow$ Traveler Advocate (up to 3 rounds) $\to$ Final Adjudicator (Judge) | $T = 0.0$ | 3 to 7 calls (empirically ~3.67 calls) | Proposed Primary System |

---

## 3. Calibration of Compute Control Parameter $k$ for S1c

To guarantee a fair token-matched compute control, parameter $k$ for $S_{1c}$ is defined as:
$$k = \text{round}\left(\frac{\text{Mean Total Tokens}(S_3)}{\text{Mean Total Tokens}(S_1)}\right)$$
* **Calibration Protocol:** $k$ is calculated from the **full 200-profile dev pilot** ($N = 200$ complete profiles per system, reporting sample size $n$, sample standard deviation $\text{SD}$, and standard error).
* Preliminary numbers from initial dev-pilot evaluations indicate $S_3$ averages ~3.67 calls/profile (~7,194 tokens) and $S_1$ averages ~1,246 tokens, yielding $k \approx 4$ to $5$. The definitive integer $k$ will be locked upon completion of the full 200-profile dev pilot.

---

## 4. Evaluation Cohorts & Stratified Test Subset Specification

### A. Full Test Cohort ($N = 4,000$)
* **Source:** `experiments/data/dataset_test.csv` (3,200 Genuine, 800 Threats).
* **Evaluations Conducted:**
  - Deterministic Rule Engine (**S0**)
  - Supervised ML Baselines (**Logistic Regression, Random Forest, XGBoost**)
  - Serves as the complete-population reference distribution at \$0.00 compute cost.

### B. Core Stratified Test Subset ($N = 1,000$)
* **Sampling Protocol:** Drawn from `dataset_test.csv` using `StratifiedShuffleSplit` with seed `2026`, jointly stratified across binary ground truth and non-traditional livelihood (`ground_truth_label` $\times$ `is_non_traditional`).
* **Strata Distribution:**
  - Genuine Traditional (`0_0`): **465 profiles**
  - Genuine Non-Traditional (`0_1`): **335 profiles** (Total Genuine = 800, 80.0%)
  - Threat Traditional (`1_0`): **118 profiles**
  - Threat Non-Traditional (`1_1`): **82 profiles** (Total Threats = 200, 20.0%)
* **Artifact Manifest:**
  - Storage Path: `experiments/data/test_subset_1000_ids.json`
  - Seed: `2026`
  - SHA-256 Digest: `6c74a4a2563f0decaea5296ad9975413101d4d4cb86e528ef29e2da7ecc61dc0`
  - Integrity Guarantee: Zero LLM calls have been executed on this cohort; raw test data remains untouched.

### C. Fairness Evaluation Cohort ($N = 500$)
* Enriched for non-traditional livelihoods across informal, freelance, and gig sectors to assess demographic parity and equalized opportunity.

### D. Counterfactual Demographic Flip Cohort ($N = 300$ pairs $\times$ 3 attributes)
* **Attributes Tested:** Gender (Male $\leftrightarrow$ Female), Region (Origin province perturbations), and Nationality Group (Domestic $\leftrightarrow$ Foreign strictly on non-WORK visas).
* **Triplicate Null-Control Design:** For each base profile, systems evaluate:
  1. Original unperturbed profile
  2. Identical unperturbed null-control profile (to measure stochastic baseline flip rate)
  3. Attribute-flipped profile
* **Scope Control:** $S_{1c}$ is excluded from counterfactual flips to conserve budget; systems evaluated are $S_1, S_{1b}, S_2, S_3$ ($300 \times 3 \text{ copies} \times 3 \text{ attributes} = 2,700$ evaluations per system; 10,800 total evaluations).

---

## 5. Statistical Power Analysis & Permutation Noise Floors ($N = 1,000$)

Monte Carlo power simulations (1,000 iterations per effect size, concordance $\rho = 0.85$, $\alpha = 0.05$ one-sided) and sharp null permutation audits (10,000 resamples) establish the statistical precision of the 1,000-profile subset:

### A. Minimum Detectable Effects (at 80% Power)
1. **Paired Difference in False Positive Rate ($\Delta \text{FPR}$ at Recall = 0.70):**
   * Negative class sample size: $N_{\text{genuine}} = 800$.
   * **Minimum Detectable Difference ($\text{MDE}_{80\%}$):** **$\Delta \text{FPR} = 0.036$ (3.6%)**.
   * An observed FPR reduction from 15.0% to 11.4% achieves $\ge 80\%$ power.
2. **Difference in Non-Traditional Bias Gap ($\Delta \text{Gap}_{\text{NT}} = \text{Gap}_A - \text{Gap}_B$):**
   * Strata sample sizes: $N_{\text{trad}} = 465, N_{\text{nt}} = 335$.
   * **Minimum Detectable Difference ($\text{MDE}_{80\%}$):** **$\Delta \text{Gap}_{\text{NT}} = 0.069$ (6.9%)**.

### B. Permutation Noise Floors (10,000 Resamples under Sharp Null $H_0$)
Under the sharp null hypothesis that system outputs are statistically exchangeable:
* **FPR Difference ($\Delta \text{FPR}$):**
  - Standard Deviation: **0.0083** (0.83%)
  - Mean Absolute Difference: **0.0110** (1.10%)
  - 95th Percentile ($p = 0.05$ critical noise floor): **0.0275** (2.75%)
  - 99th Percentile: **0.0350** (3.50%)
* **Non-Traditional Gap Difference ($\Delta \text{Gap}_{\text{NT}}$):**
  - Standard Deviation: **0.0170** (1.70%)
  - Mean Absolute Difference: **0.0221** (2.21%)
  - 95th Percentile ($p = 0.05$ critical noise floor): **0.0545** (5.45%)
  - 99th Percentile: **0.0720** (7.20%)

*Conclusion:* Any observed $\Delta \text{FPR} > 2.75\%$ and $\Delta \text{Gap}_{\text{NT}} > 5.45\%$ decisively clears the 95% permutation noise floor.

---

## 6. Budget, Verified Pricing & Per-Phase Spend Caps

### A. Verified Pricing Source
* **API Provider:** Groq Official Documentation for `openai/gpt-oss-120b`.
* **Prompt (Input) Tokens:** **\$0.15** per 1,000,000 tokens.
* **Completion (Output) Tokens:** **\$0.60** per 1,000,000 tokens.
* **Rate Structure:** Output tokens cost exactly **4.0x** input tokens (\$0.60 vs \$0.15). Token usage tracking logs input, completion, and reasoning tokens separately.

### B. Phase-by-Phase Cost Estimates & Enforced Caps (2x Multiplier)

| Phase | Cohort Description | Systems Evaluated | Total LLM Calls | Estimated Tokens | Estimated Cost | Hard Spend Cap (2x Estimate) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dev Pilot** | 200 dev profiles (gated stage) | $S_1, S_{1b}, S_2, S_3$ | ~1,734 | ~2.35 M | **\$0.70** | **\$2.00** |
| **Phase 1** | Test Set ($N = 1,000$ core subset) | $S_0, S_1, S_{1b}, S_{1c}, S_2, S_3$ | ~11,670 | ~18.88 M | **\$5.15** | **\$10.50** |
| **Phase 2** | Fairness Set ($N = 500$ cohort) | $S_0, S_1, S_{1b}, S_{1c}, S_2, S_3$ | ~5,835 | ~9.44 M | **\$2.58** | **\$5.20** |
| **Phase 3** | Counterfactual Flips (300 pairs $\times$ 3 attrs) | $S_1, S_{1b}, S_2, S_3$ ($S_{1c}$ dropped) | ~18,000 | ~35.60 M | **\$9.84** | **\$20.00** |
| **Full Total**| Complete Thesis Benchmark | All Phases | **~37,239** | **~66.27 M** | **\$18.27** | **\$37.70** |

*Note on Full 4,000 Cohort:* S0 and supervised ML baselines on all 4,000 test profiles incur **\$0.00** API spend and will be executed immediately alongside Phase 1.

---

## 7. Error Handling, Missing Values & Quality Gates

1. **Parse & Network Failure Protocol:**
   - On exhausted retries, responses are recorded strictly as missing values (`decision = null`, `risk_score = null`, `is_parse_failure = true`), never imputed as `OFF-LOAD`.
   - Complete-case reporting (dropping failures) serves as the primary analysis; sensitivity reporting (imputing failures as `OFF-LOAD`) serves as secondary robustness check.
2. **Execution Quality Gates:**
   - Any run where parse failure rate exceeds **2.0%** halts immediately.
   - Any advocate claim hallucination rate exceeding **5.0%** triggers an architectural audit.
3. **Deterministic Advocate Claim Audit:**
   - All traveler advocate mitigating arguments are audited deterministically against passenger profile fields using regex and rule logic (`experiments/harness/claim_auditor.py`), requiring zero LLM calls.
